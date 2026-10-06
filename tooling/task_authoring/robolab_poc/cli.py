"""Shell driver for the RoboLab-style scene PoC. Every command prints JSON on stdout.

    python -m tooling.task_authoring.robolab_poc.cli regions
    python -m tooling.task_authoring.robolab_poc.cli catalog [--dataset PATH]
    python -m tooling.task_authoring.robolab_poc.cli search mug
    python -m tooling.task_authoring.robolab_poc.cli solve SPEC.json [--seed N]
    python -m tooling.task_authoring.robolab_poc.cli emit SPEC.json [SPEC.json ...] [--seed N] [--force]
    python -m tooling.task_authoring.robolab_poc.cli plot SPEC.json [--seed N]

`solve` is a dry run. `emit` writes realm/config/tasks/ROBOLAB_POC/<name>/default.yaml plus a
poc_provenance.json sidecar, and refuses specs with error findings. `plot` writes a top-down PNG,
drawn from the robot's point of view, to tmp/robolab_poc/<name>.png.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from tooling.task_authoring.robolab_poc import catalog as catalog_mod
from tooling.task_authoring.robolab_poc.frames import find_region, usable_regions
from tooling.task_authoring.robolab_poc.scene import SPATIAL, PHYSICAL, SpecError, solve

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY = "ROBOLAB_POC"
DEFAULT_OUT = REPO_ROOT / "realm" / "config" / "tasks" / FAMILY
PLOT_DIR = REPO_ROOT / "tmp" / "robolab_poc"
CAMERA_EXTRINSICS = REPO_ROOT / "realm" / "config" / "env" / "external_sensors" / "camera_extrinsics_droid_realm.yaml"


def _print(payload) -> None:
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _load_spec(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seed(args, spec: dict) -> int:
    return args.seed if args.seed is not None else int(spec.get("seed", 0))


def cmd_regions(args) -> int:
    _print({
        "relations": sorted(SPATIAL) + list(PHYSICAL),
        "relation_meaning": {
            "left-of / right-of": "the robot's left / right",
            "in-front-of": "closer to the robot than the reference",
            "behind": "farther from the robot than the reference",
            "in": "starts inside `container` (dropped from just above its rim)",
            "on": "starts on top of `support`",
        },
        "regions": [{
            "id": frame.id,
            "width_m": round(frame.region["width"], 3),
            "depth_m": round(frame.region["depth"], 3),
            "robot_yaw_deg": frame.yaw,
            "solver_bounds_forward_left": [round(v, 3) for v in frame.solver_bounds()],
        } for frame in usable_regions()],
    })
    return 0


def cmd_catalog(args) -> int:
    built = catalog_mod.build_catalog(dataset=Path(args.dataset) if args.dataset else None)
    out = Path(args.out)
    out.write_text(json.dumps(built, indent=1) + "\n", encoding="utf-8")
    _print({"written": str(out), "source": built["source"], "assets": len(built["assets"]),
            "categories": built["categories"], "fingerprint": built["fingerprint"]})
    return 0


def cmd_search(args) -> int:
    hits = catalog_mod.search(catalog_mod.load_catalog(Path(args.catalog)), " ".join(args.query))
    _print({"query": " ".join(args.query), "hits": hits,
            "note": "no hit means the category does not exist; do not invent one" if not hits else None})
    return 0


def _solve(path: str, args):
    spec = _load_spec(path)
    seed = _seed(args, spec)
    result = solve(spec, catalog_mod.load_catalog(Path(args.catalog)), seed=seed, margin=args.margin,
                   attempts=args.attempts)
    return spec, result.provenance.get("seed", seed), result


def cmd_solve(args) -> int:
    try:
        _, seed, result = _solve(args.spec, args)
    except SpecError as error:
        _print({"ok": False, "error": f"spec: {error}"})
        return 1
    _print({**result.summary(), "seed": seed, "attempts": result.provenance.get("attempts")})
    return 0 if result.ok else 1


def cmd_emit(args) -> int:
    poses = None
    reports, status = [], 0
    for path in args.specs:
        try:
            spec, seed, result = _solve(path, args)
        except SpecError as error:
            reports.append({"spec": path, "ok": False, "error": f"spec: {error}"})
            status = 1
            continue
        if not result.ok:
            reports.append({"spec": path, **result.summary()})
            status = 1
            continue
        task_dir = Path(args.out) / spec["name"]
        if task_dir.exists() and not args.force:
            reports.append({"spec": path, "ok": False, "error": f"{task_dir} exists; pass --force to replace"})
            status = 1
            continue
        if args.cameras:
            names = args.cameras.split(",")
            result.document["camera_extrinsics"] = {"cam1": names[0], "cam2": names[1]}
            result.provenance["camera_extrinsic_sources"] = names
        else:
            from tooling.task_authoring.authoring import load_camera_extrinsics
            from tooling.task_authoring.generate_realm_droid100 import sample_camera_pair

            poses = poses if poses is not None else load_camera_extrinsics(CAMERA_EXTRINSICS)
            pair = sample_camera_pair(poses, random.Random(seed + 1))
            result.document["camera_extrinsics"] = {
                key: {"pos": value["pos"], "rot": value["rot"]} for key, value in pair.items()}
            result.provenance["camera_extrinsic_sources"] = [pair["cam1"]["source"], pair["cam2"]["source"]]
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "default.yaml").write_text(
            yaml.safe_dump(result.document, sort_keys=False, width=120), encoding="utf-8")
        (task_dir / "poc_provenance.json").write_text(
            json.dumps({**result.provenance, "findings": result.findings, "layout": result.layout}, indent=2) + "\n",
            encoding="utf-8")
        cfg_path = f"{FAMILY}/{spec['name']}/default.yaml"
        reports.append({"spec": path, "ok": True, "task_cfg_path": cfg_path,
                        "findings": result.findings, "layout": result.layout})
    _print({"results": reports})
    return status


def cmd_plot(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    try:
        spec, seed, result = _solve(args.spec, args)
    except SpecError as error:
        _print({"ok": False, "error": f"spec: {error}"})
        return 1
    if result.document is None:
        _print(result.summary())
        return 1
    frame = find_region(spec.get("region"))
    a0, a1, b0, b1 = frame.solver_bounds(inset=0.0)
    fig, ax = plt.subplots(figsize=(5, 6))
    # Robot's view: forward is up, the robot's left is on the left of the image.
    ax.add_patch(Rectangle((-b1, a0), b1 - b0, a1 - a0, fill=False, lw=1.5, ls="--", color="0.4"))
    colors = {"main": "tab:red", "target": "tab:blue", "source": "tab:purple", "distractor": "0.55"}
    for name, item in result.layout.items():
        a, b = item["robot_frame_forward_left"]
        bx, by = item["bounding_box"][:2]
        world = (by, bx) if item["yaw_deg"] == 90 else (bx, by)
        fa, fb = frame.solver_footprint(world)
        ax.add_patch(Rectangle((-b - fb / 2, a - fa / 2), fb, fa, alpha=0.35, color=colors[item["role"]]))
        ax.annotate(f"{name}\n({item['role']})", (-b, a), ha="center", va="center", fontsize=7)
    ax.annotate("robot", (-(b0 + b1) / 2, a0 - 0.06), ha="center", fontsize=8)
    ax.set_xlim(-b1 - 0.05, -b0 + 0.05)
    ax.set_ylim(a0 - 0.1, a1 + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("robot's left  <-  lateral (m)  ->  robot's right")
    ax.set_ylabel("forward from robot base (m)")
    bad = [f["code"] for f in result.findings if f["severity"] == "error"]
    ax.set_title(f"{spec['name']}\n{frame.id}, seed {seed}" + (f"\nERRORS: {', '.join(bad)}" if bad else ""), fontsize=8)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else PLOT_DIR / f"{spec['name']}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    _print({"written": str(out), "ok": result.ok, "findings": result.findings})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default=str(catalog_mod.DEFAULT_CATALOG))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("regions", help="usable support regions and the relation vocabulary")
    build = sub.add_parser("catalog", help="(re)build catalog.json")
    build.add_argument("--dataset", default=None, help="OmniGibson behavior-1k-assets dir (lab machine)")
    build.add_argument("--out", default=str(catalog_mod.DEFAULT_CATALOG))
    find = sub.add_parser("search", help="find asset categories")
    find.add_argument("query", nargs="+")

    for name, helptext in (("solve", "dry run"), ("plot", "top-down PNG")):
        cmd = sub.add_parser(name, help=helptext)
        cmd.add_argument("spec")
        cmd.add_argument("--seed", type=int, default=None)
        cmd.add_argument("--margin", type=float, default=0.015)
        cmd.add_argument("--attempts", type=int, default=50, help="seeds to try (seed, seed+1, ...)")
        if name == "plot":
            cmd.add_argument("--out", default=None)
    emit = sub.add_parser("emit", help="write task configs")
    emit.add_argument("specs", nargs="+")
    emit.add_argument("--seed", type=int, default=None)
    emit.add_argument("--margin", type=float, default=0.015)
    emit.add_argument("--attempts", type=int, default=50, help="seeds to try (seed, seed+1, ...)")
    emit.add_argument("--out", default=str(DEFAULT_OUT))
    emit.add_argument("--force", action="store_true")
    emit.add_argument("--cameras", default=None, help="NAME1,NAME2 from camera_extrinsics.yaml instead of sampling")

    args = parser.parse_args(argv)
    handlers = {"regions": cmd_regions, "catalog": cmd_catalog, "search": cmd_search,
                "solve": cmd_solve, "emit": cmd_emit, "plot": cmd_plot}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
