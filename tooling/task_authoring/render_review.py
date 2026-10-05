"""Render + probe harness: S5 of the agentic pipeline. RUNS IN THE CONTAINER (GPU).

Host-side validation proves arithmetic. It cannot see mesh interiors, contact, or what an object
actually looks like once the sim has settled it. This harness closes that gap: it builds one task
config, settles it, probes the live poses, renders every camera the config enables (both external
views plus the wrist camera, when `multi_view` is on) at the authored pose and again after settling,
and writes a single JSON the vision stage reads.

Three properties this file is built around:

**Compute first, look second** (AGENTIC_PIPELINE.md section 4.1). Every number a reviewer could
want -- settled vs authored pose deltas, per-object drift, support heights, contact counts -- is
measured here and written into the JSON. A model shown an image AND the numbers can be asked to
report only what the numbers do not already settle, which is the difference between a review loop
that converges and one that hallucinates.

**Exit codes are not trusted.** Isaac exits 0 on unhandled exceptions and segfaults at teardown on
passing runs, so this prints an explicit verdict and writes the verdict into the JSON. The caller
reads the JSON, never `$?`.

**One simulator per process.** `og.shutdown()` is process-terminal -- there is no second scene to
be had in a live interpreter -- so `--family` does not loop over configs in-process. It spawns one
child per config and reads back the JSON each child wrote. That also bounds the blast radius: a
child that segfaults at teardown takes its own config down, not the remaining 99, and a child that
hangs is killed by `--timeout` instead of stalling the sweep.

Run against a running Slurm allocation (`uv` does not exist in the container; the image's conda env
already has REALM on `PYTHONPATH`):

    ./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py \
        --task_cfg_path REALM_DROID100/<task>/default.yaml --out tmp/droid100/review

    ./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py \
        --family REALM_DROID100 --out tmp/droid100/review
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

# `realm.config.shared` imports only numpy, so the placement constant can be bound to the real
# value instead of restated here -- DROP_HEIGHT is a number that moves placements.
from realm.config.shared import DROP_HEIGHT as DROP_HEIGHT_M

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

#: Per-object thresholds. Exceeded drift is a finding, not automatically a failure: a small
#: settle is expected, a large one means the authored layout was not physically stable.
DRIFT_WARN_M = 0.01
DRIFT_FAIL_M = 0.03
#: An authored z this far below the support means the object starts interpenetrating.
PENETRATION_TOL_M = 0.005
#: When placement cannot pack an object it writes z = support + DROP_HEIGHT (0.10 flat) instead of
#: the authored convention z = support + bbox_z/2 + 0.05, and only logs at ERROR level. The two
#: readings collide for bbox_z = 0.10, so the signature also requires the object to have SETTLED
#: (or nearly) onto the support -- an authored z on the drop plane that is still airborne is a
#: floating bug, not a failed pack. `placement.py`'s own ERROR log names the object; this finding
#: is the durable on-disk record of the same event.
DROP_TOL_M = 0.005


def authored_names(cfg: dict) -> set:
    """Names the task config itself placed, as opposed to the scene's own fixtures.

    `env.cfg["objects"]` is main + target + distractors + immutables *plus* `scene_definition.yaml`'s
    objects, and `scene_setup.apply_scene_fixes_from_cfg` legitimately removes some of the latter
    (`to_remove`) and pins others (`to_fix`). A removed scene fixture is not an authored object that
    went missing, and reporting it as one would fail the config for nothing.
    """
    names = set()
    for role in ("main_objects", "target_objects", "distractors", "immutables"):
        names |= {str(o["name"]) for o in (cfg.get(role) or []) if o.get("name")}
    return names


def probe(env) -> list[dict]:
    """Read the live scene: authored vs settled pose, per object.

    Returns one row per object with the numbers a reviewer needs. Nothing here is model-derived:
    these are read off the prims.

    Both sides of the drift comparison are **scene-frame**, which is the frame REALM itself uses for
    `cfg["objects"][i]["position"]` (`_helpers.set_scene_positions` / `vb_pose._place` both write it
    with `frame="scene"`, and `backfill_object_cfgs` reads it back the same way). Comparing a
    `relative_bbox_position` -- a region-relative offset -- against a world-frame live pose inflates
    every drift by the spawn region's distance from the origin; `Pomaria_1_int/Kitchen_Counter` sits
    at (-4.7, -2.0, 1.05), which would report ~5 m of drift on an object that never moved.
    """
    rows = []
    authored = authored_names(env.cfg)
    for obj_cfg in env.cfg.get("objects", []):
        name = obj_cfg.get("name")
        try:
            obj = env.omnigibson_env.scene.object_registry("name", name)
        except Exception:
            obj = None
        if obj is None:
            rows.append({"name": name, "present": False, "authored": name in authored})
            continue
        pos, ori = obj.get_position_orientation(frame="scene")
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else pos
        pos = np.asarray(pos, dtype=float)
        authored_pos = obj_cfg.get("position") or []
        bbox = obj_cfg.get("bounding_box") or obj_cfg.get("scale") or []
        row = {
            "name": name,
            "present": True,
            "authored": name in authored,
            "category": getattr(obj, "category", None),
            "settled_position": [round(float(v), 5) for v in pos],
            "settled_orientation": [round(float(v), 5) for v in ori],
            "authored_position": [round(float(v), 5) for v in authored_pos]
            if authored_pos else None,
            "bbox": [round(float(v), 5) for v in bbox] if bbox else None,
        }
        if authored_pos and len(authored_pos) == 3:
            row["drift_xy"] = round(float(abs(pos[0] - authored_pos[0])
                                          + abs(pos[1] - authored_pos[1])), 5)
            row["drift_z"] = round(float(pos[2] - authored_pos[2]), 5)
        try:
            row["mass"] = round(float(obj.root_link.mass), 5)
        except Exception:
            pass
        rows.append(row)
    return rows


def stability_findings(rows: list[dict], support_z: float) -> list[dict]:
    """Computed findings from the probe, before any image is shown to a model.

    Only objects the task authored are judged (`authored`, defaulting to True so a hand-built row
    is still judged): scene fixtures the config never placed are reported but not held against the
    config, because `scene_setup` removes and pins some of them deliberately.
    """
    findings = []
    for row in rows:
        judged = row.get("authored", True)
        if not row.get("present"):
            if judged:
                findings.append({
                    "code": "MISSING", "object": row["name"],
                    "reason": "authored in the config but absent from the built scene",
                })
            continue
        drift_xy = row.get("drift_xy")
        drift_z = row.get("drift_z")
        authored_pos = row.get("authored_position")
        # A failed pack is authored at z = support + DROP_HEIGHT and then falls; by probe time its
        # drift reads NEGATIVE, so the +Z signature is the authored height, not the drift. Resting
        # on the support is what distinguishes it from a genuinely floating object authored at the
        # same height (bbox_z = 0.10 makes the two authored poses identical).
        dropped = (authored_pos and support_z is not None and drift_z is not None
                   and abs((authored_pos[2] - support_z) - DROP_HEIGHT_M) <= DROP_TOL_M
                   and drift_z < -DROP_TOL_M)
        if not judged:
            continue
        if dropped:
            findings.append({
                "code": "DROPPED", "object": row["name"],
                "reason": (
                    f"authored {authored_pos[2] - support_z:.4f} m above the support -- placement.py's "
                    f"{DROP_HEIGHT_M} m drop height, which it uses when it cannot find a "
                    f"collision-free slot -- and fell {abs(drift_z):.4f} m to come to rest. The "
                    f"layout has no room for this footprint: re-solve with more clearance or a "
                    f"smaller object."
                ),
            })
        elif drift_xy is not None and drift_xy > DRIFT_FAIL_M:
            findings.append({
                "code": "UNSTABLE", "object": row["name"],
                "reason": (
                    f"settled {drift_xy:.4f} m from its authored XY, past the {DRIFT_FAIL_M} m "
                    f"threshold: the layout is not resting where it was placed"
                ),
            })
        elif drift_xy is not None and drift_xy > DRIFT_WARN_M:
            findings.append({
                "code": "DRIFT", "object": row["name"],
                "reason": f"settled {drift_xy:.4f} m from its authored XY",
            })
        drift_z = row.get("drift_z")
        bbox = row.get("bbox")
        if drift_z is not None and bbox and support_z is not None:
            lowest = row["settled_position"][2] - float(bbox[2]) / 2
            if lowest < support_z - PENETRATION_TOL_M:
                findings.append({
                    "code": "PENETRATION", "object": row["name"],
                    "reason": (
                        f"lower face at z={lowest:.4f} is {support_z - lowest:.4f} m below the "
                        f"support at z={support_z:.4f}"
                    ),
                })
        if row.get("authored_position") and bbox and support_z is not None:
            authored_low = row["authored_position"][2] - float(bbox[2]) / 2
            if authored_low > support_z + 0.10:
                findings.append({
                    "code": "FLOATING", "object": row["name"],
                    "reason": (
                        f"authored lower face sits {authored_low - support_z:.4f} m above the "
                        f"support with nothing holding it up"
                    ),
                })
    return findings


def collect_rgb(obs, prefix: str = "") -> list[tuple[str, object]]:
    """Every (camera-name, array) pair whose obs leaf is an `rgb` image.

    Obs is **nested**, keyed `obs['external']['external_sensor0']['rgb']` and
    `obs[<robot>]['<robot>:<link>:Camera:<i>']['rgb']` -- see `inference.utils.extract_from_obs`. A
    flat scan of the top-level keys for a name starting with `rgb` matches nothing, which is how this
    harness ended up writing zero images for every config.
    """
    found: list[tuple[str, object]] = []
    if isinstance(obs, dict):
        for key, value in obs.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if key == "rgb":
                # Name the frame after the camera, not the leaf.
                found.append((prefix or path, value))
            else:
                found.extend(collect_rgb(value, path))
    return found


def _to_uint8(array):
    array = array.detach().cpu().numpy() if hasattr(array, "detach") else np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype("uint8")
    return array


def render(env, out_dir: Path, stem: str, timestep: str) -> tuple[list[str], list[str]]:
    """Render every configured view to PNG. Returns (paths written, errors).

    Errors are returned, not swallowed: a harness that writes no images and reports no failure is
    indistinguishable from a scene that rendered fine, and the vision stage downstream would then be
    reviewing nothing while reporting a clean pass.
    """
    written, errors = [], []
    # PIL, not cv2: PIL is already a hard dependency of the sim image (realm_logging encodes video
    # through it), whereas opencv is declared nowhere.
    try:
        from PIL import Image
    except ImportError:
        return written, ["PIL is not importable; no images could be written"]
    try:
        import omnigibson as og
        # Sensor buffers are only refreshed on an explicit render under render-on-demand.
        og.sim.render()
    except Exception as error:
        errors.append(f"og.sim.render() failed: {type(error).__name__}: {error}")
    try:
        obs, _ = env.omnigibson_env.get_obs()
    except Exception as error:
        errors.append(f"get_obs failed: {type(error).__name__}: {error}")
        return written, errors

    frames = collect_rgb(obs)
    if not frames:
        errors.append(f"obs carried no rgb leaf; top-level keys were {sorted(obs or {})}")
    for name, array in frames:
        try:
            slug = name.replace("/", "_").replace(":", "_")
            path = out_dir / f"{stem}_{slug}_{timestep}.png"
            Image.fromarray(_to_uint8(array)).save(path)
            written.append(str(path))
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    return written, errors


def review_one(task_cfg_path: str, out_dir: Path, *, robot: str = "DROID_mounted",
               steps: int = 60) -> dict:
    """Build, settle, probe and render ONE task config.

    Process lifetime is the caller's problem, not this function's: `og.shutdown()` is terminal, so a
    function that called it could never be called twice. `main()` therefore gives each config its own
    process. The JSON is written before anything else can go wrong, including a raise in this
    function's own teardown, because that file is the only trustworthy output.
    """
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.environments.perturbations._helpers import settle
    from realm.sim_config import set_sim_config

    stem = Path(task_cfg_path).parent.name
    record: dict = {"task": stem, "task_cfg_path": task_cfg_path, "renders": [],
                    "render_errors": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{stem}.json"

    set_sim_config(robot=robot)
    # `perturbations` is a list, not `perturbation` -- the constructor iterates it, so a bare string
    # was both a TypeError and, before that, silently the wrong shape. `multi_view` is what makes the
    # config's cam2 exist at all: without it `_apply_camera_cfg` deletes the second external sensor,
    # so a "four view" review would have had one view to look at.
    env = RealmEnvironmentDynamic(
        task_cfg_path=task_cfg_path, perturbations=["Default"], robot=robot,
        rendering_mode="rt", no_rendering=False, multi_view=True,
    )
    try:
        env.reset()
        task_dir = out_dir / stem
        task_dir.mkdir(parents=True, exist_ok=True)
        paths, errors = render(env, task_dir, stem, "t000")
        record["renders"] += paths
        record["render_errors"] += errors
        # REALM's own settle: hold the arm at reset_qpos with the gripper open, skipping camera work.
        # Raw `og.sim.step()` leaves the arm unactuated, so it sags into the scene and knocks objects
        # over -- an unstable-layout finding the policy would never actually see.
        settle(env, steps=steps)
        region = getattr(env, "spawn_bbox", None)
        support_z = float(region[4]) if region is not None else None
        rows = probe(env)
        record["objects"] = rows
        record["support_z"] = support_z
        record["settle_steps"] = steps
        record["compute_findings"] = stability_findings(rows, support_z)
        paths, errors = render(env, task_dir, stem, "t060")
        record["renders"] += paths
        record["render_errors"] += errors
        # No images at all is a harness failure, not a clean scene: never let it read as a pass.
        record["no_images"] = not record["renders"]
        hard = {"MISSING", "UNSTABLE", "PENETRATION", "DROPPED"}
        failed = any(finding["code"] in hard for finding in record["compute_findings"])
        record["verdict"] = "FAIL" if (failed or record["no_images"]) else "PASS"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
        record["verdict"] = "ERROR"

    out_json.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return record


def _child_command(target: str, out: Path, args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-u", str(Path(__file__).resolve()),
            "--task_cfg_path", target, "--out", str(out),
            "--robot", args.robot, "--steps", str(args.steps)]


def _read_record(out: Path, target: str) -> dict:
    """The child's JSON is the only output worth trusting; its exit code is not evidence."""
    stem = Path(target).parent.name
    path = out / f"{stem}.json"
    if not path.is_file():
        return {"task": stem, "task_cfg_path": target, "verdict": "NO_JSON",
                "error": f"child wrote no {path.name} (crash, OOM or timeout)"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"task": stem, "task_cfg_path": target, "verdict": "BAD_JSON",
                "error": f"{type(error).__name__}: {error}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_cfg_path", default=None,
                        help="one task config, e.g. REALM_DROID100/<task>/default.yaml")
    parser.add_argument("--family", default=None,
                        help="a task family directory or name; every */default.yaml under it is "
                             "reviewed, each in its own process")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "tmp" / "droid100" / "review")
    parser.add_argument("--robot", default="DROID_mounted")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-config wall clock for --family children")
    args = parser.parse_args(argv)

    if args.task_cfg_path and args.family:
        print("give one of --task_cfg_path or --family, not both", file=sys.stderr)
        return 2

    if args.task_cfg_path:
        record = review_one(args.task_cfg_path, args.out, robot=args.robot, steps=args.steps)
        _report(record)
        return 0

    if args.family:
        base = Path(args.family)
        if not base.is_dir():
            base = PROJECT_ROOT / "realm" / "config" / "tasks" / args.family
        if not base.is_dir():
            print(f"no such family directory: {base}", file=sys.stderr)
            return 2
        targets = sorted(str(path.relative_to(PROJECT_ROOT / "realm" / "config" / "tasks"))
                         for path in base.glob("*/default.yaml"))
        if not targets:
            print(f"no */default.yaml under {base}", file=sys.stderr)
            return 2
        results = []
        for index, target in enumerate(targets, start=1):
            print(f"\n=== [{index}/{len(targets)}] {target} ===", flush=True)
            # One process per config: og.shutdown() is terminal, and a config that segfaults or hangs
            # must not take the other 99 with it.
            try:
                subprocess.run(_child_command(target, args.out, args),
                               timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired:
                print(f"  killed after {args.timeout}s", flush=True)
            record = _read_record(args.out, target)
            results.append(record)
            _report(record)
        _summary(results, args.out)
        return 0

    print("give --task_cfg_path or --family", file=sys.stderr)
    return 2


def _report(record: dict) -> None:
    print(f"VERDICT {record['verdict']}  renders={len(record.get('renders') or [])}", flush=True)
    for finding in record.get("compute_findings") or []:
        print(f"  {finding['code']:<12} [{finding['object']}] {finding['reason']}", flush=True)
    for error in record.get("render_errors") or []:
        print(f"  {'RENDER':<12} {error}", flush=True)
    if record.get("error"):
        print(f"  ERROR {record['error']}", flush=True)


def _summary(results: list[dict], out: Path) -> None:
    tally: dict[str, int] = {}
    for record in results:
        tally[record["verdict"]] = tally.get(record["verdict"], 0) + 1
    print("\n" + "=" * 70, flush=True)
    print("  ".join(f"{verdict}={count}" for verdict, count in sorted(tally.items())), flush=True)
    for record in results:
        if record["verdict"] != "PASS":
            print(f"  {record['verdict']:<9} {record['task']}", flush=True)
    summary = out / "_summary.json"
    summary.write_text(json.dumps({"results": [
        {k: r.get(k) for k in ("task", "task_cfg_path", "verdict", "renders", "render_errors",
                               "error", "support_z")}
        for r in results], "tally": tally}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"summary -> {summary}", flush=True)
    print("Read the JSON, not the exit code: Isaac exits 0 on unhandled exceptions.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
