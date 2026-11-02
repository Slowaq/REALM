"""Vision review of rendered tasks: an image critic for the generating agent, without an API key.

The render step (render_review.py) measures what numbers can: drift, sinking, tipping, leaving a
container, grasp height, visibility pixels and contrast. What only an image settles -- a hidden or
badly framed object, a dark object on a dark table, a wrong-looking scale, a scene that does not show
what the instruction says, a task that cannot be done as shown -- is judged here by a model that
looks at the settled camera images, in the same style as cli.py: a coding agent (or a person) is
the critic, one shell command at a time.

The critic never edits geometry. A failed review REOPENS the task's session with the findings
attached, and the generating agent fixes it through its own tools (another model or size, the next
camera pair, a different region, or declining). The solver and validator then judge the fix as
they judge everything else, so a review can never smuggle a hand-placed coordinate into a config.
A task that fails review MAX_ITERATIONS times is escalated to a person, never shipped.

On the GPU machine, after render_review.py has written its per-task JSONs and images:

    python -m tooling.task_authoring.agent.review_cli prompt --review-dir tmp/droid100/review_v7/droid100
    #   -> the critic brief, then per task: settled image paths, measured numbers, the record command
    python -m tooling.task_authoring.agent.review_cli record <task_folder> --render-id review_v7/droid100 \
        '{"verdict":"pass","findings":[]}'
    python -m tooling.task_authoring.agent.review_cli status

Verdicts land in tmp/droid100/vision_review/<task_folder>.json. On the machine that holds the
generator sessions (the laptop), after bringing those files over (or straight from a git ref):

    python -m tooling.task_authoring.agent.review_cli ingest --git-ref origin/scrap/review-v3-results
    python -m tooling.task_authoring.agent.cli status          # reopened tasks show as open
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

from tooling.task_authoring.agent.prompts import REVIEW_PROMPT_VERSION, REVIEW_SYSTEM
from tooling.task_authoring.agent.review import CORRECTION_TRANSFORMS, MAX_ITERATIONS, parse_findings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VERDICTS = REPO_ROOT / "tmp" / "droid100" / "vision_review"
DEFAULT_SESSIONS = REPO_ROOT / "tmp" / "droid100" / "sessions"
TASKS_ROOT = REPO_ROOT / "realm" / "config" / "tasks"
VERDICTS_PATH_IN_REPO = "tmp/droid100/vision_review"


# --------------------------------------------------------------------------------------- critic side
def settled_images(record: dict) -> list[str]:
    """The renders of the LAST timestep (after the settle) -- what the policy sees first."""
    renders = [str(path) for path in record.get("renders") or []]
    stamps = sorted({Path(path).stem.rsplit("_", 1)[-1] for path in renders})
    if not stamps:
        return []
    last = stamps[-1]
    return [path for path in renders if Path(path).stem.endswith(f"_{last}")]


def measured_summary(record: dict) -> dict:
    """The numbers the critic must not re-litigate, trimmed to what bears on the image."""
    main_names = set((record.get("visibility") or {}).keys())
    objects = []
    for row in record.get("objects") or []:
        if not row.get("authored", True):
            continue
        low, high = row.get("settled_aabb_low"), row.get("settled_aabb_high")
        objects.append({
            "name": row.get("name"), "role": row.get("role"), "category": row.get("category"),
            "main": row.get("name") in main_names,
            "settled_size_m": ([round(h - lo, 3) for h, lo in zip(high, low)] if low and high else None),
            "drift_xy": row.get("drift_xy"),
        })
    return {
        "verdict_from_numbers": record.get("verdict"),
        "findings_from_numbers": record.get("compute_findings") or [],
        "main_object_visibility": record.get("visibility") or {},
        "objects": objects,
    }


def task_brief(record: dict, render_id: str = "unnamed") -> dict:
    config_path = TASKS_ROOT / str(record.get("task_cfg_path", ""))
    instruction = task_type = None
    try:
        import yaml

        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        instruction, task_type = document.get("instruction"), document.get("task_type")
    except (OSError, ImportError, ValueError):
        pass
    task = str(record.get("task"))
    return {
        "task": task,
        "instruction": instruction,
        "task_type": task_type,
        "config": str(config_path.relative_to(REPO_ROOT)) if config_path.is_file() else None,
        "images": settled_images(record),
        "measured": measured_summary(record),
        "record_with": (f"python -m tooling.task_authoring.agent.review_cli record {task} --render-id {render_id} "
                        f"'{{\"verdict\": \"pass\", \"findings\": [], \"notes\": \"\"}}'"),
    }


def _records(review_dir: Path, tasks: list[str] | None) -> list[dict]:
    records = []
    for path in sorted(Path(review_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if tasks and record.get("task") not in tasks:
            continue
        records.append(record)
    return records


def render_name(review_dir: Path) -> str:
    """`tmp/droid100/review_v7/droid100` -> `review_v7/droid100`: names the render a verdict judged."""
    return "/".join(Path(review_dir).resolve().parts[-2:])


def cmd_prompt(args) -> int:
    records = _records(args.review_dir, args.task)
    render_id = args.render_id or render_name(args.review_dir)
    if args.only_unreviewed:
        records = [r for r in records
                   if _previous(args.verdicts / f"{r.get('task')}.json").get("render_id") != render_id]
    print(REVIEW_SYSTEM)
    print(f"How to work: for each task below, open every image path with your image reader, judge the "
          f"scene, then run its `record_with` command with your JSON verdict. A verdict with a code "
          f"outside the closed set is refused. {len(records)} task(s).\n")
    for record in records:
        print(json.dumps(task_brief(record, render_id), indent=1, default=str))
        print()
    return 0


def _previous(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _refuse(reason: str) -> int:
    print(json.dumps({"ok": False, "reason": reason, "codes": sorted(CORRECTION_TRANSFORMS)}))
    return 2


def cmd_record(args) -> int:
    # Strict where the API path is lenient: an interactive critic can retry, so an unreadable verdict
    # or an unknown code is refused (nothing written) instead of silently becoming a pass.
    raw = _json_object(args.verdict)
    if raw is None:
        return _refuse("the verdict is not a JSON object")
    payload = json.loads(raw)
    unknown = sorted({str(f.get("code", "")).upper() for f in payload.get("findings") or []
                      if isinstance(f, dict) and str(f.get("code", "")).upper() not in CORRECTION_TRANSFORMS})
    if unknown:
        return _refuse(f"codes outside the closed set: {', '.join(unknown)}")
    if any(isinstance(f, dict) and not f.get("object") for f in payload.get("findings") or []):
        return _refuse("every finding needs the object it is about")
    parsed = parse_findings(raw)
    if str(payload.get("verdict", "")).lower() == "fail" and not parsed["findings"]:
        return _refuse("a fail needs at least one finding")
    args.verdicts.mkdir(parents=True, exist_ok=True)
    path = args.verdicts / f"{args.task}.json"
    previous = _previous(path)
    # A new render of a reopened task is a new iteration; re-recording the same render (a corrected
    # verdict) is not. The render id is required so a re-review can never pass for the old one.
    render_id = args.render_id
    iteration = int(previous.get("iteration", 0)) + (0 if previous and previous.get("render_id") == render_id else 1)
    task_id = None
    config = TASKS_ROOT / "REALM_DROID100" / args.task / "default.yaml"
    try:
        import yaml

        task_id = ((yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("provenance") or {}).get("task_id")
    except (OSError, ImportError, ValueError):
        pass
    verdict = {
        "task": args.task, "task_id": task_id, "verdict": parsed["verdict"],
        "findings": parsed["findings"], "notes": parsed["notes"], "iteration": max(1, iteration),
        "render_id": render_id, "prompt_version": REVIEW_PROMPT_VERSION, "critic": args.critic,
        "date": date.today().isoformat(),
    }
    path.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "written": str(path), "verdict": parsed["verdict"],
                      "findings": len(parsed["findings"]), "iteration": verdict["iteration"]}))
    return 0


def _json_object(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        json.loads(text[start:end + 1])
    except ValueError:
        return None
    return text[start:end + 1]


def _verdict_files(args) -> list[dict]:
    if getattr(args, "git_ref", None):
        listing = subprocess.run(
            ["git", "--no-optional-locks", "ls-tree", "-r", "--name-only", args.git_ref, VERDICTS_PATH_IN_REPO],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False)
        out = []
        for name in listing.stdout.split():
            if name.endswith(".json"):
                shown = subprocess.run(["git", "--no-optional-locks", "show", f"{args.git_ref}:{name}"],
                                       capture_output=True, text=True, cwd=REPO_ROOT, check=False)
                try:
                    out.append(json.loads(shown.stdout))
                except ValueError:
                    continue
        return out
    return [_previous(path) for path in sorted(Path(args.verdicts).glob("*.json"))]


def cmd_status(args) -> int:
    verdicts = _verdict_files(args)
    counts = {"pass": 0, "fail": 0}
    by_code: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.get("verdict", "pass")] = counts.get(verdict.get("verdict", "pass"), 0) + 1
        for finding in verdict.get("findings") or []:
            by_code[finding["code"]] = by_code.get(finding["code"], 0) + 1
        if verdict.get("verdict") == "fail":
            print(f"FAIL {verdict.get('task')}  " + "; ".join(
                f"{f['code']}[{f['object']}]" for f in verdict.get("findings") or []))
    print(json.dumps({"reviewed": len(verdicts), **counts, "codes": by_code}))
    return 0


# --------------------------------------------------------------------------------------- author side
def cmd_ingest(args) -> int:
    """Reopen every session whose task failed review, with the findings attached for the author."""
    sessions = Path(args.sessions)
    reopened = escalated = passed = missing = 0
    for verdict in _verdict_files(args):
        task_id = verdict.get("task_id")
        path = sessions / f"{task_id}.json"
        if not task_id or not path.is_file():
            missing += 1
            print(f"NO SESSION {verdict.get('task')} (task_id {task_id})")
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        history = state.setdefault("reviews", [])
        if any(h.get("render_id") == verdict.get("render_id") and h.get("iteration") == verdict.get("iteration")
               for h in history):
            continue  # already ingested
        history.append({k: verdict.get(k) for k in ("iteration", "render_id", "verdict", "findings", "notes",
                                                    "critic", "date", "prompt_version")})
        if verdict.get("verdict") != "fail":
            passed += 1
        elif sum(1 for h in history if h.get("verdict") == "fail") >= MAX_ITERATIONS:
            state.pop("review_findings", None)
            state["outcome"] = {"status": "escalated", "record": (state.get("outcome") or {}).get("record"),
                                "reason": f"failed vision review {MAX_ITERATIONS} times; needs a person"}
            escalated += 1
            print(f"ESCALATED {verdict.get('task')}")
        else:
            state["outcome"] = None
            state["review_findings"] = verdict.get("findings") or []
            state.pop("review_seen", None)
            reopened += 1
            print(f"REOPENED {verdict.get('task')}  " + "; ".join(
                f"{f['code']}[{f['object']}]" for f in verdict.get("findings") or []))
        path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "reopened": reopened, "escalated": escalated, "no_session": missing}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    sub = parser.add_subparsers(dest="command", required=True)
    prompt = sub.add_parser("prompt", help="critic brief + per-task images and measured numbers")
    prompt.add_argument("--review-dir", type=Path, required=True, help="render_review.py --out directory")
    prompt.add_argument("--task", action="append", help="limit to these task folders (repeatable)")
    prompt.add_argument("--only-unreviewed", action="store_true")
    prompt.add_argument("--render-id", default=None, help="name of this render (default: last two path parts)")
    record = sub.add_parser("record", help="store one task's verdict")
    record.add_argument("task")
    record.add_argument("verdict", help='JSON: {"verdict": ..., "findings": [...], "notes": ...}')
    record.add_argument("--render-id", required=True, help="which render was judged, e.g. review_v7/droid100")
    record.add_argument("--critic", default="interactive")
    status = sub.add_parser("status", help="summary of stored verdicts")
    status.add_argument("--git-ref", default=None)
    ingest = sub.add_parser("ingest", help="reopen sessions whose task failed review")
    ingest.add_argument("--git-ref", default=None, help="read verdicts from this ref instead of --verdicts")
    ingest.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    args = parser.parse_args(argv)
    return {"prompt": cmd_prompt, "record": cmd_record, "status": cmd_status, "ingest": cmd_ingest}[args.command](args)


__all__ = ["CORRECTION_TRANSFORMS", "main", "settled_images", "task_brief"]

if __name__ == "__main__":
    raise SystemExit(main())
