"""The six generator tools as a shell CLI, so a coding agent can BE the generator.

`run.py` drives the tools either with a scripted offline agent or through the Anthropic API. This
CLI is the third driver: an interactive agent session -- a coding agent or a person -- calls
the same tools one shell command at a time. No API key and no SDK are needed, and the agent gets
exactly the tool surface the API path gets: nothing here accepts a coordinate either.

State lives in one JSON file per instruction (`tmp/droid100/sessions/<task_id>.json`). A session is
rehydrated by REPLAYING the last `propose_layout` arguments through the deterministic solver, so
nothing positional is ever stored or hand-edited between calls.

    python -m tooling.task_authoring.agent.cli prompt                     # the generator brief
    python -m tooling.task_authoring.agent.cli next  data/DROID100_tabletop.json
    python -m tooling.task_authoring.agent.cli start "Put the marker in the cup" --ranking-id R
    python -m tooling.task_authoring.agent.cli call <task_id> search_assets '{"query":"mug","role":"target"}'
    python -m tooling.task_authoring.agent.cli call <task_id> list_scene_regions
    python -m tooling.task_authoring.agent.cli call <task_id> propose_layout '{...roles...}'
    python -m tooling.task_authoring.agent.cli call <task_id> validate_draft
    python -m tooling.task_authoring.agent.cli call <task_id> submit_task '{"decisions":["..."]}'
    python -m tooling.task_authoring.agent.cli call <task_id> report_ungroundable '{"reason":"..."}'
    python -m tooling.task_authoring.agent.cli status data/DROID100_tabletop.json
    python -m tooling.task_authoring.agent.cli export --output realm/config/tasks/REALM_DROID100

`--catalog` / `--dataset` select the asset catalogue exactly as in run.py (default: the committed
`asset_catalog.json`). Every tool result is printed as JSON on stdout; exit code 0 means the CLI
ran, not that the tool succeeded -- read `ok` in the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tooling.task_authoring.agent import catalog as asset_catalog
from tooling.task_authoring.agent import run, tools
from tooling.task_authoring.agent.corrections import DEFAULT_CORRECTIONS_DIR, task_id
from tooling.task_authoring.agent.prompts import GENERATOR_SYSTEM, PROMPT_VERSION

DEFAULT_SESSIONS = run.REPO_ROOT / "tmp" / "droid100" / "sessions"
MODEL_ID = "interactive"
TOOL_NAMES = (
    "search_assets", "list_scene_regions", "propose_layout",
    "validate_draft", "submit_task", "report_ungroundable",
)


def _load_entries(path: Path) -> tuple[list[dict], str | None]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        entries = [{"instruction": str(item["instruction"]), "rank": item.get("rank")}
                   for item in payload.get("tasks", [])]
        return entries, payload.get("ranking_id")
    return [{"instruction": line.strip()} for line in text.splitlines() if line.strip()], None


class Store:
    """One JSON state file per task_id."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def load(self, key: str) -> dict:
        path = self.path(key)
        if not path.is_file():
            raise SystemExit(f"no session {key!r}; run `start` first (sessions live in {self.directory})")
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, state: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path(state["task_id"]).write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")

    def all(self) -> list[dict]:
        if not self.directory.is_dir():
            return []
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.directory.glob("*.json"))]


def _session(state: dict, assets: dict) -> tools.Session:
    session = tools.Session(
        state["instruction"], seed=state.get("seed", 100), ranking_id=state.get("ranking_id"),
        assets_by_category=assets, corrections_dir=DEFAULT_CORRECTIONS_DIR,
    )
    if state.get("last_proposal") is not None:
        # Deterministic replay: same roles + seed + catalogue -> byte-identical document.
        session.propose_layout(**json.loads(json.dumps(state["last_proposal"])))
    return session


def _emit(payload: dict) -> int:
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_prompt(_args, _assets, _info) -> int:
    print(GENERATOR_SYSTEM)
    print(INTERACTIVE_ADDENDUM)
    return 0


def cmd_start(args, assets, info) -> int:
    store = Store(args.sessions)
    key = task_id(args.instruction, args.ranking_id)
    existing = store.path(key)
    if existing.is_file() and not args.restart and store.load(key).get("catalog_fingerprint") == info["fingerprint"]:
        state = store.load(key)
        return _emit({"task_id": key, "resumed": True, "outcome": state.get("outcome"),
                      "note": "session exists; pass --restart to begin again"})
    state = {
        "task_id": key, "instruction": args.instruction, "ranking_id": args.ranking_id,
        "rank": args.rank, "seed": args.seed, "prompt_version": PROMPT_VERSION,
        "catalog_fingerprint": info["fingerprint"], "last_proposal": None, "outcome": None,
        "calls": 0,
    }
    store.save(state)
    classification, resolved = run.classify(args.instruction, assets)
    return _emit({
        "task_id": key,
        "instruction": args.instruction,
        "hint": {"task_type": classification, "nouns": resolved,
                 "note": "a regex guess; you own the decision"},
        "catalog": {"source": info["source"], "categories": len(assets)},
        "next": f"call {key} search_assets ... for each noun, then propose_layout",
    })


def cmd_call(args, assets, info) -> int:
    store = Store(args.sessions)
    state = store.load(args.task_id)
    if state.get("outcome"):
        return _emit({"ok": False, "reason": f"this task is already {state['outcome']['status']}; "
                                              f"`start --restart` to redo it"})
    if state.get("catalog_fingerprint") != info["fingerprint"]:
        return _emit({"ok": False, "reason": "the asset catalogue changed since `start`; `start --restart`"})
    if args.tool not in TOOL_NAMES:
        return _emit({"error": f"unknown tool {args.tool!r}; available: {', '.join(TOOL_NAMES)}"})
    try:
        arguments = json.loads(args.arguments) if args.arguments else {}
    except ValueError as error:
        return _emit({"ok": False, "reason": f"arguments are not valid JSON: {error}"})
    if not isinstance(arguments, dict):
        return _emit({"ok": False, "reason": "arguments must be a JSON object"})

    session = _session(state, assets)
    try:
        result = tools.dispatch(session, args.tool, dict(arguments))
    except TypeError as error:
        result = {"ok": False, "reason": f"bad arguments for {args.tool}: {error}"}
    state["calls"] = int(state.get("calls", 0)) + 1
    if args.tool == "propose_layout" and result.get("ok"):
        state["last_proposal"] = arguments
    if session.submission:
        record = _record(state, session, info)
        state["outcome"] = {"status": "submitted", "record": record}
    elif session.ungroundable:
        state["outcome"] = {"status": "declined", "record": _record(state, session, info)}
    store.save(state)
    return _emit(result)


def _record(state: dict, session: tools.Session, info: dict) -> dict:
    """Same shape as run.generate_one's record, so export/report treat both drivers alike."""
    return {
        "task_id": state["task_id"],
        "instruction": state["instruction"],
        "ranking_id": state.get("ranking_id"),
        "rank": state.get("rank"),
        "prompt_version": PROMPT_VERSION,
        "model": MODEL_ID,
        "catalog_fingerprint": info["fingerprint"],
        "grounded": bool(session.submission),
        "ungroundable": session.ungroundable,
        "validation": (session.submission or {}).get("validation"),
        "document": (session.submission or {}).get("document"),
    }


def cmd_next(args, assets, info) -> int:
    entries, ranking_id = _load_entries(args.instructions)
    ranking_id = args.ranking_id or ranking_id
    store = Store(args.sessions)
    for entry in entries[args.offset:]:
        key = task_id(entry["instruction"], ranking_id)
        path = store.path(key)
        if path.is_file():
            state = store.load(key)
            if state.get("outcome") and state.get("catalog_fingerprint") == info["fingerprint"]:
                continue
        return _emit({
            "instruction": entry["instruction"], "rank": entry.get("rank"), "ranking_id": ranking_id,
            "task_id": key,
            "start": (f"python -m tooling.task_authoring.agent.cli start {json.dumps(entry['instruction'])} "
                      f"--ranking-id {ranking_id} --rank {entry.get('rank')}"),
        })
    return _emit({"done": True, "note": "every instruction has an outcome; run `export`"})


def cmd_status(args, assets, info) -> int:
    store = Store(args.sessions)
    states = store.all()
    counts = {"submitted": 0, "declined": 0, "open": 0, "stale": 0}
    rows = []
    for state in states:
        status = (state.get("outcome") or {}).get("status", "open")
        if state.get("catalog_fingerprint") != info["fingerprint"]:
            status = "stale"
        counts[status] = counts.get(status, 0) + 1
        rows.append(f"{status:9} {state['task_id']}  {state['instruction'][:70]}")
    if args.instructions:
        entries, _ = _load_entries(args.instructions)
        counts["remaining_in_list"] = max(0, len(entries) - counts["submitted"] - counts["declined"])
        # `stale` = made against a different asset catalogue; `next` offers those again.
    if not args.quiet:
        print("\n".join(rows))
    print(json.dumps(counts))
    return 0


def cmd_replay(args, assets, info) -> int:
    """Re-solve every submitted task with the CURRENT solver, keeping the agent's choices.

    A solver fix (clutter rules, sizing, placement) otherwise only reaches tasks generated after it.
    Each submitted session's last propose_layout arguments are replayed deterministically; the new
    document replaces the old one if it validates, and the agent's decisions are carried over. A
    task that no longer validates is reopened so `next` offers it again.
    """
    store = Store(args.sessions)
    updated = reopened = unchanged = 0
    for state in store.all():
        outcome = state.get("outcome") or {}
        if outcome.get("status") != "submitted" or state.get("catalog_fingerprint") != info["fingerprint"]:
            continue
        old = outcome["record"]
        session = _session(state, assets)
        verdict = session.validate_draft() if session.document is not None else {"ok": False, "findings": []}
        if not verdict.get("ok"):
            state["outcome"] = None
            state["reopened_by_replay"] = [f.get("code") for f in verdict.get("findings") or []]
            store.save(state)
            reopened += 1
            print(f"REOPENED {state['task_id']}  {state['instruction'][:60]}  {state['reopened_by_replay']}")
            continue
        decisions = ((old.get("document") or {}).get("provenance") or {}).get("decisions")
        session.submit_task(decisions)
        record = _record(state, session, info)
        if json.dumps(record["document"], sort_keys=True) == json.dumps(old.get("document"), sort_keys=True):
            unchanged += 1
            continue
        state["outcome"] = {"status": "submitted", "record": record}
        store.save(state)
        updated += 1
    print(json.dumps({"updated": updated, "unchanged": unchanged, "reopened": reopened}))
    return 0


def cmd_export(args, assets, info) -> int:
    """Write every submitted config, collapsing ones that solve to the same task, plus a report."""
    store = Store(args.sessions)
    states = [state for state in store.all() if state.get("outcome")]
    stale = [state for state in states if state.get("catalog_fingerprint") != info["fingerprint"]]
    if stale:
        print(f"skipping {len(stale)} session(s) made against a different asset catalogue; "
              f"`next` will offer them again", file=sys.stderr)
    records = [state["outcome"]["record"] for state in states if state not in stale]
    records.sort(key=lambda record: (record.get("rank") is None, record.get("rank") or 0, record["task_id"]))
    seen: dict[tuple, str] = {}
    written, duplicates = [], []
    for record in records:
        if not record["grounded"]:
            continue
        key = run.signature(record["document"])
        if key in seen and not args.keep_duplicates:
            record["duplicate_of"] = seen[key]
            duplicates.append(record["task_id"])
            continue
        seen[key] = record["task_id"]
        if not args.dry_run:
            record["path"] = str(run.write_task(record, args.output))
        written.append(record["task_id"])
    report = {
        "prompt_version": PROMPT_VERSION, "model": MODEL_ID, "catalog": info,
        "grounded": len(written), "duplicate": len(duplicates),
        "declined": sum(1 for record in records if not record["grounded"]),
        "total": len(records), "results": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    verb = "would be written" if args.dry_run else "written"
    print(f"{len(written)} unique tasks {verb} to {args.output}, {len(duplicates)} duplicates, "
          f"{report['declined']} declined; report: {args.report}")
    return 0


INTERACTIVE_ADDENDUM = """
## Driving the tools from a shell

You call each tool as
  python -m tooling.task_authoring.agent.cli call <task_id> <tool> '<json arguments>'
and read the JSON it prints. `start` gives you the task_id. A task ends when submit_task returns
ok:true or you call report_ungroundable; after that, call `next` for the following instruction.
Never edit a YAML file or a session file by hand -- that is the solver's job, and a hand edit is
exactly the non-reproducible number this pipeline exists to prevent.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prompt", help="print the generator brief")

    start = sub.add_parser("start", help="open a session for one instruction")
    start.add_argument("instruction")
    start.add_argument("--ranking-id", default=None)
    start.add_argument("--rank", type=int, default=None)
    start.add_argument("--seed", type=int, default=100)
    start.add_argument("--restart", action="store_true")

    call = sub.add_parser("call", help="call one tool in a session")
    call.add_argument("task_id")
    call.add_argument("tool")
    call.add_argument("arguments", nargs="?", default=None, help="JSON object")

    nxt = sub.add_parser("next", help="the next instruction in a list with no outcome yet")
    nxt.add_argument("instructions", type=Path)
    nxt.add_argument("--ranking-id", default=None)
    nxt.add_argument("--offset", type=int, default=0)

    status = sub.add_parser("status", help="outcomes so far")
    status.add_argument("instructions", type=Path, nargs="?", default=None)
    status.add_argument("--quiet", action="store_true")

    export = sub.add_parser("export", help="write submitted configs (deduplicated) + a report")
    export.add_argument("--output", type=Path, default=run.DEFAULT_OUTPUT)
    export.add_argument("--report", type=Path, default=run.REPO_ROOT / "tmp" / "droid100" / "interactive_report.json")
    export.add_argument("--keep-duplicates", action="store_true")
    export.add_argument("--dry-run", action="store_true")

    sub.add_parser("replay", help="re-solve submitted tasks with the current solver, keeping choices")

    args = parser.parse_args(argv)
    try:
        assets, info = asset_catalog.resolve(dataset=args.dataset, catalog=args.catalog)
    except asset_catalog.CatalogError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    handlers = {
        "prompt": cmd_prompt, "start": cmd_start, "call": cmd_call,
        "next": cmd_next, "status": cmd_status, "export": cmd_export, "replay": cmd_replay,
    }
    return handlers[args.command](args, assets, info)


if __name__ == "__main__":
    raise SystemExit(main())
