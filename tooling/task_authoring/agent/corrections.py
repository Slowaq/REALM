"""Content-keyed correction store: the replacement for the rank-keyed `REVIEWED_*` tables.

The four `REVIEWED_*` dictionaries in `generate_realm_droid100.py` address tasks by rank, which is
a position in one DROID frequency ranking. Re-running `select_droid100.py` against a different
DROID sample renumbers every rank, so the accumulated review is silently discarded -- and worse,
`generate_realm_droid100.check_reviewed_alignment` shows that a source which merely *declares* the
reviewed `ranking_id` inherits the overrides without inheriting the review.

This store keys a task by content instead:

    task_id = sha1(instruction + "|" + ranking_id)[:12]

so a re-ranking keeps its review, and a correction is only ever attached to the task it was
authored against. Each record keeps `from` as well as `to`, which lets a later run detect that the
generator no longer makes the rejected choice and retire the correction, and `source` so a
measured fix is distinguishable from a model's judgement call.

The store is advisory to the *generator* and authoritative to the *agent*: a stored correction is
merged into a solved document before validation, never reverse-engineered into the solver.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORRECTIONS_DIR = REPO_ROOT / "tooling" / "task_authoring" / "corrections"

#: Object-field corrections the store may apply. Deliberately excludes anything positional:
#: a stored coordinate would defeat the determinism contract (AGENTIC_PIPELINE.md section 1.4).
APPLICABLE_FIELDS = ("category", "model", "orientation", "rgba", "primitive_type", "extent")

#: Corrections that rewrite the document's declared text rather than an object.
DOCUMENT_FIELDS = ("instruction", "task_type", "instruction_obj_to_replace", "instruction_target_to_replace")


def task_id(instruction: str, ranking_id: str | None) -> str:
    """Stable content key. Re-ranking does not change it; a different instruction does."""
    payload = f"{instruction}|{ranking_id or ''}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_store(corrections_dir: Path | None = None) -> dict[str, dict]:
    """task_id -> record. A malformed file is skipped rather than crashing a 100-task run."""
    directory = corrections_dir if corrections_dir is not None else DEFAULT_CORRECTIONS_DIR
    store: dict[str, dict] = {}
    if not directory.is_dir():
        return store
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        key = str(record.get("task_id") or path.stem)
        store[key] = record
    return store


def find_corrections(document: dict, corrections_dir: Path | None = None) -> dict | None:
    """The store record for this document, matched on its own instruction and ranking_id."""
    provenance = document.get("provenance") or {}
    instruction = str(document.get("instruction", ""))
    ranking_id = provenance.get("ranking_id") or document.get("ranking_id")
    key = str(provenance.get("task_id") or task_id(instruction, ranking_id))
    return load_store(corrections_dir).get(key)


def merge_document(document: dict, *, corrections_dir: Path | None = None) -> list[dict]:
    """Apply every applicable correction to `document` in place. Returns what was applied.

    An object-name correction that names an object this document does not have is reported as
    `skipped` rather than raising: the generator may legitimately no longer produce the rejected
    arrangement, and AGENTIC_PIPELINE.md section 4.5 wants that visible, not fatal.
    """
    record = find_corrections(document, corrections_dir)
    if not record:
        return []
    applied = []
    by_name: dict[str, dict] = {}
    for role in ("main_objects", "target_objects", "distractors", "immutables"):
        for config in document.get(role) or []:
            if isinstance(config, dict) and config.get("name"):
                by_name[str(config["name"])] = config

    for correction in record.get("corrections") or []:
        if not isinstance(correction, dict):
            continue
        entry = {
            "iteration": correction.get("iteration"),
            "source": correction.get("source"),
            "code": correction.get("code"),
            "object": correction.get("object"),
            "action": correction.get("action"),
        }
        to_value = correction.get("to") or {}
        if isinstance(to_value, dict) and any(key in to_value for key in DOCUMENT_FIELDS):
            for key in DOCUMENT_FIELDS:
                if key in to_value:
                    document[key] = to_value[key]
            entry["applied"] = {key: to_value[key] for key in DOCUMENT_FIELDS if key in to_value}
            applied.append(entry)
            continue
        name = correction.get("object")
        target = by_name.get(str(name)) if name else None
        if target is None:
            entry["skipped"] = f"no authored object named {name!r} in this document"
            applied.append(entry)
            continue
        changes = {}
        for field in APPLICABLE_FIELDS:
            if isinstance(to_value, dict) and field in to_value:
                target[field] = to_value[field]
                changes[field] = to_value[field]
        entry["applied"] = changes or None
        applied.append(entry)
    return applied


def record_correction(
    document: dict,
    *,
    code: str,
    action: str,
    obj: str | None = None,
    to: dict | None = None,
    reason: str = "",
    source: str = "agent",
    iteration: int = 1,
    corrections_dir: Path | None = None,
) -> Path:
    """Append one correction to this task's store file, creating it if needed.

    `from` is captured from the live document so a later run can tell whether the generator still
    makes the choice this correction rejects.
    """
    directory = corrections_dir if corrections_dir is not None else DEFAULT_CORRECTIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    provenance = document.get("provenance") or {}
    instruction = str(document.get("instruction", ""))
    ranking_id = provenance.get("ranking_id")
    key = str(provenance.get("task_id") or task_id(instruction, ranking_id))
    path = directory / f"{key}.json"
    if path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
    else:
        record = {}
    record.setdefault("task_id", key)
    record.setdefault("instruction", instruction)
    record.setdefault("ranking_id", ranking_id)
    record.setdefault("corrections", [])

    from_value: dict = {}
    if obj:
        for role in ("main_objects", "target_objects", "distractors", "immutables"):
            for config in document.get(role) or []:
                if isinstance(config, dict) and str(config.get("name")) == str(obj):
                    from_value = {field: config.get(field) for field in APPLICABLE_FIELDS if field in config}
    record["corrections"].append({
        "iteration": iteration,
        "source": source,
        "code": code,
        "object": obj,
        "action": action,
        "from": from_value or None,
        "to": to or {},
        "instruction_after": instruction,
        "reason": reason,
        "applied": date.today().isoformat(),
    })
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
