"""The vision review stage: S6 findings -> S7 deterministic patches.

The loop is deliberately narrow. A model is shown the render plus every number the harness already
computed, and may report only what those numbers cannot settle. Every finding maps to a NAMED
transform from a closed set, and each transform calls the solver again rather than writing
geometry. That is what keeps the correction step reproducible: given the same finding and the same
seed, the patch is the same patch.

Corrections are persisted to the content-keyed store (see corrections.py), replacing the
rank-keyed REVIEWED_* tables that could not survive a re-ranking.

Bounded at two iterations, the measured number that took an earlier pass from 40/100 to 99/100.
A task still failing after two escalates; it is never silently shipped.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from tooling.task_authoring.agent import layout
from tooling.task_authoring.agent.corrections import DEFAULT_CORRECTIONS_DIR, record_correction
from tooling.task_authoring.agent.prompts import PROMPT_VERSION, REVIEW_SYSTEM
from tooling.task_authoring.validation import validate

#: A model finding maps to one of these. Anything else is dropped: an unrecognized code cannot be
#: applied deterministically, so letting it through would mean hand-editing geometry.
CORRECTION_TRANSFORMS = {
    "ORIENTATION": "make_upright",
    "PENETRATION": "move_apart",
    "BURIED": "move_apart",
    "OFF_SUPPORT": "move_apart",
    "CONTAINMENT": "reduce_size",
    "SCALE": "reduce_size",
    "FLOATING": "move_apart",
    "MISSING": "escalate",
    "UNSTABLE": "move_apart",
    "DRIFT": "move_apart",
}

MAX_ITERATIONS = 2
MODEL = "claude-opus-5"


def build_review_request(record: dict, config_text: str, renders: list[Path]) -> dict:
    """Assemble the model turn: images first, then the config and the computed numbers.

    Every computed number is included so the model can be told, explicitly, not to re-litigate
    what the harness already settled.
    """
    content: list[dict] = []
    for path in renders[:8]:
        try:
            encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": encoded},
        })
    already = record.get("compute_findings") or []
    content.append({
        "type": "text",
        "text": (
            f"Task config:\n```yaml\n{config_text}\n```\n\n"
            f"Pose probe after settling (authored vs settled, per object):\n"
            f"```json\n{json.dumps(record.get('objects') or [], indent=1)}\n```\n\n"
            f"Support surface z: {record.get('support_z')}\n\n"
            f"Already determined computationally -- do NOT report these again:\n"
            f"```json\n{json.dumps(already, indent=1)}\n```\n\n"
            f"Report only physical defects the images show that the numbers above do not settle. "
            f"Answer with JSON only."
        ),
    })
    return {"role": "user", "content": content}


def parse_findings(text: str) -> dict:
    """Extract the JSON verdict from a model reply, tolerating prose or a fenced block."""
    stripped = text.strip()
    if "```" in stripped:
        for chunk in stripped.split("```"):
            candidate = chunk.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                stripped = candidate
                break
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"verdict": "pass", "findings": [], "notes": "unparseable reply; treated as pass"}
    try:
        payload = json.loads(stripped[start:end + 1])
    except ValueError:
        return {"verdict": "pass", "findings": [], "notes": "invalid JSON; treated as pass"}
    if not isinstance(payload, dict):
        return {"verdict": "pass", "findings": [], "notes": "not an object; treated as pass"}
    findings = []
    for item in payload.get("findings") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).upper()
        if code not in CORRECTION_TRANSFORMS:
            # A code outside the closed set cannot be applied deterministically.
            continue
        if not item.get("object"):
            continue
        findings.append({
            "code": code,
            "object": str(item["object"]),
            "reason": str(item.get("reason", ""))[:400],
        })
    verdict = "fail" if findings else "pass"
    return {"verdict": verdict, "findings": findings, "notes": str(payload.get("notes", ""))[:400]}


def apply_correction(document: dict, finding: dict) -> tuple[bool, str]:
    """Apply one finding through a named transform. Returns (changed, description).

    Only the transforms that a solved document can express without new geometry are applied here.
    `swap_asset` and `change_camera` need a re-solve, which the caller drives; `escalate` refuses.
    """
    action = CORRECTION_TRANSFORMS.get(finding["code"])
    name = finding["object"]
    target = None
    for role in ("main_objects", "target_objects", "distractors", "immutables"):
        for config in document.get(role) or []:
            if str(config.get("name")) == name:
                target = config
                break
        if target:
            break
    if target is None:
        return False, f"no authored object named {name!r}"
    if action == "make_upright":
        if target.get("orientation") == [0.0, 0.0, 0.0, 1.0]:
            return False, f"{name!r} is already upright"
        target["orientation"] = [0.0, 0.0, 0.0, 1.0]
        return True, f"set {name!r} upright"
    if action == "reduce_size":
        # One uniform step, never per-axis: per-axis scaling changes the asset's proportions.
        bbox = target.get("bounding_box")
        if not bbox:
            return False, f"{name!r} has no bounding_box to reduce"
        target["bounding_box"] = [round(float(value) * 0.85, 7) for value in bbox]
        if "scale" in target:
            target["scale"] = list(target["bounding_box"])
        return True, f"shrunk {name!r} by one 0.85 uniform step"
    if action == "move_apart":
        # Re-solving is the caller's job: it owns the placed list and the region. Recording the
        # intent is what this layer can do deterministically.
        return False, f"{name!r} needs a re-solve (move_apart): the caller must re-run the solver"
    return False, f"action {action!r} for {finding['code']} requires escalation"


def review_once(
    record: dict,
    document: dict,
    config_text: str,
    renders: list[Path],
    *,
    client=None,
    corrections_dir: Path | None = DEFAULT_CORRECTIONS_DIR,
    iteration: int = 1,
) -> dict:
    """One review iteration: ask, parse, apply, re-validate, record.

    `client` is injected so this is testable without network access.
    """
    if client is None:
        try:
            import anthropic
        except ImportError:
            return {
                "skipped": "the `anthropic` package is not installed; install it or run the "
                           "compute-only pass (render_review.py)",
                "findings": record.get("compute_findings") or [],
            }
        client = anthropic.Anthropic()

    request = build_review_request(record, config_text, renders)
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=4000,
            system=[{"type": "text", "text": REVIEW_SYSTEM}],
            messages=[request],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "findings": []}

    parsed = parse_findings(text)
    applied, skipped = [], []
    for finding in parsed["findings"]:
        changed, description = apply_correction(document, finding)
        if changed:
            record_correction(
                document, code=finding["code"], action=CORRECTION_TRANSFORMS[finding["code"]],
                obj=finding["object"], to=None, reason=finding["reason"],
                source="vision", iteration=iteration, corrections_dir=corrections_dir,
            )
            applied.append({"finding": finding, "result": description})
        else:
            skipped.append({"finding": finding, "result": description})

    report = validate(document, task=str(record.get("task", "")))
    return {
        "iteration": iteration,
        "prompt_version": PROMPT_VERSION,
        "verdict": parsed["verdict"],
        "notes": parsed["notes"],
        "applied": applied,
        "skipped": skipped,
        "validation_after": report.as_dict(),
        "clean": report.ok,
    }


def review_and_patch(
    record: dict,
    document: dict,
    config_text: str,
    renders: list[Path],
    *,
    client=None,
    corrections_dir: Path | None = DEFAULT_CORRECTIONS_DIR,
) -> dict:
    """Bounded loop: at most MAX_ITERATIONS, stopping early once validation stays clean."""
    history = []
    for iteration in range(1, MAX_ITERATIONS + 1):
        result = review_once(
            record, document, config_text, renders,
            client=client, corrections_dir=corrections_dir, iteration=iteration,
        )
        history.append(result)
        if result.get("error") or result.get("skipped"):
            break
        if result.get("verdict") == "pass" or not result.get("applied"):
            break
        if not result.get("clean"):
            # A patch that silences a vision finding while breaking a geometric rule is the loop's
            # characteristic failure mode; stop rather than stack a second patch on a broken draft.
            break
    return {
        "iterations": len(history),
        "converged": bool(history and history[-1].get("clean")),
        "escalate": not bool(history and history[-1].get("clean")),
        "history": history,
    }


def config_text(document: dict) -> str:
    import yaml

    return yaml.safe_dump(document, sort_keys=False, width=100)


__all__ = [
    "CORRECTION_TRANSFORMS", "MAX_ITERATIONS", "apply_correction", "build_review_request",
    "config_text", "parse_findings", "review_and_patch", "review_once", "layout",
]
