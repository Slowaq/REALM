"""Select the 100 most common groundable DROID tabletop instructions.

This is the upstream stage of the REALM_DROID100 pipeline. It reads the DROID
`language_instruction*` columns, ranks phrasings by episode frequency, and keeps only
those `generate_realm_droid100.py` can actually ground: the instruction must infer a
supported REALM task type and expose enough `CONCEPT_PATTERN` terms for that type.

The emitted JSON is the `--source` consumed by `generate_realm_droid100.py`:

    {"tasks": [{"rank": 1, "instruction": "...", "task_type": "put", "episodes": 8}, ...]}

Candidates are validated through the generator's own `concepts()` so a phrasing that
would raise mid-generation is dropped here instead of aborting a 100-task run.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tooling.task_authoring.generate_realm_droid100 import concepts


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODES = REPO_ROOT / "data" / "droid_1.0.1" / "chunk-000"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "droid" / "DROID100_tabletop.json"
INSTRUCTION_COLUMNS = ("language_instruction", "language_instruction_2", "language_instruction_3")

# Fixture verbs and nouns REALM has no movable asset or task type for.
FIXTURE_PATTERN = re.compile(
    r"\b(oven|toaster|microwave|coffee\s?maker|coffeemaker|dishwasher|fridge|refrigerator|"
    r"faucet|sink\s+handle|stove|burner|kettle|blender|door|cabinet|drawer|button|switch|"
    r"lever|knob|light|lamp|keyboard|laptop|mouse|chair|curtain|blind)\b",
    re.IGNORECASE,
)
# Multi-stage or non-manipulation phrasings the single-main contract cannot represent.
UNSUPPORTED_PATTERN = re.compile(
    r"\b(clean|wipe|sweep|pour|fold|open|close|press|turn\s+(?:on|off)|plug|unplug|"
    r"throw|trash|organize|tidy|sort|assemble)\b",
    re.IGNORECASE,
)


def infer_task_type(instruction: str) -> str | None:
    """Apply the documented REALM interpretation rules; return None when unsupported."""
    lowered = instruction.lower()
    if re.search(r"\b(?:stack|on top of|onto)\b", lowered):
        return "stack"
    if re.search(r"\b(?:put|place|move|drop)\b", lowered) and re.search(r"\b(?:in|into|inside)\b", lowered):
        return "put"
    if re.search(r"\b(?:pick|grab|lift|take|remove)\b", lowered):
        return "pick"
    if re.search(r"\b(?:rotate|reorient|flip)\b", lowered):
        return "rotate"
    if re.search(r"\bpush\b", lowered):
        return "push"
    if re.search(r"\b(?:put|place|move|drop)\b", lowered) and re.search(r"\bon\b", lowered):
        return "stack"
    return None


def normalize(instruction: str) -> str:
    return re.sub(r"\s+", " ", instruction).strip().rstrip(".")


def count_instructions(episodes: Path) -> Counter[str]:
    """Count episodes mentioning each phrasing (deduplicated within an episode)."""
    import pyarrow.parquet as pq

    counts: Counter[str] = Counter()
    for path in sorted(episodes.glob("*.parquet")):
        try:
            table = pq.read_table(path, columns=list(INSTRUCTION_COLUMNS))
        except Exception:
            continue
        seen = set()
        for column in INSTRUCTION_COLUMNS:
            for value in table.column(column).to_pylist():
                if isinstance(value, str) and value.strip():
                    seen.add(normalize(value))
        counts.update(seen)
    return counts


def groundable(instruction: str) -> str | None:
    """Return the task type if the generator can ground this instruction, else None."""
    if FIXTURE_PATTERN.search(instruction) or UNSUPPORTED_PATTERN.search(instruction):
        return None
    task_type = infer_task_type(instruction)
    if task_type is None:
        return None
    try:
        concepts(instruction, task_type)
    except ValueError:
        return None
    return task_type


def select(episodes: Path, limit: int = 100) -> dict[str, object]:
    counts = count_instructions(episodes)
    tasks, rejected = [], 0
    for instruction, episode_count in counts.most_common():
        if len(tasks) >= limit:
            break
        task_type = groundable(instruction)
        if task_type is None:
            rejected += 1
            continue
        tasks.append({
            "rank": len(tasks) + 1,
            "instruction": instruction,
            "task_type": task_type,
            "episodes": episode_count,
        })
    return {
        "family": "REALM_DROID100",
        # Names this ranking so rank-keyed reviewed overrides authored against a different
        # DROID sample cannot silently attach to unrelated tasks.
        "ranking_id": f"droid100-local-{episodes.name}",
        "episode_source": str(episodes),
        "unique_instructions": len(counts),
        "rejected_candidates": rejected,
        "tasks": tasks,
    }


MULTI_OBJECT = re.compile(
    r"\b(two|three|four|some|all|both|them|together|pile|stack of|several|each|every)\b"
    r"|\b(cups|mugs|bottles|blocks|objects|markers|pens|items|things|plates|bowls)\b"
)


def select_clusters(counts_path: Path, limit: int = 300) -> dict[str, object]:
    """Rank TASKS, not phrasings, from a per-instruction count file.

    `counts_path` is the JSON the data survey writes ({"instructions": [{"instruction",
    "episodes", "successful_episodes", "locations"}...]}). Ranking exact phrasings made the
    top 100 mostly rewordings of "put the marker in the cup". Here every phrasing is grounded to a
    signature (task_type, main category, receiver/source category) with the agent's own offline
    classifier and the committed asset catalogue, phrasings with the same signature are merged,
    and clusters are ranked by distinct DROID locations, then successful episodes. Failed episodes
    never count. The representative phrasing of a cluster is its most demonstrated wording; the
    generator agent still decides the final grounding.
    """
    from tooling.task_authoring.agent import catalog, run

    payload = json.loads(counts_path.read_text(encoding="utf-8"))
    assets, catalog_info = catalog.resolve()
    clusters: dict[tuple, dict] = {}
    rejected: Counter[str] = Counter()
    for row in payload.get("instructions", []):
        text = normalize(str(row.get("instruction", "")))
        successful = int(row.get("successful_episodes") or 0)
        if not text or successful <= 0:
            rejected["no successful episode"] += 1
            continue
        if FIXTURE_PATTERN.search(text) or UNSUPPORTED_PATTERN.search(text):
            rejected["fixture or unsupported verb"] += 1
            continue
        if MULTI_OBJECT.search(text.lower()):
            rejected["several objects"] += 1
            continue
        task_type, nouns = run.classify(text, assets)
        if not task_type:
            rejected["no scorable verb"] += 1
            continue
        if not nouns:
            rejected["no catalogue noun"] += 1
            continue
        main, other = nouns[0], (nouns[-1] if len(nouns) > 1 else "")
        if task_type in {"put", "stack"} and not other:
            rejected["no receiver"] += 1
            continue
        key = (task_type, main, other)
        cluster = clusters.setdefault(key, {"locations": set(), "successful": 0, "phrasings": Counter()})
        cluster["locations"].update(row.get("locations") or [])
        cluster["successful"] += successful
        cluster["phrasings"][text] += successful
    ranked = sorted(
        clusters.items(),
        key=lambda item: (-len(item[1]["locations"]), -item[1]["successful"], item[0]),
    )
    tasks = []
    for rank, ((task_type, main, other), cluster) in enumerate(ranked[:limit], start=1):
        phrasings = [text for text, _ in sorted(cluster["phrasings"].items(), key=lambda kv: (-kv[1], kv[0]))]
        tasks.append({
            "rank": rank,
            "instruction": phrasings[0],
            "task_type": task_type,
            "signature": [task_type, main, other],
            "episodes": cluster["successful"],
            "locations": len(cluster["locations"]),
            "variants": phrasings[1:6],
        })
    return {
        "family": "REALM_DROID100",
        "ranking_id": f"droid-clusters-{payload.get('chunks', 'x')}chunks-v1",
        "episode_source": payload.get("source_roots"),
        "ranking": "task clusters by distinct locations, then successful episodes",
        "catalog_fingerprint": catalog_info["fingerprint"],
        "unique_instructions": len(payload.get("instructions", [])),
        "clusters": len(clusters),
        "rejected_phrasings": dict(rejected.most_common()),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--counts", type=Path, default=None,
                        help="per-instruction count JSON from the data survey; ranks merged task clusters")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.counts is not None:
        selection = select_clusters(args.counts, args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        print(f"Selected {len(selection['tasks'])} of {selection['clusters']} task clusters "
              f"(from {selection['unique_instructions']} phrasings) -> {args.output}")
        return
    selection = select(args.episodes, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected {len(selection['tasks'])} of {selection['unique_instructions']} "
        f"unique instructions -> {args.output}"
    )


if __name__ == "__main__":
    main()
