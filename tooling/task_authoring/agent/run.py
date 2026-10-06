"""Drive one instruction through the generator agent and emit a REALM task config.

Two paths, one tool surface:

  --offline   a scripted agent that calls the tools in the documented order and emits the result.
              No API key, no network. This is what makes the pipeline testable on a development
              machine and what proves the tool contract is complete: if the offline driver can
              ground an instruction using only the six tools, so can a model.
  default     the Anthropic tool-runner (requires `anthropic` and an API key).

Determinism: outputs are cached on `(task_id, PROMPT_VERSION, model_id)`. A re-run with an
unchanged cache performs no API calls and reproduces byte-identical YAML, which is the property
that lets a generated family be a benchmark (AGENTIC_PIPELINE.md section 1.4).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

from tooling.task_authoring.agent import catalog as asset_catalog
from tooling.task_authoring.agent import layout, tools
from tooling.task_authoring.agent.corrections import DEFAULT_CORRECTIONS_DIR, task_id
from tooling.task_authoring.agent.prompts import GENERATOR_SYSTEM, PROMPT_VERSION


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "realm" / "config" / "tasks" / "REALM_DROID100"
DEFAULT_CACHE = REPO_ROOT / "tmp" / "droid100" / "agent_cache"
#: Override with --model or REALM_AGENT_MODEL. Part of the cache key, so switching models re-runs.
MODEL = os.environ.get("REALM_AGENT_MODEL", "claude-opus-5")
MAX_TOOL_ROUNDS = 24

#: Verbs the offline driver recognizes. Mirrors the intent rules in
#: tooling/task_authoring/pages/agentic_task_authoring.py so the two cannot disagree.
VERB_RULES = (
    ("open_drawer", ("open the drawer", "open drawer", "pull the drawer open")),
    ("close_drawer", ("close the drawer", "close drawer", "shut the drawer")),
    ("stack", ("stack ", "on top of", "onto ")),
    ("push", ("push ",)),
    ("rotate", ("rotate ", "reorient ", "turn the ")),
)
#: Concept -> catalogue category. Same mapping the batch generator uses.
CONCEPT_TO_CATEGORY = asset_catalog.CONCEPT_TO_CATEGORY
CONCEPT_WORDS = tuple(sorted(CONCEPT_TO_CATEGORY, key=len, reverse=True))
#: Instruction grammar, verbs and bare colour names. Excluded when the catalogue is scanned for
#: nouns, so a category that merely starts with an English word ("mug_holder" vs "the") cannot
#: invent an object. Only words that could plausibly collide need to be here; the catalogue is
#: still the final authority via `ground_word`.
STOPWORDS = frozenset("""
a an the this that these those it its into in inside on onto onto of to for from with and or then
put place move drop insert pick grab lift take remove get stack rotate reorient turn push open
close shut slide pull drag set bring there here top bottom left right up down over under next
object objects thing things item items something anything one two three some any all both each
colour color coloured colored
blue green yellow orange red white black silver
""".split())
RECEIVER_WORDS = ("bowl", "box", "pot", "pan", "cup", "mug", "plate", "tray", "basket", "bin")
COLOR_WORDS = {
    "blue": [0.1, 0.25, 0.9, 1.0], "green": [0.1, 0.7, 0.25, 1.0],
    "yellow": [0.9, 0.8, 0.1, 1.0], "orange": [0.95, 0.4, 0.05, 1.0],
    "red": [0.85, 0.1, 0.1, 1.0], "white": [0.9, 0.9, 0.9, 1.0],
    "black": [0.08, 0.08, 0.08, 1.0], "silver": [0.65, 0.68, 0.72, 1.0],
}
DISTRACTOR_PREFS = ("apple", "orange", "lemon", "marker", "tablefork", "sponge",
                    "chocolate_bar", "toy_dice", "cork", "pear", "plum")


#: Phrasings REALM's single-main-object contract cannot represent ("two cups", "them", "the pile").
MULTI_OBJECT = re.compile(
    r"\b(two|three|four|some|all|both|them|together|pile|stack of|several|each|every)\b"
    r"|\b(cups|mugs|bottles|blocks|objects|markers|pens|items|things|plates|bowls)\b"
)
#: "... from X and put it on the table": a removal whose destination is the support itself.
REMOVAL_TO_SUPPORT = re.compile(
    r"\b(remove|take|pick|lift|get)\b.*\b(from|out of|off)\b.*\b(on|onto|to)\b\s+(the\s+)?"
    r"(table|desk|counter|countertop|tabletop|surface)\b"
)


def classify(instruction: str, assets: dict | None = None) -> tuple[str, list[str]]:
    """instruction -> (task_type, ordered object nouns). Deterministic, catalogue-aware."""
    lowered = instruction.lower()
    if REMOVAL_TO_SUPPORT.search(lowered):
        # The rubric for removal is `pick` (LIFT_LARGE); "on the table" is the support, not a target.
        return "pick", nouns(instruction, assets)
    for task_type, needles in VERB_RULES:
        if any(needle in lowered for needle in needles):
            return task_type, nouns(instruction, assets)
    if re.search(r"\b(put|place|move|drop|insert)\b", lowered) and re.search(r"\b(in|into|inside|on|onto)\b", lowered):
        # "in/into/inside" names a receiver and wins over a stray "on" ("the pen on the table").
        into = re.search(r"\b(in|into|inside)\b", lowered)
        onto = re.search(r"\b(on|onto|on top of)\b", lowered)
        return ("stack" if onto and not into else "put"), nouns(instruction, assets)
    if re.search(r"\b(pick|grab|lift|take|remove|get)\b", lowered):
        return "pick", nouns(instruction, assets)
    return "", nouns(instruction, assets)


def ground_word(word: str, assets: dict) -> str | None:
    """The indexed category a word names, or None. Mirrors `Session.search_assets` matching.

    Word-boundary aware (see `catalog.match_category`). The catalogue is the authority on what is
    an object: a word counts as a noun here only if something actually answers to it.
    """
    return asset_catalog.match_category(word, assets)


def nouns(instruction: str, assets: dict | None = None) -> list[str]:
    """Content nouns in order of appearance, resolved to catalogue categories where known.

    With `assets` supplied, any word that names an indexed category counts -- the catalogue decides
    what an object is, so a task family is not limited to whatever the static vocabulary below
    happened to list. Without it, fall back to that vocabulary, which is what keeps `classify`
    usable as a pure function. An instruction naming an object the catalogue has never heard of
    yields no noun for it either way, which is the honest answer: `search_assets` will not find it
    and the driver declines rather than inventing a stand-in.
    """
    lowered = instruction.lower()
    found: list[tuple[int, str]] = []
    for word in CONCEPT_WORDS:
        match = re.search(rf"\b{re.escape(word)}s?\b", lowered)
        if match:
            found.append((match.start(), CONCEPT_TO_CATEGORY[word]))
    if assets:
        for match in re.finditer(r"\b([a-z][a-z_]{2,})\b", lowered):
            word = match.group(1)
            if word in CONCEPT_WORDS or word in STOPWORDS:
                continue
            category = ground_word(word, assets)
            if category:
                found.append((match.start(), category))
    if not found:
        return []
    # Position order decides; name length breaks the tie, so a category that merely contains
    # another one cannot reorder the pair.
    found.sort(key=lambda item: (item[0], -len(item[1])))
    ordered: list[str] = []
    for _, category in found:
        if category not in ordered:
            ordered.append(category)
    return ordered


def rewrite_instruction(instruction: str, substitutions: dict[str, str]) -> tuple[str, list[str]]:
    """Replace instruction words whose grounded category differs, so the config names what exists.

    `instruction_obj_to_replace` must be a literal substring of `instruction` or the S-NOUN/S-VRB
    perturbations silently no-op (validation.py INSTRUCTION_CLOSURE). When "spoon" grounds to the
    catalogue's `teaspoon`, the honest instruction says "teaspoon" -- the config may only promise
    objects that are actually in the scene.
    """
    notes: list[str] = []
    text = instruction
    for word, category in substitutions.items():
        if word == category:
            continue
        pattern = re.compile(rf"\b{re.escape(word)}s?\b", re.IGNORECASE)
        if not pattern.search(text):
            continue
        text = pattern.sub(category.replace("_", " "), text)
        notes.append(f"rewrote {word!r} as {category!r}: that is the indexed asset standing in")
    return text, notes


def build_roles(instruction: str, task_type: str, session: tools.Session) -> tuple[dict, list[str]]:
    """The offline equivalent of the model's role assignment: semantics only, no geometry."""
    decisions: list[str] = []
    lowered = instruction.lower()
    resolved = nouns(instruction, session.assets)
    colors = [word for word in COLOR_WORDS if re.search(rf"\b{word}\b", lowered)]
    first_color = colors[0] if colors else None
    # Words the instruction uses for a concept the catalogue names differently, so the emitted
    # instruction can name the asset that is really in the scene.
    substitutions: dict[str, str] = {}
    for word, category in CONCEPT_TO_CATEGORY.items():
        if re.search(rf"\b{word}s?\b", lowered) and word != category:
            substitutions[word] = category

    if task_type in {"open_drawer", "close_drawer"}:
        box = session.search_assets("bottom_cabinet", "main", (0.5, 0.5))
        if not box["found"]:
            raise layout.LayoutError("no articulated cabinet is indexed")
        return {
            "task_type": task_type,
            "instruction": instruction,
            "main": {"name": "cabinet", "category": box["category"], "model": box["candidates"][-1]["model"]},
        }, decisions

    main_category = resolved[0] if resolved else "bowl"
    main: dict = {"name": main_category}
    if first_color and main_category in {"bowl", "plate", "block"}:
        # A coloured block is the only object whose colour the config can guarantee.
        main = {"name": f"{first_color}_block", "primitive": "block", "rgba": COLOR_WORDS[first_color], "extent": 0.05}
        decisions.append(
            f"authored the {first_color} object as a PrimitiveObject cube: only a primitive's "
            f"colour is guaranteed by the config"
        )
    else:
        hit = session.search_assets(main_category, "main", (0.16, 0.18))
        if not hit["found"]:
            raise layout.LayoutError(
                f"no indexed asset for {main_category!r}. Call report_ungroundable."
            )
        model = hit["candidates"][0]["model"]
        if hit["category"] != main_category:
            decisions.append(f"grounded {main_category!r} as the indexed category {hit['category']!r}")
        main.update({"category": hit["category"], "model": model})

    # Every role is resolved before the instruction is finalized, because the config may only name
    # objects that are actually in it.
    roles: dict = {"task_type": task_type, "instruction": instruction, "main": main}

    if task_type in {"put", "stack"}:
        # The LAST noun is the receiver ("put X in Y", "take X out of Z and put it in Y"); ordering
        # beats a keyword scan, which would pick the main object whenever it is also a container
        # word ("Stack the cup on the plate" must not take the cup as the support).
        receiver_category = resolved[-1] if len(resolved) > 1 else None
        if receiver_category is None:
            word = next((item for item in RECEIVER_WORDS if re.search(rf"\b{item}\b", lowered)), None)
            receiver_category = CONCEPT_TO_CATEGORY.get(word, word) if word else "bowl"
        hit = session.search_assets(receiver_category, "target", (0.28, 0.28))
        if not hit["found"]:
            raise layout.LayoutError(
                f"task_type {task_type!r} needs a receiver and no indexed asset matches "
                f"{receiver_category!r}; report_ungroundable rather than inventing one"
            )
        roles["target"] = {"name": hit["category"], "category": hit["category"], "model": hit["candidates"][-1]["model"]}

    if task_type == "pick" and len(resolved) > 1:
        source_category = resolved[1]
        hit = session.search_assets(source_category, "source", (0.28, 0.28))
        if hit["found"]:
            roles["source"] = {
                "name": hit["category"], "category": hit["category"], "model": hit["candidates"][-1]["model"]
            }
            roles["initial_state"] = [
                {"predicate": "inside", "subject": str(main["name"]), "object": hit["category"]}
            ]
            decisions.append(
                f"grounded the source as an immutable {hit['category']!r} with an inside relation"
            )

    used = {str(main.get("category") or "")}
    used |= {str(roles[key]["category"]) for key in ("target", "source") if key in roles}
    distractors = []
    for category in DISTRACTOR_PREFS:
        if len(distractors) >= 3:
            break
        if category in used:
            continue
        hit = session.search_assets(category, "distractor", (0.12, 0.12))
        if not hit["found"]:
            continue
        distractors.append({
            "name": f"distractor_{hit['category']}",
            "category": hit["category"],
            "model": hit["candidates"][0]["model"],
        })
    roles["distractors"] = distractors
    # Rewrite last: the substitution map is built from the raw instruction, but the text is only
    # finalized once every role is resolved, so the instruction names exactly what is in the scene.
    final_instruction, notes = rewrite_instruction(instruction, substitutions)
    roles["instruction"] = final_instruction
    decisions.extend(notes)
    return roles, decisions


def run_offline(instruction: str, session: tools.Session) -> dict:
    """Scripted agent: the documented procedure, driven without a model.

    Every exit either submits a validated draft or records *why* it did not. A refusal with no
    reason is indistinguishable from a crash, and the run report is the only thing an operator
    sees when a family comes back short.
    """
    if MULTI_OBJECT.search(instruction.lower()):
        return session.report_ungroundable(
            "the instruction manipulates several objects; REALM scores exactly one main object",
            nouns(instruction, session.assets),
        )
    task_type, _ = classify(instruction, session.assets)
    if not task_type:
        return session.report_ungroundable(
            f"no REALM task type recognizes this instruction. The scored verbs are put, pick, "
            f"rotate, push, stack, open_drawer, close_drawer.",
            [instruction],
        )
    try:
        roles, decisions = build_roles(instruction, task_type, session)
    except layout.LayoutError as error:
        return session.report_ungroundable(str(error), nouns(instruction, session.assets))

    regions = session.list_scene_regions()["regions"]
    best_index, best_area = 0, -1.0
    for region in regions:
        area = float(region["width"]) * float(region["depth"])
        if area > best_area:
            best_index, best_area = int(region["region_index"]), area
    roles["region_index"] = best_index
    roles["decisions"] = decisions

    proposed = session.propose_layout(**roles)
    if not proposed.get("ok"):
        return session.report_ungroundable(
            f"placement is impossible for the honest role assignment: {proposed.get('reason')}",
            nouns(instruction, session.assets),
        )

    validation = session.validate_draft()
    if not validation.get("ok"):
        # Not a refusal: the tool contract is fine, this draft is wrong. Record the errors so the
        # report names them, and leave `submission` unset so the config is never written.
        errors = [str(item.get("code")) for item in validation.get("findings") or []]
        session.ungroundable = {
            "instruction": instruction,
            "reason": (
                f"the honest role assignment produced a draft that does not validate: "
                f"{validation.get('error_count')} error(s) {errors}. This is a layout defect, not a "
                f"missing asset -- report it rather than regenerating with a different asset."
            ),
            "blocking_terms": nouns(instruction, session.assets),
        }
        return {"ok": False, "reason": session.ungroundable["reason"]}
    return session.submit_task(decisions)


def run_agent(instruction: str, session: tools.Session, model: str = MODEL) -> dict:
    """The model path. Requires `anthropic` and an API key; falls back with a clear message."""
    try:
        import anthropic
    except ImportError:
        return {
            "ok": False,
            "reason": (
                "the `anthropic` package is not installed. Run `uv add anthropic`, or use "
                "`--offline` which drives the same six tools without a model."
            ),
        }
    client = anthropic.Anthropic()
    schemas = [
        {"name": schema["name"], "description": schema["description"], "input_schema": schema["input_schema"]}
        for schema in tools.tool_schemas()
    ]
    messages = [{"role": "user", "content": brief(instruction, session)}]
    transcript: list[dict] = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=[{"type": "text", "text": GENERATOR_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=schemas,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": [block.model_dump() for block in response.content]})
        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            break
        results = []
        for call in calls:
            payload = tools.dispatch(session, call.name, call.input)
            transcript.append({"tool": call.name, "input": call.input, "result": payload})
            results.append({
                "type": "tool_result", "tool_use_id": call.id, "content": tools.dumps(payload),
            })
        messages.append({"role": "user", "content": results})
        if session.submission or session.ungroundable:
            break
    if session.submission:
        return session.submission
    if session.ungroundable:
        return session.ungroundable
    return {"ok": False, "reason": "the agent stopped without submitting or declining", "transcript": transcript}


def brief(instruction: str, session: tools.Session) -> str:
    """The volatile per-task half of the prompt, kept out of the cached system prefix."""
    classification, resolved = classify(instruction)
    return (
        f"Instruction: {instruction!r}\n\n"
        f"A local classifier proposes task_type={classification!r} and nouns={resolved}. "
        f"That classifier mirrors REALM's own intent rules and is usually right, but you own the "
        f"decision: override it if the instruction's rubric cannot see what it promises.\n\n"
        f"Begin by searching for the assets these nouns name. When you are done, call submit_task "
        f"with your decisions, or report_ungroundable with a reason."
    )


def cache_path(instruction: str, ranking_id: str | None, cache_dir: Path) -> Path:
    return cache_dir / f"{task_id(instruction, ranking_id)}.json"


def generate_one(
    instruction: str,
    *,
    offline: bool,
    dataset: Path | None = None,
    output: Path | None = None,
    seed: int = 100,
    ranking_id: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    corrections_dir: Path | None = DEFAULT_CORRECTIONS_DIR,
    assets_by_category: dict | None = None,
    catalog: Path | None = None,
    model: str = MODEL,
) -> dict:
    """One instruction -> a validated config (or an honest refusal). Cached on content.

    The cache is valid only for the same prompt version, model and asset catalogue. Keying on the
    catalogue fingerprint is what stops a run made against a missing dataset (100/100 declined)
    from being replayed verbatim after the dataset path is fixed.
    """
    resolved_cache = cache_dir if cache_dir is not None else DEFAULT_CACHE
    key = task_id(instruction, ranking_id)
    cached = cache_path(instruction, ranking_id, resolved_cache)
    if assets_by_category is None:
        assets_by_category, _ = asset_catalog.resolve(dataset=dataset, catalog=catalog)
    catalog_key = asset_catalog.fingerprint(assets_by_category)
    model_id = "offline" if offline else model
    if use_cache and cached.is_file():
        try:
            record = json.loads(cached.read_text(encoding="utf-8"))
            if (
                record.get("prompt_version") == PROMPT_VERSION
                and record.get("model") == model_id
                and record.get("catalog_fingerprint") == catalog_key
            ):
                if record["document"] and output is not None:
                    write_task(record, output, None)
                return record
        except (OSError, ValueError):
            pass

    session = tools.Session(
        instruction, dataset=dataset, seed=seed, ranking_id=ranking_id,
        corrections_dir=corrections_dir, assets_by_category=assets_by_category,
    )
    outcome = run_offline(instruction, session) if offline else run_agent(instruction, session, model)
    record = {
        "task_id": key,
        "instruction": instruction,
        "ranking_id": ranking_id,
        "prompt_version": PROMPT_VERSION,
        "model": model_id,
        "catalog_fingerprint": catalog_key,
        "grounded": bool(session.submission),
        "ungroundable": session.ungroundable,
        "validation": (session.submission or {}).get("validation"),
        "document": (session.submission or {}).get("document"),
    }
    if not record["grounded"] and not record["ungroundable"]:
        # The model stopped (or the API failed) without submitting or declining. Keep the reason
        # in the report, and do not cache it: it says nothing about the instruction.
        record["failure"] = str((outcome or {}).get("reason") or "no submission and no decline")
        return record
    resolved_cache.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    if record["document"] and output is not None:
        write_task(record, output, session)
    return record


def write_task(record: dict, output: Path, session: tools.Session | None = None) -> Path:
    directory = output / f"{_slug(record['instruction'])[:72]}_{record['task_id'][:6]}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "default.yaml").write_text(
        yaml.safe_dump(record["document"], sort_keys=False, width=120), encoding="utf-8"
    )
    return directory


def signature(document: dict) -> tuple:
    """What makes two generated tasks the same benchmark task: verb + the grounded role assets.

    DROID phrasings repeat ("Put the marker in the cup" / "... in the mug" / "Pick up the marker
    and put it in the cup"). Once grounded they solve to the same scene, and counting them twice
    inflates the family without adding a task.
    """
    def ident(role: str) -> str:
        configs = document.get(role) or []
        if not configs:
            return ""
        config = configs[0]
        return str(config.get("category") or f"{config.get('primitive_type')}:{config.get('rgba')}")

    return (
        str(document.get("task_type")),
        ident("main_objects"), ident("target_objects"), ident("immutables"),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("instructions", type=Path,
                        help="a JSON file {tasks:[{instruction, rank?}]}, or a text file, one per line")
    parser.add_argument("--offline", action="store_true",
                        help="drive the six tools with a scripted agent; no API key required")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="scan this behavior-1k-assets tree instead of the committed catalogue")
    parser.add_argument("--catalog", type=Path, default=None,
                        help=f"asset catalogue JSON (default: {asset_catalog.DEFAULT_CATALOG.name})")
    parser.add_argument("--model", default=MODEL, help="Anthropic model id for the non-offline path")
    parser.add_argument("--limit", type=int, default=None, help="only the first N instructions")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="write every grounded config, even ones that solve to the same task")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--ranking-id", default=None)
    parser.add_argument("--no-write", action="store_true", help="report only; write no configs")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", type=Path, default=None, help="write the run report as JSON")
    args = parser.parse_args(argv)

    if not args.instructions.is_file():
        print(f"no such instructions file: {args.instructions}", file=sys.stderr)
        return 2
    text = args.instructions.read_text(encoding="utf-8")
    entries: list[dict] = []
    if args.instructions.suffix == ".json":
        payload = json.loads(text)
        for item in payload.get("tasks", []):
            entries.append({"instruction": str(item["instruction"]), "rank": item.get("rank")})
        args.ranking_id = args.ranking_id or payload.get("ranking_id")
    else:
        entries = [{"instruction": line.strip()} for line in text.splitlines() if line.strip()]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("no instructions found", file=sys.stderr)
        return 2

    # Resolve the catalogue ONCE and say which one it is. A wrong --dataset is an error here,
    # not 100 silent declines.
    try:
        assets, catalog_info = asset_catalog.resolve(dataset=args.dataset, catalog=args.catalog)
    except asset_catalog.CatalogError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"catalogue: {catalog_info['source']} ({len(assets)} categories, "
          f"fingerprint {catalog_info['fingerprint']}) from {catalog_info['path']}")
    if catalog_info["source"] == "seed-from-configs":
        print("  note: this is the SEED catalogue; build the full one with "
              "`python -m tooling.task_authoring.agent.catalog build --dataset ...`")
    if not args.offline and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set (or pass --offline)", file=sys.stderr)
        return 2

    results, counts = [], {"grounded": 0, "duplicate": 0, "declined": 0, "failed": 0}
    seen: dict[tuple, str] = {}
    for entry in entries:
        record = generate_one(
            entry["instruction"], offline=args.offline, dataset=args.dataset, output=None,
            seed=args.seed, ranking_id=args.ranking_id, use_cache=not args.no_cache,
            assets_by_category=assets, model=args.model,
        )
        record["rank"] = entry.get("rank")
        status, detail = "DECLINED", ""
        if record["grounded"]:
            key = signature(record["document"])
            if key in seen and not args.keep_duplicates:
                record["duplicate_of"] = seen[key]
                status, detail = "DUP     ", f" -- same task as {seen[key]}"
                counts["duplicate"] += 1
            else:
                seen.setdefault(key, record["task_id"])
                status = "OK      "
                counts["grounded"] += 1
                if not args.no_write:
                    record["path"] = str(write_task(record, args.output))
        elif record.get("failure"):
            status, detail = "FAILED  ", f" -- {record['failure'][:90]}"
            counts["failed"] += 1
        else:
            detail = f" -- {record['ungroundable']['reason'][:90]}"
            counts["declined"] += 1
        results.append(record)
        if not args.quiet:
            print(f"{status} {record['task_id']}  {entry['instruction'][:60]}{detail}")

    print(f"\n{counts['grounded']} unique tasks, {counts['duplicate']} duplicates, "
          f"{counts['declined']} declined, {counts['failed']} failed, {len(entries)} total")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "prompt_version": PROMPT_VERSION,
            "offline": args.offline,
            "model": "offline" if args.offline else args.model,
            "catalog": catalog_info,
            **counts,
            "total": len(entries),
            "results": results,
        }, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
