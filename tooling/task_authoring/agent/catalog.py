"""The asset catalogue the agent grounds against, decoupled from the 1k-asset dataset tree.

`layout.index_assets()` walks the OmniGibson `behavior-1k-assets` tree. That tree is tens of GB and
lives on the lab workstation / cluster, not on a laptop -- and when it is missing the index is
empty and every instruction is declined as "ungroundable", which reads like a grounding failure
rather than a missing dataset.

The agent only ever needs three facts per asset: category, model id, and the natural bbox extent.
This module snapshots those into a small JSON file that is committed to the repo, so the host-side
pipeline (agent, validator, offline driver) runs anywhere, and the dataset is needed only to
refresh the snapshot and inside the container.

Resolution order (`resolve`):

  1. `--catalog FILE`                      an explicit snapshot
  2. `--dataset DIR` that indexes assets   a live scan of the tree (and it MUST index something:
                                            an explicit dataset that yields zero assets is an error)
  3. `DEFAULT_CATALOG`                      the committed snapshot

Build the full snapshot once, on a machine that has the dataset (no GPU needed):

    python -m tooling.task_authoring.agent.catalog build \
        --dataset "$REALM_DATA_PATH/datasets/behavior-1k-assets"

Until that has been run, the committed file is a SEED built from the assets REALM's own task
configs already load (`seed-from-configs`). Seed bboxes are authored extents, not natural ones, and
the seed covers only ~40 categories, so a run against it declines more than a run against the full
catalogue would. `source` in the file says which one you have.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "tooling" / "task_authoring" / "asset_catalog.json"
DEFAULT_TASK_CONFIGS = REPO_ROOT / "realm" / "config" / "tasks"
ROLE_KEYS = ("main_objects", "target_objects", "distractors", "immutables")
#: Fixtures and scene furniture that appear in configs but are never tabletop objects.
NON_TABLETOP = frozenset({"breakfast_table", "carpet", "pot_plant", "hat"})


#: Everyday word -> catalogue category, where DROID speech and BEHAVIOR naming differ.
CONCEPT_TO_CATEGORY = {
    "marker": "marker", "pen": "pen", "cup": "mug", "mug": "mug", "bowl": "bowl",
    "lid": "lid", "pot": "saucepot", "pan": "frying_pan", "towel": "dishtowel",
    "box": "storage_box", "tape": "masking_tape", "plate": "plate", "can": "can",
    "cloth": "microfiber_cloth", "spoon": "teaspoon", "screwdriver": "screwdriver",
}
#: Words that name the support surface the scene already provides, never an object to author.
#: Colour adjectives that are also catalogue categories ("orange") are never object nouns here.
COLOR_WORDS = frozenset({"orange", "lime", "lemon", "olive", "plum", "cherry", "peach", "chocolate"})
SUPPORT_WORDS = frozenset({"table", "desk", "counter", "countertop", "tabletop", "surface", "workspace"})
#: Instruction words that are never object nouns. Against the full dataset catalogue the compound
#: rules in match_category reach real categories from them ("and" -> nightstand, "it" -> post_it,
#: "pick" -> toothpick, "take" -> shiitake, "up" -> classroom_mock_up, "top" -> desk_top), which made
#: the phantom-noun check reject ordinary instructions.
FUNCTION_WORDS = frozenset({
    "a", "an", "the", "it", "its", "them", "this", "that", "and", "or", "then", "with",
    "in", "into", "inside", "on", "onto", "top", "of", "off", "out", "from", "to", "up", "down",
    "over", "under", "back", "away", "next",
    "put", "place", "set", "pick", "take", "grab", "lift", "remove", "move", "stack", "cover",
    "uncover", "drop", "push", "press", "open", "close", "turn", "rotate", "flip",
})


class CatalogError(RuntimeError):
    """The catalogue could not be resolved. The message says what to do."""


def match_category(word: str, categories) -> str | None:
    """The catalogue category a single word names, or None.

    Word-boundary aware, unlike a raw substring test: "can" must reach `can_of_soda`, never
    `box_of_cane_sugar`; "pot" must reach `saucepot`, never `pot_plant`. In priority order:

      1. exact category                       "bowl"  -> bowl
      2. the category's head noun is the word  "cup"   -> coffee_cup   ("X_Y" head is Y,
                                                "can"   -> can_of_soda   "X_of_Y" head is X)
      3. a closed compound ending in the word  "spoon" -> teaspoon, "pot" -> saucepot

    Plurals fall back to the singular. Ties go to the shortest, then alphabetical, category.
    """
    word = str(word).strip().lower()
    names = list(categories)
    if not word:
        return None
    if word in names:
        return word

    def head(category: str) -> str:
        if "_of_" in category:
            return category.split("_of_", 1)[0]
        return category.rsplit("_", 1)[-1]

    for rule in (
        lambda c: head(c) == word,
        lambda c: len(word) >= 3 and "_" not in c and c.endswith(word) and len(c) - len(word) >= 3,
    ):
        hits = sorted((c for c in names if rule(c)), key=lambda c: (len(c), c))
        if hits:
            return hits[0]
    if word.endswith("s") and len(word) > 3:
        return match_category(word[:-1], names)
    return None


def phantom_nouns(document: dict, assets_by_category: dict[str, list[dict]]) -> list[str]:
    """Words in the instruction that name a catalogue object which is NOT in the scene.

    "Put the glass lid on the black pot" grounded without a pot is a config that promises an
    object it does not contain. Geometry validation cannot see that; this can.
    """
    import re

    present = set()
    for role in ROLE_KEYS:
        for config in document.get(role) or []:
            present.add(str(config.get("category") or ""))
            present.update(str(config.get("name") or "").lower().replace("_", " ").split())
    concept_values = set(CONCEPT_TO_CATEGORY.values())
    phantoms = []
    for word in re.findall(r"[a-z]+", str(document.get("instruction", "")).lower()):
        if word in SUPPORT_WORDS or word in COLOR_WORDS or word in FUNCTION_WORDS or word in present:
            continue
        if word in CONCEPT_TO_CATEGORY:
            category = CONCEPT_TO_CATEGORY[word]
        elif word in concept_values:
            category = word
        else:
            category = match_category(word, assets_by_category)
        if category and category not in present:
            phantoms.append(word)
    return sorted(set(phantoms))


def _entry(category: str, model: str, bbox, source: str) -> dict:
    return {
        "category": str(category),
        "model": str(model),
        "bbox": [round(float(value), 5) for value in bbox],
        "bbox_source": source,
    }


def build_from_dataset(dataset: Path) -> dict:
    """Scan the behavior-1k-assets tree. Reads only paths and `misc/metadata.json`; no OmniGibson."""
    from tooling.task_authoring.authoring import discover_assets

    assets = [
        _entry(item["category"], item["model"], item["bbox"], str(item.get("bbox_source", "metadata")))
        for item in discover_assets(dataset)
        if item.get("model")
    ]
    if not assets:
        raise CatalogError(
            f"no USD assets found under {dataset}. Point --dataset at the behavior-1k-assets "
            f"directory (the one containing objects/<category>/<model>/)."
        )
    return {"source": "dataset", "dataset": str(dataset), "assets": _dedupe(assets)}


def build_from_configs(config_root: Path = DEFAULT_TASK_CONFIGS) -> dict:
    """Seed catalogue: every DatasetObject REALM's own task configs already load.

    Each (category, model) is known to exist in the dataset because a shipped task loads it. The
    bbox is the most common authored extent for that model, which is a scaled value, not the
    asset's natural size -- hence `bbox_source: authored_config`.
    """
    seen: dict[tuple[str, str], Counter] = {}
    for path in sorted(config_root.rglob("*.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue
        for role in ROLE_KEYS:
            for config in document.get(role) or []:
                if not isinstance(config, dict) or config.get("type") != "DatasetObject":
                    continue
                category, model = config.get("category"), config.get("model")
                bbox = config.get("bounding_box")
                if not category or not model or category in NON_TABLETOP:
                    continue
                if not (isinstance(bbox, list) and len(bbox) == 3):
                    continue
                # Some perturbation variants author placeholder 1 cm boxes; they are not sizes.
                if max(float(value) for value in bbox) < 0.02:
                    continue
                seen.setdefault((str(category), str(model)), Counter())[tuple(float(v) for v in bbox)] += 1
    assets = [
        _entry(category, model, counts.most_common(1)[0][0], "authored_config")
        for (category, model), counts in sorted(seen.items())
    ]
    return {"source": "seed-from-configs", "dataset": None, "assets": assets}


def _dedupe(assets: list[dict]) -> list[dict]:
    unique: dict[tuple[str, str], dict] = {}
    for asset in assets:
        unique.setdefault((asset["category"], asset["model"]), asset)
    return [unique[key] for key in sorted(unique)]


def save(catalog: dict, path: Path = DEFAULT_CATALOG) -> Path:
    payload = dict(catalog)
    payload["fingerprint"] = fingerprint(by_category(catalog["assets"]))
    payload["categories"] = len({item["category"] for item in catalog["assets"]})
    payload["models"] = len(catalog["assets"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load(path: Path = DEFAULT_CATALOG) -> dict:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CatalogError(f"cannot read asset catalogue {path}: {error}") from error
    except ValueError as error:
        raise CatalogError(f"asset catalogue {path} is not valid JSON: {error}") from error
    if not catalog.get("assets"):
        raise CatalogError(f"asset catalogue {path} lists no assets")
    return catalog


def by_category(assets: list[dict]) -> dict[str, list[dict]]:
    """The `assets_by_category` shape layout.py and tools.py consume."""
    indexed: dict[str, list[dict]] = {}
    for asset in assets:
        indexed.setdefault(str(asset["category"]), []).append(dict(asset))
    return indexed


def fingerprint(assets_by_category: dict[str, list[dict]]) -> str:
    """Content hash of what the agent can see. Part of the run-cache key: a decline made against
    a smaller catalogue must not be replayed against a bigger one."""
    rows = sorted(
        (category, str(item.get("model")), tuple(round(float(v), 5) for v in item.get("bbox") or ()))
        for category, items in assets_by_category.items()
        for item in items
    )
    return hashlib.sha1(json.dumps(rows).encode("utf-8")).hexdigest()[:12]


def resolve(dataset: Path | None = None, catalog: Path | None = None) -> tuple[dict[str, list[dict]], dict]:
    """(assets_by_category, provenance). Raises CatalogError instead of returning an empty index."""
    if catalog is not None:
        loaded = load(catalog)
        assets = by_category(loaded["assets"])
        return assets, {"source": loaded.get("source", "catalog"), "path": str(catalog),
                        "fingerprint": fingerprint(assets)}
    if dataset is not None:
        from tooling.task_authoring.agent.layout import index_assets

        assets = index_assets(dataset)
        if not assets:
            raise CatalogError(
                f"--dataset {dataset} indexed zero assets (missing or wrong directory). Every "
                f"instruction would be declined. Fix the path, or drop --dataset to use the "
                f"committed catalogue {DEFAULT_CATALOG.relative_to(REPO_ROOT)}."
            )
        return assets, {"source": "dataset", "path": str(dataset), "fingerprint": fingerprint(assets)}
    if DEFAULT_CATALOG.is_file():
        return resolve(catalog=DEFAULT_CATALOG)
    raise CatalogError(
        f"no asset catalogue: {DEFAULT_CATALOG.relative_to(REPO_ROOT)} is missing and no --dataset "
        f"was given. Run `python -m tooling.task_authoring.agent.catalog build --dataset ...`."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect the agent's asset catalogue.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="scan a behavior-1k-assets tree (run where the dataset lives)")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--out", type=Path, default=DEFAULT_CATALOG)
    seed = sub.add_parser("seed-from-configs", help="seed from the assets REALM's task configs load")
    seed.add_argument("--out", type=Path, default=DEFAULT_CATALOG)
    show = sub.add_parser("show", help="summarize a catalogue")
    show.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            path = save(build_from_dataset(args.dataset), args.out)
        elif args.command == "seed-from-configs":
            path = save(build_from_configs(), args.out)
        else:
            catalog = load(args.catalog)
            assets = by_category(catalog["assets"])
            print(f"{args.catalog}: source={catalog.get('source')} categories={len(assets)} "
                  f"models={len(catalog['assets'])} fingerprint={fingerprint(assets)}")
            print(", ".join(sorted(assets)))
            return 0
    except CatalogError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    catalog = load(path)
    print(f"wrote {path}: source={catalog['source']} categories={catalog['categories']} "
          f"models={catalog['models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
