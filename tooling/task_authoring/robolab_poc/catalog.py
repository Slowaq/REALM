"""Asset catalogue for the RoboLab-style PoC: the objects a scene spec may name.

Two sources, both host-only:

* REALM's own task configs (always available): every portable `DatasetObject` already used in
  `realm/config/tasks/**`. Its `bounding_box` is the AUTHORED extent, not the asset's natural one.
* An OmniGibson dataset tree (`--dataset`, lab machine): every USD model with its metadata bbox.
  These entries replace config-derived ones for the same (category, model).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_ROOT = REPO_ROOT / "realm" / "config" / "tasks"
DEFAULT_CATALOG = Path(__file__).with_name("catalog.json")
ROLE_KEYS = ("main_objects", "target_objects", "distractors", "immutables")


def _from_configs(tasks_root: Path) -> dict[tuple[str, str], dict]:

    found: dict[tuple[str, str], dict] = {}
    for path in sorted(tasks_root.rglob("*.yaml")):
        if path.relative_to(tasks_root).parts[0] == "ROBOLAB_POC":
            continue  # never learn from our own output
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue
        for role in ROLE_KEYS:
            for obj in document.get(role) or []:
                if not isinstance(obj, dict) or obj.get("type") != "DatasetObject" or obj.get("fixed_base"):
                    continue
                category, model, bbox = obj.get("category"), obj.get("model"), obj.get("bounding_box")
                if not (category and model and isinstance(bbox, list) and len(bbox) == 3):
                    continue
                if max(float(value) for value in bbox) < 0.02:
                    continue  # placeholder extents (e.g. trajectory_replay's 1 cm debug boxes)
                key = (str(category), str(model))
                upright = _is_upright(obj.get("orientation"))
                if key not in found or (upright and not found[key]["upright_in_source"]):
                    found[key] = {
                        "category": key[0],
                        "model": key[1],
                        "bbox": [round(float(value), 7) for value in bbox],
                        "bbox_source": "authored_config",
                        "seen_in": path.relative_to(tasks_root).as_posix(),
                        # False: the source config rolled/pitched the asset, so this extent may not
                        # be the upright one. Review such objects in the render.
                        "upright_in_source": upright,
                    }
    return found


def _is_upright(orientation) -> bool:
    """No orientation, or a yaw-only XYZW quaternion."""
    if not orientation:
        return True
    try:
        x, y = float(orientation[0]), float(orientation[1])
    except (TypeError, ValueError, IndexError):
        return False
    return abs(x) < 1e-6 and abs(y) < 1e-6


def _from_dataset(dataset: Path) -> dict[tuple[str, str], dict]:

    from tooling.task_authoring.authoring import discover_assets

    found = {}
    for asset in discover_assets(dataset):
        if asset["bbox_source"] != "model metadata":
            continue
        key = (str(asset["category"]), str(asset["model"]))
        found[key] = {
            "category": key[0],
            "model": key[1],
            "bbox": [round(float(value), 7) for value in asset["bbox"]],
            "bbox_source": "model_metadata",
            "upright_in_source": True,
        }
    return found


def build_catalog(tasks_root: Path = TASKS_ROOT, dataset: Path | None = None) -> dict:

    entries = _from_configs(tasks_root)
    if dataset is not None:
        if not dataset.is_dir():
            raise FileNotFoundError(f"--dataset {dataset} is not a directory")
        scanned = _from_dataset(dataset)
        if not scanned:
            raise ValueError(f"--dataset {dataset} contains no USD model with metadata bbox")
        entries.update(scanned)
    assets = [entries[key] for key in sorted(entries)]
    fingerprint = hashlib.sha1(json.dumps(assets, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "source": "configs+dataset" if dataset is not None else "configs",
        "fingerprint": fingerprint,
        "categories": len({asset["category"] for asset in assets}),
        "assets": assets,
    }


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict:

    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run `python -m tooling.task_authoring.robolab_poc.cli catalog`")
    return json.loads(path.read_text(encoding="utf-8"))


def by_category(catalog: dict) -> dict[str, list[dict]]:

    grouped: dict[str, list[dict]] = {}
    for asset in catalog["assets"]:
        grouped.setdefault(asset["category"], []).append(asset)
    return grouped


def search(catalog: dict, query: str, limit: int = 12) -> list[dict]:
    """Whole-word match on category tokens, so 'can' does not hit 'box_of_cane_sugar'."""

    words = [word for word in re.split(r"[\s_]+", query.lower().strip()) if word]
    hits = []
    for category, assets in sorted(by_category(catalog).items()):
        tokens = set(category.split("_"))
        if category == query.lower().strip() or (words and all(word in tokens for word in words)):
            hits.append({"category": category, "models": [
                {"model": asset["model"], "bbox": asset["bbox"], "upright_in_source": asset.get("upright_in_source", True)}
                for asset in assets
            ]})
    return hits[:limit]
