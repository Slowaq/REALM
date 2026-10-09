"""Semantic role assignment -> a geometrically solved REALM task document.

This is the boundary the whole agentic pipeline rests on: **the model proposes semantics, code
disposes geometry.** The model names objects, picks between candidate models, and asserts which
relation holds at the start. Every number in the emitted document -- bbox, scale, quaternion,
`relative_bbox_position` -- comes from this module calling the same solver
`generate_realm_droid100.py` uses for the batch path, under an explicitly seeded RNG.

Keeping the solve here rather than in the tool wrapper is what lets `propose_layout` and the batch
generator be checked against each other: `test_agent.py` asserts they agree on the same roles.

Determinism: the caller supplies a `random.Random`. Placement candidates are tried in a fixed
order and distractor choice is made from a seeded draw, so the same (roles, region, seed) always
produces byte-identical output.
"""
from __future__ import annotations

import functools
import math
import random
from pathlib import Path

from tooling.task_authoring.agent.corrections import merge_document
from tooling.task_authoring.validation import CAPACITY_MARGIN, fits_lengthwise
from tooling.task_authoring.authoring import (
    discover_assets,
    load_camera_extrinsics,
    load_droid_categories,
    load_scene_regions,
)
from tooling.task_authoring.generate_realm_droid100 import (
    ELLIPTICAL_SUPPORTS,
    SUPPORT_CLEARANCE,
    UNSAFE_SCENE_REGIONS,
    dataset_object,
    ensure_receiver_capacity,
    fit_bbox,
    overlaps,
    place_initial_relation,
    primitive,
    sample_camera_pair,
    sample_distractors,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "data" / "datasets_og391" / "behavior-1k-assets"
DEFAULT_CAMERA_EXTRINSICS = (
    REPO_ROOT / "realm" / "config" / "env" / "external_sensors" / "camera_extrinsics_droid_realm.yaml"
)
DEFAULT_REGIONS = REPO_ROOT / "realm" / "config" / "scenes" / "scenes.yaml"

#: Task types REALM has a scoring rubric for. Closed set -- see AGENTIC_PIPELINE.md section 2.2.
TASK_TYPES = ("put", "pick", "rotate", "push", "stack", "open_drawer", "close_drawer")

#: Roles the agent may assign, in the order the emitted document lists them.
ROLES = ("main", "target", "source", "distractor")

#: XY ceiling for the main object and for receivers/supports/sources. The receiver ceiling is
#: 0.28 m, wider than the batch generator's 0.17 m: real saucepots, storage boxes and frying pans
#: are 0.22-0.5 m across, and a 0.17 m cap shrank every pot to a toy. 0.28 m still leaves room for
#: the main object beside it on the narrowest (0.4 x 0.5 m) region.
MAIN_MAX_XY = (0.14, 0.16)
OTHER_MAX_XY = (0.28, 0.28)
#: Automatic clutter is drawn only from the DROID object categories in categories.yaml: real
#: tabletop objects DROID scenes contain. The full catalogue also holds hundreds of food states
#: (cooked_*, diced_*, half_*) that no DROID table shows.
DROID_CATEGORIES = REPO_ROOT / "realm" / "config" / "objects" / "categories.yaml"

#: Largest bbox dimension a distractor may have, matching realm/config/shared.py.
DISTRACTOR_MAX_DIM = 0.12


class LayoutError(ValueError):
    """Placement is impossible for this role assignment. The message is agent-readable prose."""


def usable_regions(scenes: Path = DEFAULT_REGIONS) -> list[dict]:
    """Scene regions large enough to hold a task, excluding the ones review found unusable."""
    return [
        region
        for region in load_scene_regions(scenes)
        if region["width"] >= 0.4
        and region["depth"] >= 0.4
        and region["z"] > 0
        and (region["scene"], region["support"]) not in UNSAFE_SCENE_REGIONS
    ]


def index_assets(dataset: Path = DEFAULT_DATASET) -> dict[str, list[dict]]:
    """Category -> candidate assets. Empty when the dataset is absent (report, never invent)."""
    indexed: dict[str, list[dict]] = {}
    for asset in discover_assets(dataset):
        indexed.setdefault(str(asset["category"]), []).append(asset)
    return indexed


def build_object(
    name: str,
    category: str,
    model: str,
    assets_by_category: dict[str, list[dict]],
    max_xy: tuple[float, float],
) -> tuple[dict, dict | None]:
    """One authored object config. Raises LayoutError when the (category, model) does not exist.

    Returning a config for an unindexed asset would put an unresolvable model id in a task YAML
    that OmniGibson then fails to load -- and the failure surfaces in the container, on the
    authoring host, long after the agent run that produced it.
    """
    candidates = assets_by_category.get(category)
    if not candidates:
        raise LayoutError(
            f"category {category!r} is not in the indexed catalogue. Do not invent an asset: "
            f"call search_assets for a real one, or report_ungroundable."
        )
    asset = next((item for item in candidates if str(item["model"]) == model), None)
    if asset is None:
        known = ", ".join(sorted(str(item["model"]) for item in candidates)[:8])
        raise LayoutError(
            f"model {model!r} does not exist in category {category!r} (known: {known}). "
            f"Call search_assets for {category!r} and use a returned model id."
        )
    config = dataset_object(category, assets_by_category)
    config["name"] = name
    config["model"] = model
    original = [round(float(value), 7) for value in asset["bbox"]]
    config["bounding_box"] = original
    fitted, scale = fit_bbox(original, max_xy)
    config["bounding_box"] = fitted
    config["orientation"] = [0.0, 0.0, 0.0, 1.0]
    audit = None
    if scale < 1:
        audit = {
            "name": name,
            "original_bbox": original,
            "authored_bbox": fitted,
            "scale": round(scale, 5),
            "reason": "max_footprint",
        }
    return config, audit


def build_primitive(name: str, concept: str, rgba: list[float], extent: float = 0.05) -> dict:
    """A PrimitiveObject -- the only object whose colour the config can actually guarantee."""
    config = primitive(concept, name, 0)
    config["name"] = name
    config["rgba"] = list(rgba)
    config["bounding_box"] = [extent, extent, extent]
    config["scale"] = [extent, extent, extent]
    config["orientation"] = [0.0, 0.0, 0.0, 1.0]
    return config


def place(config: dict, placed: list[dict], region: dict) -> None:
    """Place one config on the region; identical candidate order to the batch generator."""
    elliptical = region["support"] in ELLIPTICAL_SUPPORTS
    width, depth = region["width"], region["depth"]
    candidates = (
        (0.30, 0.30), (0.70, 0.70), (0.30, 0.70), (0.70, 0.30),
        (0.50, 0.50), (0.25, 0.50), (0.75, 0.50), (0.50, 0.25),
        (0.50, 0.75), (0.15, 0.50), (0.85, 0.50), (0.50, 0.85),
    )
    from tooling.task_authoring.generate_realm_droid100 import bbox_fits_support

    bx, by, _ = config["bounding_box"]
    for ux, uy in candidates:
        x = max(bx / 2, min(width - bx / 2, ux * width))
        y = max(by / 2, min(depth - by / 2, uy * depth))
        z = float(config["bounding_box"][2]) / 2 + SUPPORT_CLEARANCE
        authored_z = math.ceil(z * 10_000_000) / 10_000_000
        config["relative_bbox_position"] = [round(x, 5), round(y, 5), authored_z]
        if bbox_fits_support(x, y, config["bounding_box"], width, depth, elliptical) and not overlaps(config, placed):
            placed.append(config)
            return
    raise LayoutError(
        f"no collision-free placement for {config['name']!r} "
        f"(footprint {bx:.3f}x{by:.3f} m on a {width:.2f}x{depth:.2f} m region). "
        f"Remove a distractor or choose a smaller asset."
    )


def solve_layout(
    roles: dict,
    region: dict,
    assets_by_category: dict[str, list[dict]],
    *,
    rng: random.Random,
    distractors: int = 3,
    distractors_available: list[str] | None = None,
) -> dict:
    """Role assignment -> a complete task document, or LayoutError explaining why it is impossible.

    `roles` keys: `task_type` (required), `instruction` (required), `main` (required),
    `target`, `source`, `distractors` (list). Each object role is
    `{name, category, model}` or `{name, primitive, rgba}`.

    `instruction` is the text the emitted config declares, so every noun it depends on must be a
    role here -- `instruction_obj_to_replace` has to be a substring of it or the S-/SB-
    perturbations silently no-op.
    """
    task_type = str(roles.get("task_type", ""))
    if task_type not in TASK_TYPES:
        raise LayoutError(f"task_type {task_type!r} is not one of {', '.join(TASK_TYPES)}")
    instruction = str(roles.get("instruction", "")).strip()
    if not instruction:
        raise LayoutError("an instruction is required; it is what the task configs declare")
    if "main" not in roles:
        raise LayoutError("a `main` role is required: exactly one object is manipulated")

    # A main object that starts resting ON a source (a lid on a pot) must be at least as wide as
    # the source's opening, so it gets the receiver ceiling, not the 14 cm hand-object ceiling.
    rests_on_source = bool(roles.get("source")) and _declared_predicate(roles) in (None, "on_top_of")
    # The same holds for the GOAL of a lid task: "put the lid on the pot" needs a lid as wide as the
    # pot, so a lid that is the main object of a stack gets the receiver ceiling too.
    main_words = set(str((roles.get("main") or {}).get("category") or "").split("_"))
    lid_like = bool(main_words & {"lid", "cover", "cap"})
    lid_goal = task_type == "stack" and lid_like
    main_config, audits = _build_role(
        "main", roles["main"], assets_by_category,
        # Only a lid-like object needs to be as wide as what it rests on. Giving every resting
        # object the receiver ceiling let a "toy cart" on a box come out 28 x 20 x 30 cm.
        OTHER_MAX_XY if ((rests_on_source and lid_like) or lid_goal) else MAIN_MAX_XY)
    target_config = source_config = None
    if roles.get("target"):
        target_config, audit = _build_role("target", roles["target"], assets_by_category, OTHER_MAX_XY)
        audits += audit
    if roles.get("source"):
        source_config, audit = _build_role("source", roles["source"], assets_by_category, OTHER_MAX_XY)
        audits += audit

    if task_type in {"put", "stack"} and target_config is None:
        raise LayoutError(
            f"task_type {task_type!r} needs a `target` role: put needs a receiver to place into, "
            f"stack needs a support to place onto."
        )

    # Receivers and supports are sized against the main object before placement, exactly as the
    # batch path does; the audit is reported so a shrink is never invisible.
    capacity = None
    receiver = target_config or source_config
    # The container an elongated object is (or will be) inserted into long-axis-vertical: the put
    # target, or a pick's source that holds it `inside` at the start (place_initial_relation stands
    # it upright there, so its lying-down footprint is the wrong thing to fit).
    lengthwise_container = None
    if task_type == "put" and target_config is not None:
        lengthwise_container = target_config
    elif task_type == "pick" and target_config is None and source_config is not None \
            and _declared_predicate(roles) == "inside":
        lengthwise_container = source_config
    if (
        lengthwise_container is not None
        and fits_lengthwise(main_config["bounding_box"], lengthwise_container["bounding_box"], CAPACITY_MARGIN["put"])
    ):
        # A pen into a mug: inserted long-axis-vertical, so its cross-section is what must fit.
        # For put the object still STARTS lying on the table; only the capacity judgement changes.
        capacity = {"task_type": "put", "lengthwise_insertion": True, "uniform_scale": 1.0,
                    "main_bbox": list(main_config["bounding_box"]),
                    "target_bbox": list(lengthwise_container["bounding_box"])}
        receiver = None
    if receiver is not None:
        # The margin follows the RELATION: resting on something (stack, or a lid on a pot) needs the
        # stack margin, which lets the object be wider than its support; being inside something
        # needs the put margin. Using `put` for "lid on pot" made every lid narrower than the pot.
        if task_type == "stack" or (receiver is source_config and rests_on_source):
            capacity_type = "stack"
        elif task_type in {"put", "pick"}:
            capacity_type = "put"
        else:
            capacity_type = task_type
        capacity = ensure_receiver_capacity(main_config, receiver, capacity_type)
        if capacity["uniform_scale"] < 1:
            audits.append({
                "name": main_config["name"],
                "reason": "receiver_capacity",
                "original_bbox": capacity["main_bbox_before_capacity_fit"],
                "authored_bbox": capacity["main_bbox_after_capacity_fit"],
                "scale": capacity["uniform_scale"],
            })

    placed: list[dict] = []
    relation_audit = None
    predicate = _declared_predicate(roles)
    target_name = str((roles.get("target") or {}).get("name") or "")
    for entry in roles.get("initial_state") or []:
        if isinstance(entry, dict) and target_name and str(entry.get("object")) == target_name:
            # The target is where the main object must END UP. Starting it there makes the task
            # solved at reset: put/stack would score PLACE_INTO/PLACE_ONTO without the robot moving.
            raise LayoutError(
                f"initial_state puts the main object {entry.get('predicate')} the target "
                f"{target_name!r}, so the task would start already solved. The target is the goal; "
                f"only a `source` (what a pick removes the object from) may hold it at the start."
            )
    # Only a SOURCE holds the main object at the start ("take the marker out of the mug"). For put
    # and stack the main object and the target start apart: the goal relation is what the robot
    # must achieve, and matches how REALM_DROID10's stack_cubes / put_* configs are authored.
    anchor = source_config
    if anchor is not None:
        resolved_predicate = predicate or "on_top_of"
        place(anchor, placed, region)
        relation = "on_top" if resolved_predicate == "on_top_of" else "inside"
        relation_audit = place_initial_relation(main_config, anchor, relation)
        if relation == "inside":
            relation_audit.update(_lie_flat_if_wide(main_config, anchor))
        placed.append(main_config)
        # A source task may still have a receiver ("take the pen out of the mug and put it in the
        # bowl"); it is packed like any other object, never left unplaced.
        packable = [target_config]
    else:
        resolved_predicate = None
        packable = [main_config, target_config]
    # Pack the larger footprint first: placing a small main object centrally can strand a large
    # bowl despite ample free support area.
    for config in sorted(
        [item for item in packable if item and "relative_bbox_position" not in item],
        key=lambda item: math.prod(item["bounding_box"][:2]),
        reverse=True,
    ):
        place(config, placed, region)

    distractor_configs, distractor_audit = _place_distractors(
        roles, region, assets_by_category, placed, rng, distractors, distractors_available,
        _confusable_with(main_config, target_config, source_config),
    )

    document = {
        "task": {"type": "DummyTask", "termination_config": {}, "reward_config": {}},
        "task_type": task_type,
        "instruction": instruction,
        "instruction_obj_to_replace": _instruction_token(instruction, roles["main"]["name"]),
        "instruction_target_to_replace": (
            _instruction_token(instruction, (roles.get("target") or {}).get("name", ""))
            if roles.get("target") else ""
        ),
        "instruction_verb_to_replace": _verb_for(task_type),
        "supported_scenes": {region["scene"]: [region["support"]]},
        "main_objects": [main_config],
        "target_objects": [target_config] if target_config else [],
        "distractors": distractor_configs,
        "immutables": [source_config] if source_config is not None and predicate else [],
    }
    # Emit the declared relation. Without it the validator sees a main object resting on a source
    # and cannot tell an intended relation from a floating object, so it reports FLOATING_OBJECT on
    # a layout the solver got right. The relation is also what `target_state` documents.
    if source_config is not None and predicate:
        document["initial_state"] = [{
            "predicate": predicate,
            "subject": str(main_config["name"]),
            "object": str(source_config["name"]),
        }]
        document["target_state"] = {
            "rubric": task_type,
            "predicate": predicate,
            "subject": str(main_config["name"]),
            "object": str(source_config["name"]),
        }
    elif target_config is not None and task_type == "stack":
        # Goal only: the main object ends on the support. It does not start there.
        document["target_state"] = {
            "rubric": task_type, "predicate": "on_top_of",
            "subject": str(main_config["name"]), "object": str(target_config["name"]),
        }
    elif target_config is not None:
        document["target_state"] = {
            "rubric": task_type, "predicate": "inside",
            "subject": str(main_config["name"]), "object": str(target_config["name"]),
        }
    audit = {
        "region": {key: region[key] for key in ("scene", "support", "width", "depth", "z")},
        "resized_assets": audits,
        "receiver_capacity": capacity,
        "initial_relation": relation_audit,
        "distractors": distractor_audit,
    }
    return {"document": document, "audit": audit}


def _build_role(role: str, spec: dict, assets_by_category: dict, max_xy) -> tuple[dict, list[dict]]:
    if not isinstance(spec, dict):
        raise LayoutError(f"the {role} role must be an object, got {type(spec).__name__}")
    name = str(spec.get("name") or "").strip()
    if not name:
        raise LayoutError(f"the {role} role needs a `name`")
    if spec.get("primitive"):
        rgba = spec.get("rgba") or [0.1, 0.25, 0.9, 1.0]
        if not (isinstance(rgba, list) and len(rgba) == 4):
            raise LayoutError(f"{name!r}: rgba must be four numbers in 0..1")
        extent = float(spec.get("extent") or 0.05)
        if not 0.01 <= extent <= 0.2:
            raise LayoutError(f"{name!r}: extent must be between 0.01 and 0.2 m, got {extent}")
        return build_primitive(name, str(spec["primitive"]), [float(v) for v in rgba], extent), []
    category = str(spec.get("category") or "").strip()
    model = str(spec.get("model") or "").strip()
    if not category or not model:
        raise LayoutError(
            f"the {role} role {name!r} needs both `category` and `model` from search_assets "
            f"(or `primitive` + `rgba` for a coloured block)"
        )
    config, audit = build_object(name, category, model, assets_by_category, max_xy)
    return config, ([audit] if audit else [])


def _lie_flat_if_wide(main: dict, container: dict) -> dict:
    """An elongated object inside a container WIDER than it is long lies on the container floor.

    The shared solver always stands a pen/marker upright inside a container. In a mug the rim holds
    it; in a 25 cm pot it would simply tip over when the sim settles, so the authored start state
    is not the state the policy sees. Lying flat on the floor is where it would end up anyway.
    """
    dims = [float(value) for value in main["bounding_box"]]
    longest = max(dims)
    others = sorted(dims)[:2]
    opening = sorted(float(value) for value in container["bounding_box"][:2])
    if longest < 2 * others[1] or opening[0] < longest * 1.05:
        return {"lying_flat": False}
    # Long axis along X, lying (yaw only); the container's floor is approximated by its bbox bottom.
    main["orientation"] = [0.0, 0.0, 0.0, 1.0] if dims[0] >= dims[1] else [0.0, 0.0, 0.7071068, 0.7071068]
    container_bottom = float(container["relative_bbox_position"][2]) - float(container["bounding_box"][2]) / 2
    x, y, _ = main["relative_bbox_position"]
    main["relative_bbox_position"] = [x, y, round(container_bottom + dims[2] / 2 + 0.01, 7)]
    return {"lying_flat": True}


def _declared_predicate(roles: dict) -> str | None:
    initial = roles.get("initial_state") or []
    if not isinstance(initial, list):
        return None
    for entry in initial:
        if isinstance(entry, dict) and entry.get("predicate") in {"inside", "on_top_of"}:
            return str(entry["predicate"])
    return None


#: Categories a policy (or a person) could mistake for one another from the instruction's words.
#: A distractor from the same group as a role object makes the instruction ambiguous: "the marker"
#: next to a pen, "the cup" next to a wineglass.
CONFUSABLE_GROUPS = (
    frozenset({"marker", "pen", "pencil", "highlighter", "crayon"}),
    frozenset({"mug", "coffee_cup", "cup", "teacup", "paper_cup", "water_glass", "wineglass",
               "beaker", "tumbler"}),
    frozenset({"bowl", "mixing_bowl", "salad_bowl"}),
    frozenset({"teaspoon", "tablespoon", "spoon", "wooden_spoon"}),
    frozenset({"saucepot", "stockpot", "frying_pan", "saucepan"}),
)
#: A PrimitiveObject cube is confusable with any cube-shaped asset.
PRIMITIVE_CONFUSABLE = frozenset({"toy_dice", "cube", "block"})


def _confusable_with(*configs) -> set[str]:
    """Role categories plus every category a distractor must not share a confusion group with."""
    excluded: set[str] = set()
    for config in configs:
        if not config:
            continue
        if config.get("type") == "PrimitiveObject":
            excluded |= PRIMITIVE_CONFUSABLE
            continue
        category = str(config.get("category"))
        excluded.add(category)
        for group in CONFUSABLE_GROUPS:
            if category in group:
                excluded |= group
    return excluded


def _place_distractors(
    roles, region, assets_by_category, placed, rng, count, available, excluded,
) -> tuple[list[dict], list[dict]]:
    """Place up to `count` clutter objects, each from a distinct visual family.

    The agent names the categories it wants; the solver decides whether and where they fit. A
    distractor that cannot be placed is dropped rather than moved, because moving it would mean
    inventing a coordinate.
    """
    requested = roles.get("distractors") or []
    chosen: list[dict] = []
    audit: list[dict] = []
    families: set[str] = set()
    for spec in requested:
        if len(chosen) >= count:
            break
        if not isinstance(spec, dict):
            continue
        try:
            config, _ = _build_role("distractor", spec, assets_by_category, (DISTRACTOR_MAX_DIM, DISTRACTOR_MAX_DIM))
        except LayoutError as error:
            audit.append({"name": spec.get("name"), "dropped": str(error)})
            continue
        category = str(config.get("category", ""))
        family = _family(category)
        if rolls(category):
            audit.append({"name": config["name"], "dropped": (
                f"{category!r} is round and rolls when the scene settles")})
            continue
        if category in excluded:
            audit.append({"name": config["name"], "dropped": (
                f"{category!r} is a role object or confusable with one; the instruction would be ambiguous")})
            continue
        if family in families:
            audit.append({"name": config["name"], "dropped": f"family {family!r} already represented"})
            continue
        try:
            place(config, placed, region)
        except LayoutError as error:
            audit.append({"name": config["name"], "dropped": str(error)})
            continue
        families.add(family)
        chosen.append(config)
    if len(chosen) < count and available:
        # Top up from the catalogue pool only when the agent under-supplied; the batch generator's
        # sampler keeps family balance and category variety, which is the property being preserved.
        existing_categories = excluded | {str(item.get("category", "")) for item in chosen}
        try:
            extra = sample_distractors(
                assets_by_category, available, existing_categories,
                _counter(), _counter(), rng,
            )
        except ValueError:
            extra = []
        for candidate in extra:
            if len(chosen) >= count:
                break
            family = _family(str(candidate["category"]))
            if family in families:
                continue
            try:
                place(candidate, placed, region)
            except LayoutError:
                continue
            families.add(family)
            chosen.append(candidate)
            audit.append({"name": candidate["name"], "source": "catalogue top-up"})
    return chosen, audit


def _counter():
    from collections import Counter

    return Counter()


def _family(category: str) -> str:
    from tooling.task_authoring.generate_realm_droid100 import distractor_family

    return distractor_family(category)


def _verb_for(task_type: str) -> str:
    return {
        "put": "put", "pick": "pick", "stack": "stack", "rotate": "rotate",
        "push": "push", "open_drawer": "open", "close_drawer": "close",
    }[task_type]


def _instruction_token(instruction: str, name: str) -> str:
    """The substring of `instruction` naming this object, for the S-/SB- perturbations.

    The perturbations do `instruction.replace(token, ...)`. A token that is not a literal
    substring of the instruction makes the rewrite a silent no-op, so fall back to the last word
    of the object name and let `validation.py` flag a token the instruction does not contain.
    """
    lowered = instruction.lower()
    words = [part for part in str(name).replace("_", " ").split() if part]
    for size in range(len(words), 0, -1):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start:start + size])
            if phrase and phrase in lowered:
                return phrase
    return words[-1] if words else str(name)


def build_document(
    roles: dict,
    *,
    region_index: int = 0,
    seed: int = 100,
    dataset: Path = DEFAULT_DATASET,
    camera_extrinsics: Path = DEFAULT_CAMERA_EXTRINSICS,
    scenes: Path = DEFAULT_REGIONS,
    assets_by_category: dict | None = None,
    corrections_dir: Path | None = None,
    provenance: dict | None = None,
) -> dict:
    """Full pipeline for one agent proposal: index -> solve -> cameras -> corrections.

    The correction store is merged AFTER the solve and before validation, so a reviewed fix is
    applied to the same geometry the validator will judge (AGENTIC_PIPELINE.md section 4.5).
    """
    regions = usable_regions(scenes)
    if not regions:
        raise LayoutError("no usable scene region: check realm/config/scenes/scenes.yaml")
    if not 0 <= region_index < len(regions):
        raise LayoutError(
            f"region_index {region_index} is out of range; there are {len(regions)} usable regions"
        )
    region = regions[region_index]
    assets = assets_by_category if assets_by_category is not None else index_assets(dataset)
    if not assets:
        raise LayoutError(
            f"no assets indexed under {dataset}. Set --dataset to the OmniGibson behavior-1k-assets "
            f"tree, or pass assets_by_category explicitly."
        )
    rng = random.Random(seed)
    solved = solve_layout(
        roles, region, assets,
        rng=rng,
        distractors_available=_eligible_distractors(assets),
    )
    document = solved["document"]
    camera_rng = random.Random(seed + 1)
    poses = _camera_poses(Path(camera_extrinsics))
    if not poses:
        raise LayoutError(f"no camera poses in {camera_extrinsics}")
    sampled = sample_camera_pair(poses, camera_rng)
    document["camera_extrinsics"] = {
        key: {k: v for k, v in value.items() if k != "source"} for key, value in sampled.items()
    }
    audit = dict(solved["audit"])
    audit["camera_extrinsic_sources"] = {key: value["source"] for key, value in sampled.items()}
    if provenance:
        # Set BEFORE the merge: the store is keyed by provenance.task_id (the ORIGINAL instruction),
        # and a document whose instruction the agent rewrote would otherwise look itself up under
        # the rewritten text and never find the corrections recorded against it.
        document["provenance"] = {key: value for key, value in provenance.items() if value is not None}
    applied = merge_document(document, corrections_dir=corrections_dir)
    if applied:
        audit["corrections_applied"] = applied
    return {"document": document, "audit": audit}


@functools.lru_cache(maxsize=4)
def _camera_poses(path: Path) -> dict:
    """Load the 17 MB extrinsics file once per process.

    Re-reading it on every propose_layout call leaked: `authoring.sample_opposite_camera_pair`
    memoizes its pair list keyed by `id(poses)`, so every fresh dict pinned another copy and a
    100-instruction run died of MemoryError around instruction 30. Same content, same draws: this
    changes no number, only how often the file is parsed.
    """
    return load_camera_extrinsics(path)


#: Round objects roll on their own when the sim settles: a volleyball drifted 3.2 cm and a pear
#: 9.2 cm in the first pilot render, failing two otherwise-correct tasks as UNSTABLE.
ROLLING_WORDS = frozenset({
    "ball", "orange", "apple", "pear", "peach", "plum", "lemon", "lime", "tomato", "onion",
    "potato", "egg", "pomelo", "grapefruit", "melon", "coconut", "kiwi", "mango", "apricot",
    "nectarine", "tangerine", "clementine", "grape", "cherry", "globe", "marble", "pearl",
    # Found in the first 104-task export: still drawn as clutter before this list grew.
    "papaya", "avocado", "lychee", "fig", "date", "olive", "kumquat", "persimmon", "pomegranate",
    "acorn", "pecan", "almond", "pea", "bead", "button", "bubble",
})


def rolls(category: str) -> bool:
    """True for round objects, judged by the HEAD noun: a bottle_of_olive_oil or a
    jar_of_strawberry_jam is a container and stands still; a cherry_tomato rolls."""
    category = str(category)
    if category.startswith(("half_", "sliced_", "diced_")):
        return False
    head = category.split("_of_", 1)[0] if "_of_" in category else category.rsplit("_", 1)[-1]
    return head in ROLLING_WORDS or head.endswith(("ball", "berry", "nut"))


def _eligible_distractors(assets_by_category: dict[str, list[dict]]) -> list[str]:
    """Categories usable as automatic clutter: DROID object categories that pass the size gate.

    Falls back to the size gate alone when fewer than three DROID categories are indexed, which only
    happens with a small synthetic catalogue (tests, the seed catalogue).
    """
    sized = [category for category in _size_gated(assets_by_category) if not rolls(category)]
    droid = set(load_droid_categories(DROID_CATEGORIES))
    preferred = [category for category in sized if category in droid]
    return preferred if len(preferred) >= 3 else sized


def _size_gated(assets_by_category: dict[str, list[dict]]) -> list[str]:
    """Categories small enough to be clutter, matching the batch generator's size gate."""
    eligible = []
    for category, candidates in assets_by_category.items():
        if not candidates:
            continue
        smallest = min(candidates, key=lambda item: math.prod(item["bbox"][:2]))
        # A zero extent is a planar/degenerate mesh (the full dataset has a signpost with x=0); it is
        # not placeable clutter, and fit_bbox would divide by it.
        if min(float(value) for value in smallest["bbox"]) <= 0:
            continue
        if max(float(value) for value in smallest["bbox"][:2]) <= 0.24 and float(smallest["bbox"][2]) <= 0.35:
            eligible.append(category)
    return sorted(eligible)
