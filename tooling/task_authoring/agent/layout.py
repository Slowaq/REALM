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
from tooling.task_authoring.validation import (
    CAPACITY_MARGIN,
    fits_lengthwise,
    resting_z,
    vertical_extent,
)
from tooling.task_authoring.authoring import (
    discover_assets,
    load_camera_extrinsics,
    load_droid_categories,
    load_scene_regions,
)
from tooling.task_authoring.generate_realm_droid100 import (
    ELLIPTICAL_SUPPORTS,
    UNSAFE_SCENE_REGIONS,
    dataset_object,
    ensure_receiver_capacity,
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

#: Every object is scaled uniformly to a realistic real-world size instead of being capped:
#: real_sizes.json holds, per category, the typical LONGEST dimension of that object on a kitchen,
#: dining or desk table, with the reason. A task overrides it for one object (`size_m` +
#: `size_reason` in the role) when the instruction means another variant, e.g. a toy pan.
#: The 14 x 16 cm cap this replaces (from REALM's batch generator) shrank pots into toys,
#: miniaturised a third of the clutter and thinned pens and spoons until they could not be grasped.
REAL_SIZES_PATH = Path(__file__).resolve().parent / "real_sizes.json"
#: An agent-chosen size may differ from the category's typical size by at most this factor either
#: way (a toy pan is a third of a real one); anything further is a typo or a different object.
SIZE_OVERRIDE_RANGE = 3.0
#: A lid is scaled to this fraction over its pot's narrower horizontal side, so it rests on the rim.
LID_COVER_FACTOR = 1.04
#: Automatic clutter is drawn only from the DROID object categories in categories.yaml: real
#: tabletop objects DROID scenes contain. The full catalogue also holds hundreds of food states
#: (cooked_*, diced_*, half_*) that no DROID table shows.
DROID_CATEGORIES = REPO_ROOT / "realm" / "config" / "objects" / "categories.yaml"



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


@functools.lru_cache(maxsize=4)
def real_sizes(path: Path = REAL_SIZES_PATH) -> dict[str, dict]:
    """Category -> {size_m, reason}. Empty when the file is absent (objects then keep natural size)."""
    import json

    try:
        return dict(json.loads(Path(path).read_text(encoding="utf-8")).get("sizes") or {})
    except (OSError, ValueError):
        return {}


def realistic_size(category: str, size_m: float | None = None,
                   size_reason: str | None = None) -> tuple[float | None, str]:
    """The longest dimension an object of `category` should have, and where that number came from."""
    typical = real_sizes().get(str(category))
    if size_m is not None:
        size_m = float(size_m)
        if not str(size_reason or "").strip():
            raise LayoutError(f"size_m={size_m} for {category!r} needs a size_reason saying why this object "
                              f"is not the typical size")
        if typical and not (1 / SIZE_OVERRIDE_RANGE <= size_m / float(typical["size_m"]) <= SIZE_OVERRIDE_RANGE):
            raise LayoutError(
                f"size_m={size_m} for {category!r} is more than {SIZE_OVERRIDE_RANGE:g}x away from its typical "
                f"{typical['size_m']} m ({typical['reason']}); that is a different object, not a variant")
        return size_m, f"task override: {size_reason}"
    if typical:
        return float(typical["size_m"]), f"typical size: {typical['reason']}"
    return None, "natural size: no typical size on record for this category"


def scale_to_size(bbox, size_m: float | None) -> tuple[list[float], float]:
    """Uniform scale so the longest dimension equals `size_m` (natural size when None)."""
    original = [float(v) for v in bbox]
    if size_m is None or max(original) <= 0:
        return [round(v, 7) for v in original], 1.0
    scale = float(size_m) / max(original)
    return [round(v * scale, 7) for v in original], scale


def build_object(
    name: str,
    category: str,
    model: str,
    assets_by_category: dict[str, list[dict]],
    size_m: float | None = None,
    size_reason: str | None = None,
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
    target, source = realistic_size(category, size_m, size_reason)
    fitted, scale = scale_to_size(original, target)
    config["bounding_box"] = fitted
    config["orientation"] = [0.0, 0.0, 0.0, 1.0]
    audit = {
        "name": name,
        "original_bbox": original,
        "authored_bbox": fitted,
        "scale": round(scale, 5),
        "reason": "realistic_size",
        "size_source": source,
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
    from tooling.task_authoring.generate_realm_droid100 import SUPPORT_EDGE_CLEARANCE, bbox_fits_support

    from tooling.task_authoring.validation import oriented_footprint

    # The footprint as ORIENTED: an object turned 90 degrees to fit its receiver occupies its rotated
    # extent, which is what the validator measures; placing by the unrotated box overlapped
    # neighbours and overhung the table edge once objects were realistically sized.
    bx, by = oriented_footprint(config["bounding_box"], config.get("orientation"))
    footprint = [bx, by, config["bounding_box"][2]]
    # The 12 preferred spots first (same order as the batch generator), then a deterministic 2 cm grid:
    # at realistic sizes a 30 cm sink and a 15 cm bowl fit a 50 x 60 cm table, but not on those 12.
    grid = [(i / 20, j / 20) for j in range(21) for i in range(21)]
    # 0.1 mm inside the clearance: positions are rounded to 5 decimals, and a spot exactly ON the
    # clearance read as overhanging it after rounding.
    edge = SUPPORT_EDGE_CLEARANCE + 1e-4
    for ux, uy in list(candidates) + grid:
        # Clamp INSIDE the edge clearance bbox_fits_support demands; clamping to the bare edge made
        # every clamped spot fail for any object wider than a third of the table.
        x = max(bx / 2 + edge, min(width - bx / 2 - edge, ux * width))
        y = max(by / 2 + edge, min(depth - by / 2 - edge, uy * depth))
        # Just above the region's MEASURED surface, not 5 cm above the spawn plane: the spawn plane
        # sits up to 11 cm above the real table, and a 10-16 cm drop tipped and rolled objects.
        z = resting_z(float(config["bounding_box"][2]), region.get("scene"), region.get("support"))
        authored_z = math.ceil(z * 10_000_000) / 10_000_000
        config["relative_bbox_position"] = [round(x, 5), round(y, 5), authored_z]
        if bbox_fits_support(x, y, footprint, width, depth, elliptical) and not _overlaps_oriented(config, placed):
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

    # Only a SOURCE the object starts on (a lid on a pot) uses the stack margin below.
    rests_on_source = bool(roles.get("source")) and _declared_predicate(roles) in (None, "on_top_of")
    main_config, audits = _build_role("main", roles["main"], assets_by_category)
    target_config = source_config = None
    if roles.get("target"):
        target_config, audit = _build_role("target", roles["target"], assets_by_category)
        audits += audit
    if roles.get("source"):
        source_config, audit = _build_role("source", roles["source"], assets_by_category)
        audits += audit
    # A lid's size is set by the pot it closes, not by a typical size: a 24 cm lid on a 20 cm pot
    # slides off, on a 28 cm pot it falls in. Unless the agent sized it explicitly, a lid that starts
    # on its pot or must end on it is scaled to cover that pot's narrower side.
    main_words = set(str((roles.get("main") or {}).get("category") or "").split("_"))
    lid_partner = (source_config if rests_on_source else None) or (target_config if task_type == "stack" else None)
    if (main_words & {"lid", "cover"} and lid_partner is not None
            and "size_m" not in (roles.get("main") or {}) and lid_partner.get("type") != "PrimitiveObject"):
        opening = min(float(v) for v in lid_partner["bounding_box"][:2]) * LID_COVER_FACTOR
        bbox = main_config["bounding_box"]
        factor = opening / min(float(v) for v in bbox[:2])
        main_config["bounding_box"] = [round(float(v) * factor, 7) for v in bbox]
        audits.append({"name": main_config["name"], "reason": "lid_fits_its_pot", "scale": round(factor, 5),
                       "size_source": f"covers {lid_partner['name']!r} ({opening:.3f} m)"})

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
    if receiver is not None and receiver is source_config and rests_on_source:
        # Resting ON its source (a lid on a pot, a plate across a bowl) the object may be wider than
        # the source: it covers it. LID_SMALLER_THAN_OPENING and RIM_OVERHANG judge that geometry.
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
            # Shrinking the object until it fits would make it a toy. At realistic sizes it either
            # fits or the pairing is wrong: say so, and let the agent choose.
            raise LayoutError(
                f"{main_config['name']!r} at its realistic size "
                f"{[round(v, 3) for v in capacity['main_bbox_before_capacity_fit']]} does not fit "
                f"{receiver['name']!r} {[round(v, 3) for v in receiver['bounding_box']]} "
                f"({capacity_type} margin {CAPACITY_MARGIN.get(capacity_type, 1.0)}). Choose a larger "
                f"receiver model, give the receiver a justified size_m, or a smaller variant of the object."
            )

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
        if resolved_predicate == "on_top_of" and _wider(main_config, anchor):
            # Something wider rests on the anchor (a lid on a pot, a plate across a bowl): choose the
            # anchor's spot by the WIDER footprint, or the overhang can hang past the table edge.
            proxy = dict(anchor, bounding_box=[max(float(a), float(m)) for a, m in
                                               zip(anchor["bounding_box"][:2], main_config["bounding_box"][:2])]
                         + [anchor["bounding_box"][2]])
            place(proxy, placed, region)
            anchor["relative_bbox_position"] = list(proxy["relative_bbox_position"])
        else:
            place(anchor, placed, region)
        relation = "on_top" if resolved_predicate == "on_top_of" else "inside"
        relation_audit = place_initial_relation(main_config, anchor, relation)
        if relation == "inside":
            _lower_onto_measured_surface(main_config, anchor, region)
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


def _build_role(role: str, spec: dict, assets_by_category: dict) -> tuple[dict, list[dict]]:
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
    config, audit = build_object(name, category, model, assets_by_category,
                                 spec.get("size_m"), spec.get("size_reason"))
    return config, ([audit] if audit else [])


def _lower_onto_measured_surface(main: dict, container: dict, region: dict) -> None:
    """Redo place_initial_relation's floor term against the measured surface.

    It keeps the object at least `extent/2 + 5 cm` above the SPAWN plane. With the container now
    resting on the real table (up to 11 cm lower), that floor would leave a pen hanging above its
    mug instead of with its lower quarter inside the opening.
    """
    extent = vertical_extent(main["bounding_box"], main.get("orientation"))
    source_top = float(container["relative_bbox_position"][2]) + float(container["bounding_box"][2]) / 2
    z = max(resting_z(extent, region.get("scene"), region.get("support")), source_top + extent / 4)
    x, y, _ = main["relative_bbox_position"]
    main["relative_bbox_position"] = [x, y, round(z, 7)]


#: Gap between the rim and the lower face of an object dropped into a container.
RIM_DROP_CLEARANCE_M = 0.005


def _overlaps_oriented(candidate: dict, placed: list[dict], margin: float = 0.012) -> bool:
    """generate_realm_droid100.overlaps, on oriented footprints."""
    from tooling.task_authoring.validation import oriented_footprint

    x, y, _ = candidate["relative_bbox_position"]
    cx, cy = oriented_footprint(candidate["bounding_box"], candidate.get("orientation"))
    for other in placed:
        ox, oy = oriented_footprint(other["bounding_box"], other.get("orientation"))
        if (abs(x - other["relative_bbox_position"][0]) < (cx + ox) / 2 + margin
                and abs(y - other["relative_bbox_position"][1]) < (cy + oy) / 2 + margin):
            return True
    return False


def _wider(config: dict, other: dict) -> bool:
    return any(float(c) > float(o) for c, o in zip(config["bounding_box"][:2], other["bounding_box"][:2]))


def _lie_flat_if_wide(main: dict, container: dict) -> dict:
    """An elongated object inside a container WIDER than it is long lies on the container floor.

    The shared solver always stands a pen/marker upright inside a container. In a mug the rim holds
    it; in a 25 cm pot it would simply tip over when the sim settles, so the authored start state
    is not the state the policy sees. Lying flat on the floor is where it would end up anyway.

    It is authored lying just ABOVE THE RIM and dropped in by the settle, never at the bbox floor:
    the bbox bottom is not the container's floor (a storage box has a thick base, a bowl a curved
    one), and sunglasses authored 1 cm above a box's bbox bottom were ejected off the table while a
    match box at that height in a bowl clipped through its wall.
    """
    dims = [float(value) for value in main["bounding_box"]]
    longest = max(dims)
    others = sorted(dims)[:2]
    opening = sorted(float(value) for value in container["bounding_box"][:2])
    if longest < 2 * others[1] or opening[0] < longest * 1.05:
        return {"lying_flat": False}
    if dims[2] >= max(dims[:2]):
        # Upright is the asset's own resting pose (a salt shaker): place_initial_relation already
        # lowers its bottom quarter into the opening, and it drops onto the floor standing.
        return {"lying_flat": False}
    # Long axis along X, lying (yaw only), its footprint inside the opening so it drops straight in.
    main["orientation"] = [0.0, 0.0, 0.0, 1.0] if dims[0] >= dims[1] else [0.0, 0.0, 0.7071068, 0.7071068]
    container_top = float(container["relative_bbox_position"][2]) + float(container["bounding_box"][2]) / 2
    x, y, _ = main["relative_bbox_position"]
    main["relative_bbox_position"] = [x, y, round(container_top + dims[2] / 2 + RIM_DROP_CLEARANCE_M, 7)]
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
            config, _ = _build_role("distractor", spec, assets_by_category)
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
            natural = next((a["bbox"] for a in assets_by_category.get(str(candidate["category"]), [])
                            if str(a["model"]) == str(candidate["model"])), candidate["bounding_box"])
            size, _ = realistic_size(str(candidate["category"]))
            candidate["bounding_box"], _ = scale_to_size(natural, size)
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
    first, redraws = sampled, 0
    # A review that finds the main object hidden or badly framed asks for the NEXT workspace-facing
    # pair from the same stream: deterministic, recorded in the proposal, and 0 leaves every other
    # task's cameras exactly as they were.
    camera_skip = max(0, int(roles.get("camera_skip") or 0))
    skip = camera_skip
    redraw_limit = CAMERA_MAX_REDRAWS * (1 + camera_skip)
    # Some DROID calibrations are wrong or the camera was mounted looking past the table; the
    # pink-plate render showed mostly ceiling. Redraw from the same stream until both cameras look
    # at the workspace, so a task whose first pair already does keeps exactly the same cameras.
    while True:
        if all(looks_at_workspace(pose) for pose in sampled.values()):
            if skip == 0:
                break
            skip -= 1
        redraws += 1
        if redraws > redraw_limit:
            sampled = first
            break
        sampled = sample_camera_pair(poses, camera_rng)
    document["camera_extrinsics"] = {
        key: {k: v for k, v in value.items() if k != "source"} for key, value in sampled.items()
    }
    audit = dict(solved["audit"])
    audit["camera_extrinsic_sources"] = {key: value["source"] for key, value in sampled.items()}
    audit["camera_redraws"] = redraws if redraws <= redraw_limit else "none looked at the workspace"
    if camera_skip:
        audit["camera_skip"] = camera_skip
    if provenance:
        # Set BEFORE the merge: the store is keyed by provenance.task_id (the ORIGINAL instruction),
        # and a document whose instruction the agent rewrote would otherwise look itself up under
        # the rewritten text and never find the corrections recorded against it.
        document["provenance"] = {key: value for key, value in provenance.items() if value is not None}
    applied = merge_document(document, corrections_dir=corrections_dir)
    if applied:
        audit["corrections_applied"] = applied
    return {"document": document, "audit": audit}


#: Where the manipulated objects are, robot-base frame (the frame the DROID extrinsics are in).
WORKSPACE_POINT = (0.5, 0.0, 0.05)
#: A camera whose optical axis misses that point by more than this frames the table at its edge.
CAMERA_MAX_OFF_AXIS_DEG = 30.0
#: ...and one that does not look down at least this much sees wall and ceiling.
CAMERA_MIN_DOWN_PITCH_DEG = 10.0
CAMERA_MAX_REDRAWS = 50


def looks_at_workspace(pose: dict) -> bool:
    """True when an OmniGibson-convention camera (XYZW, -Z forward) points down at the workspace."""
    x, y, z, w = (float(value) for value in pose["rot"])
    # The camera's -Z axis rotated into the base frame.
    forward = (-(2 * (x * z + y * w)), -(2 * (y * z - x * w)), -(1 - 2 * (x * x + y * y)))
    to_point = [target - float(value) for target, value in zip(WORKSPACE_POINT, pose["pos"])]
    norm = math.sqrt(sum(value * value for value in to_point)) or 1.0
    cosine = sum(f * t for f, t in zip(forward, to_point)) / norm
    off_axis = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    down = -math.degrees(math.asin(max(-1.0, min(1.0, forward[2]))))
    return off_axis <= CAMERA_MAX_OFF_AXIS_DEG and down >= CAMERA_MIN_DOWN_PITCH_DEG


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
