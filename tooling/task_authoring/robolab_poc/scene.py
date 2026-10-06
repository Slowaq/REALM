"""Scene spec (objects + relations, no numbers) -> RoboLab's solver -> a REALM task config.

The spec is what a coding agent writes after reading GENERATE_SCENE.md:

    {
      "name": "put_apple_in_bowl_mug_left",
      "instruction": "put the apple in the bowl",
      "task_type": "put",                                  # put | pick | rotate | stack
      "region": "Pomaria_1_int / Kitchen_Counter",         # optional; see `cli.py regions`
      "objects": [
        {"name": "apple", "category": "apple", "role": "main"},
        {"name": "bowl", "category": "bowl", "role": "target"},
        {"name": "mug", "category": "mug", "role": "distractor"}
      ],
      "relations": [
        {"type": "left-of", "object": "mug", "reference": "bowl"}
      ]
    }

Relations are robot-relative: `left-of` / `right-of` (the robot's left/right), `in-front-of`
(closer to the robot), `behind` (farther from it), `in` (`container`) and `on` (`support`).
The spec carries no coordinates, distances, scales or yaws: code derives every number, following
AUTHORING_RULES.md (50 mm support clearance, uniform scale, yaw only, receiver capacity).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field

from tooling.task_authoring.generate_realm_droid100 import (
    RELATION_CLEARANCE,
    SUPPORT_CLEARANCE,
    SUPPORT_EDGE_CLEARANCE,
    bbox_fits_support,
    ensure_receiver_capacity,
    fit_bbox,
)
from tooling.task_authoring.robolab_poc._vendor.robolab_scene_gen import (
    FeedbackSystem,
    ObjectState,
    PhysicalSolver,
    PlaceOnBasePredicate,
    PredicateType,
    SpatialSolver,
)
from tooling.task_authoring.robolab_poc._vendor.robolab_scene_gen.predicates import (
    PlaceInPredicate,
    PlaceOnPredicate,
    RelativePositionPredicate,
)
from tooling.task_authoring.robolab_poc.catalog import by_category
from tooling.task_authoring.robolab_poc.frames import RegionFrame, find_region

ROBOLAB_COMMIT = "ad45d4f"
TASK_TYPES = ("put", "pick", "rotate", "stack")
ROLES = ("main", "target", "source", "distractor")
SPATIAL = {
    # REALM spec word -> RoboLab predicate in the solver's robot frame (+a forward, +b left)
    "left-of": PredicateType.LEFT_OF,
    "right-of": PredicateType.RIGHT_OF,
    "behind": PredicateType.FRONT_OF,       # RoboLab "front" is +X: farther from the robot
    "in-front-of": PredicateType.BACK_OF,   # closer to the robot
}
PHYSICAL = ("in", "on")
RELATION_GAP = 0.03          # edge-to-edge gap the relation asks for before collision repair
OVERLAP_MARGIN = 0.012       # generate_realm_droid100.overlaps()
CONTAINER_MOUTH = 0.43       # RoboLab's usable mouth radius as a fraction of container extent
STACK_SUPPORT_RATIO = 0.65   # AUTHORING_RULES.md stack capacity
MAX_XY = {"main": (0.14, 0.16), "receiver": (0.17, 0.17), "distractor": (0.10, 0.10)}
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
YAW_0 = [0.0, 0.0, 0.0, 1.0]
YAW_90 = [0.0, 0.0, 0.7071068, 0.7071068]


class SpecError(ValueError):
    """The spec is malformed or asks for something this PoC cannot represent."""


@dataclass
class Result:
    ok: bool
    document: dict | None = None
    findings: list[dict] = field(default_factory=list)
    layout: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    error: str | None = None

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "findings": self.findings,
            "layout": self.layout,
        }


def _finding(code: str, severity: str, message: str, **extra) -> dict:
    return {"code": code, "severity": severity, "message": message, **extra}


def _yaw90(cfg: dict) -> bool:
    return cfg.get("orientation", YAW_0) == YAW_90


def _world_footprint(cfg: dict) -> tuple[float, float]:
    bx, by = float(cfg["bounding_box"][0]), float(cfg["bounding_box"][1])
    return (by, bx) if _yaw90(cfg) else (bx, by)


def validate_spec(spec: dict) -> None:

    for key in ("name", "instruction", "task_type", "objects"):
        if key not in spec:
            raise SpecError(f"spec is missing {key!r}")
    if not NAME_PATTERN.match(str(spec["name"])):
        raise SpecError(f"name {spec['name']!r} must be snake_case")
    if spec["task_type"] not in TASK_TYPES:
        raise SpecError(f"task_type {spec['task_type']!r} not in {TASK_TYPES}")
    names = [obj.get("name") for obj in spec["objects"]]
    if len(set(names)) != len(names):
        raise SpecError(f"duplicate object names: {names}")
    for obj in spec["objects"]:
        if not NAME_PATTERN.match(str(obj.get("name", ""))):
            raise SpecError(f"object name {obj.get('name')!r} must be snake_case")
        if obj.get("role") not in ROLES:
            raise SpecError(f"{obj['name']}: role {obj.get('role')!r} not in {ROLES}")
        if not obj.get("category"):
            raise SpecError(f"{obj['name']}: category is required")
        forbidden = {"position", "x", "y", "z", "scale", "bounding_box", "orientation", "yaw"} & set(obj)
        if forbidden:
            raise SpecError(f"{obj['name']}: the spec may not set {sorted(forbidden)}; code places objects")
    roles = [obj["role"] for obj in spec["objects"]]
    if roles.count("main") != 1:
        raise SpecError("exactly one object must have role 'main'")
    if roles.count("target") > 1 or roles.count("source") > 1:
        raise SpecError("at most one 'target' and one 'source'")
    needs_target = spec["task_type"] in ("put", "stack")
    if needs_target != ("target" in roles):
        raise SpecError(
            f"task_type {spec['task_type']!r} {'needs' if needs_target else 'cannot have'} a 'target' object")

    by_name = {obj["name"]: obj for obj in spec["objects"]}
    physical_subjects, spatial_refs = set(), set()
    for rel in spec.get("relations", []):
        kind = rel.get("type")
        if kind in SPATIAL:
            for key in ("object", "reference"):
                if rel.get(key) not in by_name:
                    raise SpecError(f"{kind}: unknown {key} {rel.get(key)!r}")
            if rel["object"] == rel["reference"]:
                raise SpecError(f"{kind}: object and reference are the same")
            if "distance" in rel:
                raise SpecError(f"{kind}: distance is derived from the objects' sizes, do not set it")
            spatial_refs.update((rel["object"], rel["reference"]))
        elif kind in PHYSICAL:
            holder_key = "container" if kind == "in" else "support"
            subject, holder = rel.get("object"), rel.get(holder_key)
            if subject not in by_name or holder not in by_name:
                raise SpecError(f"{kind}: unknown object {subject!r} or {holder_key} {holder!r}")
            if subject in physical_subjects:
                raise SpecError(f"{subject} is the subject of more than one in/on relation")
            physical_subjects.add(subject)
            if by_name[subject]["role"] in ("target", "source"):
                raise SpecError(f"{subject}: a {by_name[subject]['role']} must stand on the support itself")
            if by_name[subject]["role"] == "main" and by_name[holder]["role"] != "source":
                raise SpecError(
                    f"main object {subject} may start {kind} only its 'source' (e.g. 'take X out of Y'); "
                    f"{holder} has role {by_name[holder]['role']!r}")
        else:
            raise SpecError(f"unknown relation type {kind!r}; use {sorted(SPATIAL) + list(PHYSICAL)}")
    holders = {rel.get("container") or rel.get("support") for rel in spec.get("relations", []) if rel["type"] in PHYSICAL}
    if physical_subjects & holders:
        raise SpecError("nested in/on relations are not supported")
    if physical_subjects & spatial_refs:
        raise SpecError(
            f"{sorted(physical_subjects & spatial_refs)} are placed in/on something and cannot also take "
            "part in left-of/right-of/in-front-of/behind")
    if "source" in roles and not any(
        by_name[rel["object"]]["role"] == "main" for rel in spec.get("relations", []) if rel["type"] in PHYSICAL
    ):
        raise SpecError("a 'source' object needs an in/on relation that puts the main object in/on it")


def _choose_asset(obj: dict, assets: dict[str, list[dict]], prefer_large: bool) -> dict:

    candidates = assets.get(obj["category"], [])
    if not candidates:
        raise SpecError(f"{obj['name']}: category {obj['category']!r} is not in the catalogue (ungroundable)")
    if obj.get("model"):
        match = [asset for asset in candidates if asset["model"] == obj["model"]]
        if not match:
            raise SpecError(f"{obj['name']}: model {obj['model']!r} is not a {obj['category']} in the catalogue")
        return match[0]
    upright = [asset for asset in candidates if asset.get("upright_in_source", True)] or candidates
    chooser = max if prefer_large else min
    return chooser(upright, key=lambda asset: (math.prod(asset["bbox"][:2]), asset["model"]))


def _object_configs(spec: dict, catalog: dict) -> tuple[dict[str, dict], list[dict]]:

    assets = by_category(catalog)
    holders = {rel.get("container") or rel.get("support") for rel in spec.get("relations", []) if rel["type"] in PHYSICAL}
    configs, audit = {}, []
    for obj in spec["objects"]:
        receiver = obj["role"] in ("target", "source") or obj["name"] in holders
        asset = _choose_asset(obj, assets, prefer_large=receiver)
        limit = MAX_XY["main"] if obj["role"] == "main" else MAX_XY["receiver"] if receiver else MAX_XY["distractor"]
        original = [round(float(value), 7) for value in asset["bbox"]]
        fitted, scale = fit_bbox(original, limit)
        configs[obj["name"]] = {
            "type": "DatasetObject",
            "name": obj["name"],
            "category": obj["category"],
            "model": asset["model"],
            "bounding_box": fitted,
            "orientation": list(YAW_0),
        }
        if scale < 1:
            audit.append({"name": obj["name"], "reason": "fit_role_limit", "original_bbox": original,
                          "authored_bbox": fitted, "scale": round(scale, 5)})
        if not asset.get("upright_in_source", True):
            audit.append({"name": obj["name"], "reason": "bbox_from_rotated_config",
                          "note": f"{asset.get('seen_in')} rolls/pitches this model; its extent may not be upright"})
    return configs, audit


@contextlib.contextmanager
def _seeded(seed: int):
    """RoboLab's solver draws from the global `random` module; isolate and seed it."""
    state = random.getstate()
    random.seed(seed)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        random.setstate(state)


class _YawLimitedPhysicalSolver(PhysicalSolver):
    """AUTHORING_RULES.md: yaw only, normally 0 or 90 degrees. Everything else is RoboLab's."""

    def _candidate_local_yaws(self, world_yaw, container_yaw):
        return [0.0, 90.0]


def solve(spec: dict, catalog: dict, *, seed: int = 0, margin: float = 0.015,
          frame: RegionFrame | None = None, attempts: int = 50) -> Result:
    """Solve with seeds seed, seed+1, ... and return the first draft without error findings.

    RoboLab anchors reference objects at random and its bounds clamp runs after its collision
    check, so a draft can come back overlapping or with a relation pushed off the support.
    Retrying with the next seed is RoboLab's own remedy ("adjust initial positions").
    """
    validate_spec(spec)
    frame = frame or find_region(spec.get("region"))
    tried = []
    result = None
    for offset in range(max(1, attempts)):
        result = _solve_once(spec, catalog, seed=seed + offset, margin=margin, frame=frame)
        tried.append({"seed": seed + offset, "ok": result.ok, "error": result.error,
                      "codes": sorted({f["code"] for f in result.findings if f["severity"] == "error"})})
        if result.ok:
            break
    result.provenance["attempts"] = tried
    return result


def _solve_once(spec: dict, catalog: dict, *, seed: int, margin: float, frame: RegionFrame) -> Result:

    by_name = {obj["name"]: obj for obj in spec["objects"]}
    relations = spec.get("relations", [])
    configs, resize_audit = _object_configs(spec, catalog)
    main = next(name for name, obj in by_name.items() if obj["role"] == "main")
    target = next((name for name, obj in by_name.items() if obj["role"] == "target"), None)
    source = next((name for name, obj in by_name.items() if obj["role"] == "source"), None)

    capacity = []
    if target:
        capacity.append(ensure_receiver_capacity(configs[main], configs[target], spec["task_type"]))
    for rel in relations:
        if rel["type"] in PHYSICAL and rel["object"] == main:
            holder = rel.get("container") or rel.get("support")
            capacity.append(ensure_receiver_capacity(
                configs[main], configs[holder], "put" if rel["type"] == "in" else "stack"))

    dims = {}
    for name, cfg in configs.items():
        along_a, along_b = frame.solver_footprint(_world_footprint(cfg))
        dims[name] = (along_a, along_b, float(cfg["bounding_box"][2]))

    states = {name: ObjectState(name=name) for name in configs}
    subjects = {rel["object"]: rel for rel in relations if rel["type"] in PHYSICAL}
    for name, state in states.items():
        if name not in subjects:
            state.predicates.append(PlaceOnBasePredicate(name, yaw=0.0))
    for rel in relations:
        if rel["type"] in SPATIAL:
            obj, ref = rel["object"], rel["reference"]
            axis = 1 if rel["type"] in ("left-of", "right-of") else 0
            distance = (dims[obj][axis] + dims[ref][axis]) / 2 + RELATION_GAP
            states[obj].predicates.append(RelativePositionPredicate(obj, ref, SPATIAL[rel["type"]], distance))
    containers: dict[str, list[str]] = {}
    for name, rel in subjects.items():
        if rel["type"] == "in":
            containers.setdefault(rel["container"], []).append(name)
        else:
            states[name].predicates.append(PlaceOnPredicate(name, rel["support"]))
    for container, members in containers.items():
        predicate = PlaceInPredicate(members, container)
        for member in members:
            states[member].predicates.append(predicate)

    # +1 mm so rounding authored positions to 0.01 mm cannot cross the 25 mm edge clearance.
    bounds = frame.solver_bounds(inset=SUPPORT_EDGE_CLEARANCE + 0.001)
    with _seeded(seed):
        ok, message = SpatialSolver(table_bounds=bounds, collision_margin=margin).solve(states, dims)
        if ok:
            ok, message = _YawLimitedPhysicalSolver().solve(states, dims, {}, "")
        grammar = FeedbackSystem.generate_grammar_feedback(states)
    if not ok:
        return Result(ok=False, error=f"RoboLab solver: {message}", provenance={"seed": seed})

    findings = [
        _finding("BBOX_NOT_UPRIGHT", "warning", item["note"], object=item["name"])
        for item in resize_audit if item["reason"] == "bbox_from_rotated_config"
    ]
    if grammar:
        findings.append(_finding("ROBOLAB_GRAMMAR", "warning", grammar))

    # Solver frame -> authored frame, heights per AUTHORING_RULES.md.
    layout = {}
    for name, state in states.items():
        cfg = configs[name]
        if state.yaw and round(state.yaw) % 180 == 90:
            cfg["orientation"] = YAW_0 if _yaw90(cfg) else list(YAW_90)
            dims[name] = (dims[name][1], dims[name][0], dims[name][2])
        x_rel, y_rel = frame.from_solver(float(state.x), float(state.y))
        cfg["_solver_xy"] = (float(state.x), float(state.y))
        cfg["relative_bbox_position"] = [round(x_rel, 5), round(y_rel, 5), None]
    for name, cfg in configs.items():
        if name not in subjects:
            cfg["relative_bbox_position"][2] = math.ceil((cfg["bounding_box"][2] / 2 + SUPPORT_CLEARANCE) * 1e7) / 1e7
    for name, rel in subjects.items():
        cfg, holder = configs[name], configs[rel.get("container") or rel.get("support")]
        holder_top = holder["relative_bbox_position"][2] + holder["bounding_box"][2] / 2
        if rel["type"] == "on":
            z = holder_top + RELATION_CLEARANCE + cfg["bounding_box"][2] / 2
        else:
            # RoboLab drops contents from just above the rim; keep its offset relative to REALM's base.
            z = float(states[name].z) + SUPPORT_CLEARANCE
        cfg["relative_bbox_position"][2] = round(z, 7)

    findings += _check(spec, frame, configs, dims, subjects, relations)

    for name, cfg in configs.items():
        layout[name] = {
            "role": by_name[name]["role"],
            "category": cfg["category"],
            "model": cfg["model"],
            "relative_bbox_position": cfg["relative_bbox_position"],
            "robot_frame_forward_left": [round(v, 4) for v in cfg.pop("_solver_xy")],
            "bounding_box": cfg["bounding_box"],
            "yaw_deg": 90 if _yaw90(cfg) else 0,
        }

    instruction = spec["instruction"]
    verb = spec.get("verb", spec["task_type"])
    obj_token = by_name[main].get("noun", main.replace("_", " "))
    target_token = by_name[target].get("noun", target.replace("_", " ")) if target else ""
    if obj_token not in instruction:
        findings.append(_finding("INSTRUCTION_OBJ_TOKEN", "error",
                                 f"main object noun {obj_token!r} must occur in the instruction (SB-NOUN/VSB-NOBJ)"))
    if target_token and target_token not in instruction:
        findings.append(_finding("INSTRUCTION_TARGET_TOKEN", "error",
                                 f"target noun {target_token!r} must occur in the instruction"))
    if verb not in instruction:
        findings.append(_finding("INSTRUCTION_VERB_TOKEN", "warning",
                                 f"verb {verb!r} does not occur in the instruction; SB-VRB will not substitute it"))

    distractors = [configs[name] for name, obj in by_name.items() if obj["role"] == "distractor"]
    document = {
        "task": {"type": "DummyTask", "termination_config": {}, "reward_config": {}},
        "task_type": spec["task_type"],
        "instruction": instruction,
        "instruction_obj_to_replace": obj_token,
        "instruction_target_to_replace": target_token,
        "instruction_verb_to_replace": verb,
        "supported_scenes": {frame.region["scene"]: [frame.region["support"]]},
        "camera_extrinsics": None,
        "main_objects": [configs[main]],
        "target_objects": [configs[target]] if target else [],
        "distractors": distractors,
        "immutables": [configs[source]] if source else [],
    }
    spec_hash = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    provenance = {
        "generator": "robolab_poc",
        "robolab_commit": ROBOLAB_COMMIT,
        "spec_sha1": spec_hash,
        "spec": spec,
        "seed": seed,
        "collision_margin": margin,
        "catalog_fingerprint": catalog.get("fingerprint"),
        "region": frame.id,
        "robot_yaw_deg": frame.yaw,
        "mirrored_x": frame.mirrored,
        "solver_bounds_forward_left": [round(v, 4) for v in bounds],
        "resized_assets": resize_audit,
        "receiver_capacity": capacity,
    }
    blocking = any(f["severity"] == "error" for f in findings)
    return Result(ok=not blocking, document=document, findings=findings, layout=layout, provenance=provenance)


def _check(spec, frame, configs, dims, subjects, relations) -> list[dict]:

    findings = []
    region = frame.region
    base = [name for name in configs if name not in subjects]
    for name in base:
        cfg = configs[name]
        x, y, _ = cfg["relative_bbox_position"]
        fx, fy = _world_footprint(cfg)
        if not bbox_fits_support(x, y, [fx, fy, 0.0], region["width"], region["depth"], frame.elliptical):
            findings.append(_finding("OUT_OF_SUPPORT", "error",
                                     f"{name} is not 25 mm inside the {region['support']} support", object=name))
    for i, first in enumerate(base):
        for second in base[i + 1:]:
            (x1, y1, _), (x2, y2, _) = configs[first]["relative_bbox_position"], configs[second]["relative_bbox_position"]
            (w1, d1), (w2, d2) = _world_footprint(configs[first]), _world_footprint(configs[second])
            if abs(x1 - x2) < (w1 + w2) / 2 + OVERLAP_MARGIN and abs(y1 - y2) < (d1 + d2) / 2 + OVERLAP_MARGIN:
                findings.append(_finding("OVERLAP", "error", f"{first} and {second} footprints overlap",
                                         objects=[first, second]))
    for rel in relations:
        if rel["type"] in SPATIAL:
            obj, ref = configs[rel["object"]], configs[rel["reference"]]
            # Recompute from the authored positions so the check is independent of the solver.
            ao, bo = frame.to_solver(*obj["relative_bbox_position"][:2])
            ar, br = frame.to_solver(*ref["relative_bbox_position"][:2])
            da, db = ao - ar, bo - br
            along, across = {"left-of": (db, da), "right-of": (-db, da),
                             "behind": (da, db), "in-front-of": (-da, db)}[rel["type"]]
            if not (along > 0 and along >= abs(across)):
                findings.append(_finding(
                    "RELATION_BROKEN", "error",
                    f"{rel['object']} is not {rel['type']} {rel['reference']} after collision repair "
                    f"(along {along:+.3f} m, across {across:+.3f} m)", relation=rel))
        else:
            holder_name = rel.get("container") or rel.get("support")
            holder, subject = dims[holder_name], dims[rel["object"]]
            sa, sb = configs[rel["object"]]["_solver_xy"]
            ha, hb = configs[holder_name]["_solver_xy"]
            if rel["type"] == "in":
                rx, ry = holder[0] * CONTAINER_MOUTH, holder[1] * CONTAINER_MOUTH
                fits = ((abs(sa - ha) + subject[0] / 2) / rx) ** 2 + ((abs(sb - hb) + subject[1] / 2) / ry) ** 2 <= 1.0
                if not fits:
                    findings.append(_finding("CONTAINER_MOUTH", "error",
                                             f"{rel['object']} does not fit {holder_name}'s opening; it would rest on the rim",
                                             relation=rel))
            else:
                ratio = min(1.0, holder[0] / subject[0]) * min(1.0, holder[1] / subject[1])
                if ratio < STACK_SUPPORT_RATIO:
                    findings.append(_finding("SUPPORT_TOO_SMALL", "warning",
                                             f"{holder_name} supports only {ratio:.0%} of {rel['object']}'s footprint",
                                             relation=rel))
    return findings
