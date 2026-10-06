"""AUTHORING_RULES.md in executable form: the geometry/schema gate for task configs.

Every stage of the task-authoring toolchain calls this one module, so a rule exists once:

    generator  -> validate() before writing YAML      (reject-and-retry, no bad config on disk)
    agent loop -> validate() on each proposed draft   (tool result the model must satisfy)
    vision     -> validate() after applying a patch   (a correction may not break another rule)
    CI         -> `python -m tooling.task_authoring.validation <family-dir>`

Findings are structured, not printed prose: `code` is stable and machine-readable, and `fix`
carries the concrete replacement value when the rule implies one, which is what lets the
correction loop apply a patch without asking a model to re-derive arithmetic.

What this CANNOT prove, and what therefore still requires an OmniGibson run: mesh-level
clearance (outer bboxes are a proxy for container interiors), settling stability, contact
forces, reachability, and anything texture- or material-dependent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = REPO_ROOT / "realm" / "config" / "scenes" / "scenes.yaml"
DEFAULT_PROGRESSIONS = REPO_ROOT / "realm" / "config" / "tasks" / "task_progressions.yaml"

SUPPORT_CLEARANCE = 0.05
SUPPORT_EDGE_CLEARANCE = 0.025
RELATION_CLEARANCE = 0.01
OVERLAP_MARGIN = 0.012
FLOAT_TOLERANCE = 0.02
CAPACITY_MARGIN = {"put": 1.15, "stack": 0.65}
ELLIPTICAL_SUPPORTS = {"Coffee_Table", "Circular_Table"}
# (scene, support) pairs whose configured rectangle rendered review showed is not a reliable
# support surface. Mirrors UNSAFE_SCENE_REGIONS in generate_realm_droid100.py.
UNSAFE_SCENE_REGIONS = {
    ("Pomaria_0_int", "Coffee_Table"),
    ("Pomaria_1_int", "Drawers_Near_Table"),
    ("office_cubicles_left", "Circular_Table"),
}

ROLES = ("main_objects", "target_objects", "distractors", "immutables")
# Keys OmniGibson's object constructors accept, per `type`. Everything else is absorbed by
# `**kwargs` on the DatasetObject -> USDObject chain and silently does nothing, so an invented
# `mass:`/`friction:` key is a no-op rather than an error at load. That silence is the reason for
# UNKNOWN_KEY: friction belongs in link_physics_materials, mass has no config-level setter at all
# and must be applied post-load (see scene_setup.py).
COMMON_OBJECT_KEYS = {
    "type", "name", "scale", "orientation", "relative_bbox_position", "position", "fixed_base",
    "kinematic_only", "visual_only", "self_collisions", "visible", "abilities", "prim_type",
    "in_rooms", "load_config", "link_physics_materials", "bounding_box", "category",
}
OBJECT_KEYS_BY_TYPE = {
    "DatasetObject": {"model", "dataset_name", "fit_avg_dim_volume", "expected_file_hash"},
    "USDObject": {"usd_path", "encrypted"},
    "PrimitiveObject": {"primitive_type", "rgba", "radius", "height", "size"},
    "LightObject": {"light_type", "intensity", "radius", "length", "texture_file_path"},
}
# Roles whose members are manipulated by the policy; a fixed base on any of them makes the task
# unachievable, and OmniGibson reports nothing.
MOVABLE_ROLES = ("main_objects", "target_objects", "distractors")
# Task types scored on a joint, not on moving a free body: their main object is a fixture and
# SHOULD be fixed_base. Everything else must be liftable.
ARTICULATED_TASK_TYPES = {"open_drawer", "close_drawer", "push", "turn_faucet"}
RELATION_PREDICATES = {"inside", "on_top_of"}
ELONGATION_RATIO = 2.0

# Two readings of scenes.yaml coexist in the repo and the difference is not cosmetic. The
# generator treats a spawn rectangle as THE support footprint and keeps every bbox 25 mm inside
# it; REALM_DROID10 was authored by hand against the real table, where that rectangle is only the
# region `place_within` samples from, so its objects legitimately sit outside it and legitimately
# sit at relative z 0 (env cfg `initial_pos_z_offset` lifts them at load). Under `authored` the
# support-geometry codes are reported as warnings so the rules that hold for BOTH readings --
# overlap, undeclared stacking, capacity, roles, unknown keys -- stay errors.
PROFILES = ("generated", "authored")
SUPPORT_GEOMETRY_CODES = {
    "SUPPORT_CONTAINMENT", "SUPPORT_PENETRATION", "SUPPORT_CLEARANCE", "FLOATING_OBJECT",
    # An exclusion authored against BATCH generation; a hand-authored task may use the region
    # deliberately, having seen it render.
    "SCENE_REGION_UNSAFE",
}


@dataclass
class Finding:
    """One rule violation. `fix` is the applicable correction when the rule determines one."""

    code: str
    severity: str
    message: str
    obj: str | None = None
    path: str | None = None
    fix: dict[str, object] | None = None

    def line(self) -> str:
        where = f" [{self.obj}]" if self.obj else ""
        return f"{self.severity.upper():<7} {self.code:<24}{where} {self.message}"


@dataclass
class Report:
    task: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "ok": self.ok,
            "error_count": len(self.errors),
            "findings": [asdict(item) for item in self.findings],
        }


def load_regions(scenes: Path = DEFAULT_SCENES) -> dict[tuple[str, str], dict[str, float]]:
    """Index scenes.yaml spawn rectangles by (scene, support)."""
    document = yaml.safe_load(scenes.read_text(encoding="utf-8")) or {}
    regions = {}
    for scene, supports in document.items():
        if not isinstance(supports, dict):
            continue
        for support, config in supports.items():
            if not isinstance(config, dict) or not all(
                key in config for key in ("x_min", "x_max", "y_min", "y_max")
            ):
                continue
            regions[(str(scene), str(support))] = {
                "width": float(config["x_max"]) - float(config["x_min"]),
                "depth": float(config["y_max"]) - float(config["y_min"]),
                "z": float(config.get("z", 0.0)),
            }
    return regions


def load_task_types(progressions: Path = DEFAULT_PROGRESSIONS) -> set[str]:
    """The closed task_type namespace is whatever task_progressions.yaml can score."""
    return set(yaml.safe_load(progressions.read_text(encoding="utf-8")) or {})


def objects_of(document: dict, role: str) -> list[dict]:
    values = document.get(role) or []
    return [item for item in values if isinstance(item, dict)]


def all_objects(document: dict) -> list[tuple[str, int, dict]]:
    return [
        (role, index, config)
        for role in ROLES
        for index, config in enumerate(objects_of(document, role))
    ]


def bbox_of(config: dict) -> list[float] | None:
    values = config.get("bounding_box") or config.get("scale")
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def position_of(config: dict) -> list[float] | None:
    values = config.get("relative_bbox_position")
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def yaw_only(orientation: list[float], tolerance: float = 1e-3) -> bool:
    """XYZW quaternion with no roll or pitch component."""
    return abs(float(orientation[0])) <= tolerance and abs(float(orientation[1])) <= tolerance


def oriented_footprint(bbox: list[float], orientation: list[float] | None) -> tuple[float, float]:
    """XY extent after a yaw of 0 or 90 degrees; any other yaw uses the circumscribed extent."""
    if orientation is None:
        return bbox[0], bbox[1]
    z, w = float(orientation[2]), float(orientation[3])
    yaw = 2 * math.atan2(z, w)
    if abs(math.sin(yaw)) > 0.99:
        return bbox[1], bbox[0]
    if abs(math.sin(yaw)) < 0.01:
        return bbox[0], bbox[1]
    span = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (
        bbox[0] * span[0] + bbox[1] * span[1],
        bbox[0] * span[1] + bbox[1] * span[0],
    )


def declared_relations(document: dict) -> dict[str, tuple[str, str]]:
    """Map subject name -> (predicate, object name) from the optional `initial_state` block."""
    relations = {}
    for entry in document.get("initial_state") or []:
        if not isinstance(entry, dict):
            continue
        predicate, subject, target = entry.get("predicate"), entry.get("subject"), entry.get("object")
        if predicate in RELATION_PREDICATES and subject and target:
            relations[str(subject)] = (str(predicate), str(target))
    return relations


def check_structure(document: dict, task_types: set[str]) -> list[Finding]:
    findings = []
    task_type = document.get("task_type")
    if task_type not in task_types:
        findings.append(Finding(
            "TASK_TYPE_UNKNOWN", "error",
            f"task_type {task_type!r} has no rubric in task_progressions.yaml "
            f"(known: {', '.join(sorted(task_types))})",
        ))
    for key in ("instruction", "supported_scenes", "main_objects"):
        if not document.get(key):
            findings.append(Finding("SCHEMA_MISSING_KEY", "error", f"{key} is missing or empty"))
    mains = objects_of(document, "main_objects")
    if len(mains) != 1:
        findings.append(Finding(
            "ROLE_CARDINALITY", "error",
            f"exactly one main object is required, found {len(mains)}; REALM scores one "
            "manipulated object per task",
        ))
    targets = objects_of(document, "target_objects")
    relations = declared_relations(document)
    if task_type in CAPACITY_MARGIN and not targets and not relations:
        findings.append(Finding(
            "ROLE_CARDINALITY", "error",
            f"task_type {task_type!r} needs a target object (the receiver/support)",
        ))
    if len(targets) > 1:
        findings.append(Finding(
            "ROLE_CARDINALITY", "error", f"at most one target object, found {len(targets)}",
        ))
    seen: dict[str, str] = {}
    for role, _, config in all_objects(document):
        name = str(config.get("name", ""))
        if not name:
            findings.append(Finding("SCHEMA_MISSING_KEY", "error", f"object in {role} has no name"))
            continue
        if name in seen:
            findings.append(Finding(
                "NAME_DUPLICATE", "error",
                f"name {name!r} used in both {seen[name]} and {role}; OmniGibson names are "
                "unique per scene",
                obj=name,
            ))
        seen[name] = role
    return findings


def check_object_fields(document: dict) -> list[Finding]:
    findings = []
    for role, index, config in all_objects(document):
        name = str(config.get("name", f"{role}[{index}]"))
        path = f"{role}[{index}]"
        object_type = str(config.get("type", "DatasetObject"))
        accepted = COMMON_OBJECT_KEYS | OBJECT_KEYS_BY_TYPE.get(object_type, set())
        if object_type not in OBJECT_KEYS_BY_TYPE:
            findings.append(Finding(
                "OBJECT_TYPE_UNKNOWN", "error",
                f"type {object_type!r} is not a known OmniGibson object class "
                f"({', '.join(sorted(OBJECT_KEYS_BY_TYPE))})",
                obj=name, path=f"{path}.type",
            ))
        for key in config:
            if key not in accepted:
                findings.append(Finding(
                    "UNKNOWN_KEY", "error",
                    f"{key!r} is not a config key of {object_type}; it is absorbed by **kwargs "
                    "and silently ignored at load. Friction belongs in link_physics_materials; "
                    "mass must be applied post-load in scene_setup.py",
                    obj=name, path=f"{path}.{key}",
                ))
        bbox = bbox_of(config)
        if bbox is None:
            findings.append(Finding(
                "BBOX_INVALID", "error", "bounding_box/scale must be three numbers",
                obj=name, path=f"{path}.bounding_box",
            ))
        elif min(bbox) <= 0:
            findings.append(Finding(
                "BBOX_INVALID", "error", f"non-positive extent {bbox}", obj=name,
                path=f"{path}.bounding_box",
            ))
        orientation = config.get("orientation")
        if orientation is not None:
            if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
                findings.append(Finding(
                    "ORIENTATION_INVALID", "error",
                    "orientation must be an XYZW quaternion of four numbers",
                    obj=name, path=f"{path}.orientation",
                ))
                continue
            norm = math.sqrt(sum(float(value) ** 2 for value in orientation))
            if abs(norm - 1.0) > 1e-3:
                findings.append(Finding(
                    "QUAT_NOT_NORMALIZED", "error", f"quaternion norm {norm:.5f} is not 1",
                    obj=name, path=f"{path}.orientation",
                    fix={"path": f"{path}.orientation",
                         "value": [round(float(value) / norm, 7) for value in orientation]},
                ))
        articulated_main = (
            role == "main_objects" and str(document.get("task_type")) in ARTICULATED_TASK_TYPES
        )
        if role in MOVABLE_ROLES and config.get("fixed_base") and not articulated_main:
            findings.append(Finding(
                "FIXED_BASE_ON_MOVABLE", "error",
                f"fixed_base on a {role} member pins it in place; the task becomes unachievable "
                "and nothing in OmniGibson reports it",
                obj=name, path=f"{path}.fixed_base", fix={"path": f"{path}.fixed_base", "value": False},
            ))
    return findings


def check_upright(document: dict) -> list[Finding]:
    """Distractors are always upright; a manipulated object tips only for a declared insertion."""
    findings = []
    relations = declared_relations(document)
    for role, index, config in all_objects(document):
        orientation = config.get("orientation")
        if not isinstance(orientation, (list, tuple)) or len(orientation) != 4 or yaw_only(orientation):
            continue
        name = str(config.get("name", f"{role}[{index}]"))
        path = f"{role}[{index}].orientation"
        if role == "distractors":
            findings.append(Finding(
                "UPRIGHT_DISTRACTOR", "error",
                "distractor has roll/pitch; clutter must stand as it would on a real table",
                obj=name, path=path, fix={"path": path, "value": [0.0, 0.0, 0.0, 1.0]},
            ))
            continue
        bbox = bbox_of(config)
        predicate = relations.get(name, (None, None))[0]
        elongated = bbox is not None and max(bbox) > ELONGATION_RATIO * sorted(bbox)[1]
        if predicate == "inside" and elongated:
            continue
        findings.append(Finding(
            # Only clutter is held to this as an error. A manipulated asset whose USD is not
            # authored Z-up needs a roll/pitch just to stand up, and that is reviewed per task.
            "NON_UPRIGHT_ORIENTATION", "warning",
            "roll/pitch outside a declared `inside` relation is normally an authoring error; "
            "confirm the asset's native frame requires it",
            obj=name, path=path, fix={"path": path, "value": [0.0, 0.0, 0.0, 1.0]},
        ))
    return findings


def check_support_geometry(document: dict, region: dict[str, float] | None,
                           elliptical: bool) -> list[Finding]:
    findings = []
    relations = declared_relations(document)
    for role, index, config in all_objects(document):
        name = str(config.get("name", f"{role}[{index}]"))
        path = f"{role}[{index}].relative_bbox_position"
        bbox, position = bbox_of(config), position_of(config)
        if position is None:
            findings.append(Finding(
                "SCHEMA_MISSING_KEY", "error",
                "relative_bbox_position is required (scene-frame offset from the spawn-region "
                "corner; world positions only look right in scene 0 of a vector build)",
                obj=name, path=path,
            ))
            continue
        if bbox is None:
            continue
        expected_z = bbox[2] / 2 + SUPPORT_CLEARANCE
        stacked = name in relations
        if not stacked:
            if position[2] < bbox[2] / 2 - 1e-6:
                findings.append(Finding(
                    "SUPPORT_PENETRATION", "error",
                    f"z={position[2]:.4f} puts the lower bbox face {bbox[2] / 2 - position[2]:.4f} m "
                    "below the support; the first physics step resolves that as a large contact impulse",
                    obj=name, path=path,
                    fix={"path": path, "value": [position[0], position[1], round(expected_z, 7)]},
                ))
            elif position[2] > expected_z + FLOAT_TOLERANCE:
                findings.append(Finding(
                    "FLOATING_OBJECT", "error",
                    f"z={position[2]:.4f} leaves a {position[2] - bbox[2] / 2:.4f} m gap above the "
                    "support, which nothing in the scene holds up",
                    obj=name, path=path,
                    fix={"path": path, "value": [position[0], position[1], round(expected_z, 7)]},
                ))
            elif abs(position[2] - expected_z) > 1e-6:
                findings.append(Finding(
                    "SUPPORT_CLEARANCE", "warning",
                    f"z={position[2]:.4f}; the authoring convention is bbox_height/2 + "
                    f"{SUPPORT_CLEARANCE} = {expected_z:.4f}",
                    obj=name, path=path,
                    fix={"path": path, "value": [position[0], position[1], round(expected_z, 7)]},
                ))
        if region is None:
            continue
        width, depth = region["width"], region["depth"]
        extent_x, extent_y = oriented_footprint(bbox, config.get("orientation"))
        if not _fits_support(position[0], position[1], extent_x, extent_y, width, depth, elliptical):
            findings.append(Finding(
                "SUPPORT_CONTAINMENT", "error",
                f"authored footprint {extent_x:.3f}x{extent_y:.3f} at ({position[0]:.3f}, "
                f"{position[1]:.3f}) is not {SUPPORT_EDGE_CLEARANCE} m inside the "
                f"{width:.3f}x{depth:.3f} m "
                f"{'ellipse' if elliptical else 'rectangle'}; it overhangs the support edge",
                obj=name, path=path,
            ))
    return findings


def _fits_support(x: float, y: float, extent_x: float, extent_y: float,
                  width: float, depth: float, elliptical: bool) -> bool:
    half_x, half_y = extent_x / 2, extent_y / 2
    if elliptical:
        radius_x = width / 2 - SUPPORT_EDGE_CLEARANCE
        radius_y = depth / 2 - SUPPORT_EDGE_CLEARANCE
        if radius_x <= half_x or radius_y <= half_y:
            return False
        return (
            ((abs(x - width / 2) + half_x) / radius_x) ** 2
            + ((abs(y - depth / 2) + half_y) / radius_y) ** 2
        ) <= 1.0
    return (
        half_x + SUPPORT_EDGE_CLEARANCE <= x <= width - half_x - SUPPORT_EDGE_CLEARANCE
        and half_y + SUPPORT_EDGE_CLEARANCE <= y <= depth - half_y - SUPPORT_EDGE_CLEARANCE
    )


def check_overlap(document: dict) -> list[Finding]:
    """Pairwise XY overlap, excluding pairs a declared relation genuinely pins.

    A declared `initial_state` predicate alone is NOT an exemption: no code under `realm/` reads
    that key, so declaring one silences this rule without changing what the simulator does. The
    relation is honoured only when the object being rested on or contained in is itself protected
    from re-placement -- an `immutable`, which perturbations/v_sc.py pins by name. Anything else
    stays an error.

    The main object overlapping its TARGET in a put/stack task is reported separately as
    GOAL_SATISFIED_AT_START: that arrangement is the success state, not a layout to pin.
    """
    findings = []
    relations = declared_relations(document)
    protected_names = {
        str(config.get("name")) for config in objects_of(document, "immutables")
    }
    goal_pair = None
    if str(document.get("task_type")) in {"put", "stack"}:
        mains = objects_of(document, "main_objects")
        targets = objects_of(document, "target_objects")
        if mains and targets:
            goal_pair = frozenset((str(mains[0].get("name")), str(targets[0].get("name"))))
    exempt = {
        frozenset((subject, target))
        for subject, (_, target) in relations.items()
        if target in protected_names
    }
    placed = []
    for role, index, config in all_objects(document):
        bbox, position = bbox_of(config), position_of(config)
        if bbox is None or position is None:
            continue
        placed.append((str(config.get("name", f"{role}[{index}]")), role, bbox, position,
                       config.get("orientation")))
    for first in range(len(placed)):
        for second in range(first + 1, len(placed)):
            name_a, _, bbox_a, pos_a, ori_a = placed[first]
            name_b, _, bbox_b, pos_b, ori_b = placed[second]
            if frozenset((name_a, name_b)) in exempt:
                continue
            extent_a = oriented_footprint(bbox_a, ori_a)
            extent_b = oriented_footprint(bbox_b, ori_b)
            gap_x = abs(pos_a[0] - pos_b[0]) - (extent_a[0] + extent_b[0]) / 2
            gap_y = abs(pos_a[1] - pos_b[1]) - (extent_a[1] + extent_b[1]) / 2
            if gap_x >= OVERLAP_MARGIN or gap_y >= OVERLAP_MARGIN:
                continue
            if goal_pair is not None and frozenset((name_a, name_b)) == goal_pair and gap_x < 0 and gap_y < 0:
                findings.append(Finding(
                    "GOAL_SATISFIED_AT_START", "error",
                    f"the main object and its target ({name_a!r}, {name_b!r}) share an XY "
                    f"footprint at reset, i.e. the {document.get('task_type')} task starts in or "
                    "near its success state. Author them apart; the robot must achieve the relation",
                    obj=name_a,
                ))
                continue
            if gap_x >= 0 or gap_y >= 0:
                # Separated in XY, but by less than the margin the placer leaves for settling
                # drift and mesh overhang beyond the outer bbox.
                findings.append(Finding(
                    "TIGHT_CLEARANCE", "warning",
                    f"{name_a!r} and {name_b!r} clear each other by only "
                    f"({gap_x:.4f}, {gap_y:.4f}) m, inside the {OVERLAP_MARGIN} m placement margin",
                    obj=name_a,
                ))
                continue
            # An immutable is authored scenery -- a support, a container, a table -- so an
            # overlap with one is usually an intended relation that simply was not declared.
            roles_in_pair = {placed[first][1], placed[second][1]}
            severity = "warning" if "immutables" in roles_in_pair else "error"
            vertical_gap = abs(pos_a[2] - pos_b[2]) - (bbox_a[2] + bbox_b[2]) / 2
            # Promoting the LOWER object is what survives perturbation: `immutables` ride in
            # env.distractors but are pinned through placement's main_object_names by
            # perturbations/v_sc.py, which is the only perturbation that re-places a distractor.
            # Declaring an initial_state predicate instead would silence this rule while changing
            # nothing at runtime -- no code under realm/ reads that key.
            lower, upper = (name_a, name_b) if pos_a[2] <= pos_b[2] else (name_b, name_a)
            if vertical_gap >= 0:
                findings.append(Finding(
                    "UNDECLARED_STACK", severity,
                    f"{name_a!r} and {name_b!r} share an XY footprint and are vertically "
                    "separated, i.e. one rests on the other, but the arrangement is not "
                    "author-declared; if it is intended, move the supporting object to the "
                    f"`immutables` role (else V-SC re-places it and the arrangement breaks)",
                    obj=name_a,
                    fix={"action": "promote_to_immutable", "object": lower,
                         "reason": "the resting object is separated from the support by V-SC"},
                ))
                continue
            findings.append(Finding(
                "XY_OVERLAP", severity,
                f"{name_a!r} and {name_b!r} interpenetrate: their bboxes overlap in XY by "
                f"({-gap_x:.4f}, {-gap_y:.4f}) m and also overlap in Z. If one is meant to be in "
                f"or on the other, move the container ({lower!r}) to the `immutables` role",
                obj=name_a,
                fix={"action": "promote_to_immutable", "object": lower,
                     "reason": "the containing object must be pinned for the overlap to survive"},
            ))
    return findings


def fits_lengthwise(main_bbox, target_bbox, margin: float) -> bool:
    """An elongated object (pen, marker, utensil) goes into a deep container long-axis-vertical.

    Comparing its lying-down footprint with the container's opening would demand a 16 cm-wide mug
    for a pen. What has to fit is the cross-section, and the container must be deep enough to hold
    the object upright (at least half its length, as a mug holds a standing pen).
    """
    dims = sorted(float(value) for value in main_bbox)
    longest, cross = dims[2], dims[:2]
    if longest < 2 * dims[1] or float(target_bbox[2]) < 0.5 * longest:
        return False
    opening = sorted(float(value) for value in target_bbox[:2])
    return opening[0] >= cross[1] * margin and opening[1] >= cross[1] * margin


def check_capacity(document: dict) -> list[Finding]:
    """Outer-bbox proxy for 'will the main object fit in/on the receiver'."""
    mains = objects_of(document, "main_objects")
    targets = objects_of(document, "target_objects")
    task_type = str(document.get("task_type", ""))
    margin = CAPACITY_MARGIN.get(task_type)
    if margin is None or not mains or not targets:
        return []
    main, target = mains[0], targets[0]
    main_bbox, target_bbox = bbox_of(main), bbox_of(target)
    if main_bbox is None or target_bbox is None:
        return []
    extent = oriented_footprint(main_bbox, main.get("orientation"))
    fits = (
        target_bbox[0] >= extent[0] * margin and target_bbox[1] >= extent[1] * margin
    )
    if fits or (task_type == "put" and fits_lengthwise(main_bbox, target_bbox, margin)):
        return []
    scale = min(
        target_bbox[0] / (extent[0] * margin), target_bbox[1] / (extent[1] * margin),
    )
    swapped = (
        target_bbox[0] >= extent[1] * margin and target_bbox[1] >= extent[0] * margin
    )
    hint = (
        {"path": "main_objects[0].orientation", "value": [0.0, 0.0, 0.7071068, 0.7071068]}
        if swapped else
        {"path": "main_objects[0].bounding_box",
         "value": [round(value * scale, 7) for value in main_bbox]}
    )
    return [Finding(
        "RECEIVER_CAPACITY", "error",
        f"main footprint {extent[0]:.3f}x{extent[1]:.3f} does not fit target "
        f"{target_bbox[0]:.3f}x{target_bbox[1]:.3f} at the {margin} {task_type} margin; "
        + ("yaw the main object 90 degrees" if swapped
           else f"uniform scale {scale:.3f} would fit (never squeeze one axis)"),
        obj=str(main.get("name")), fix=hint,
    )]


def check_relations(document: dict) -> list[Finding]:
    """Declared initial_state predicates must be geometrically true as authored."""
    findings = []
    by_name = {str(config.get("name")): config for _, _, config in all_objects(document)}
    for subject, (predicate, target) in declared_relations(document).items():
        if subject not in by_name or target not in by_name:
            findings.append(Finding(
                "RELATION_UNGROUNDED", "error",
                f"initial_state names {subject!r} {predicate} {target!r}, but "
                f"{'subject' if subject not in by_name else 'object'} is not an authored object",
                obj=subject,
            ))
            continue
        main, source = by_name[subject], by_name[target]
        main_bbox, source_bbox = bbox_of(main), bbox_of(source)
        main_pos, source_pos = position_of(main), position_of(source)
        if None in (main_bbox, source_bbox, main_pos, source_pos):
            continue
        path = None
        for role, index, config in all_objects(document):
            if config is main:
                path = f"{role}[{index}].relative_bbox_position"
        centred = (
            abs(main_pos[0] - source_pos[0]) <= source_bbox[0] / 2
            and abs(main_pos[1] - source_pos[1]) <= source_bbox[1] / 2
        )
        if not centred:
            findings.append(Finding(
                "RELATION_GEOMETRY", "error",
                f"{subject!r} is declared {predicate} {target!r} but its XY centre is outside "
                f"{target!r}'s footprint",
                obj=subject, path=path,
                fix={"path": path,
                     "value": [source_pos[0], source_pos[1], round(main_pos[2], 7)]},
            ))
        source_top = source_pos[2] + source_bbox[2] / 2
        if predicate == "on_top_of":
            expected = source_top + main_bbox[2] / 2 + RELATION_CLEARANCE
            if abs(main_pos[2] - expected) > FLOAT_TOLERANCE:
                findings.append(Finding(
                    "RELATION_GEOMETRY", "error",
                    f"{subject!r} on {target!r} should sit at z={expected:.4f} "
                    f"(source top + half height + {RELATION_CLEARANCE} m), authored "
                    f"{main_pos[2]:.4f}",
                    obj=subject, path=path,
                    fix={"path": path, "value": [source_pos[0], source_pos[1], round(expected, 7)]},
                ))
        elif predicate == "inside":
            vertical = max(main_bbox)
            if main_pos[2] - vertical / 2 > source_top:
                findings.append(Finding(
                    "RELATION_GEOMETRY", "error",
                    f"{subject!r} is declared inside {target!r} but its lowest point "
                    f"{main_pos[2] - vertical / 2:.4f} is above the source rim {source_top:.4f}",
                    obj=subject, path=path,
                ))
            if main_bbox[2] > source_bbox[2] * 3 and yaw_only(main.get("orientation") or [0, 0, 0, 1]):
                findings.append(Finding(
                    "CONTAINMENT_ORIENTATION", "warning",
                    f"{subject!r} is much taller than {target!r}; an elongated object is normally "
                    "inserted long-axis-vertical",
                    obj=subject,
                ))
    return findings


def check_target_state(document: dict) -> list[Finding]:
    """`target_state` documents what the author believes success means; the rubric is the truth.

    REALM scores from task_progressions.yaml via TaskProgressionMixin, never from the config, so a
    target_state naming a different rubric than the task_type means the author expected a stage
    sequence the task will not run.
    """
    target_state = document.get("target_state")
    if not isinstance(target_state, dict):
        return []
    rubric = target_state.get("rubric")
    task_type = document.get("task_type")
    if rubric is not None and rubric != task_type:
        return [Finding(
            "TARGET_STATE_RUBRIC", "error",
            f"target_state.rubric {rubric!r} disagrees with task_type {task_type!r}; scoring "
            "follows task_type, so the authored success condition would never be checked",
            path="target_state.rubric", fix={"path": "target_state.rubric", "value": task_type},
        )]
    return []


def check_instruction_closure(document: dict) -> list[Finding]:
    findings = []
    instruction = str(document.get("instruction", "")).lower()
    mains = objects_of(document, "main_objects")
    targets = objects_of(document, "target_objects")
    obj_token = str(document.get("instruction_obj_to_replace", "")).lower()
    target_token = str(document.get("instruction_target_to_replace", "")).lower()
    if not obj_token:
        findings.append(Finding(
            "INSTRUCTION_CLOSURE", "error",
            "instruction_obj_to_replace is empty; the semantic perturbations rewrite the "
            "instruction by substituting this token",
        ))
    elif obj_token not in instruction:
        findings.append(Finding(
            "INSTRUCTION_CLOSURE", "error",
            f"instruction_obj_to_replace {obj_token!r} does not appear in the instruction "
            f"{document.get('instruction')!r}; SB-NOUN/S-LANG substitution will be a no-op",
        ))
    if str(document.get("task_type")) in CAPACITY_MARGIN:
        if not target_token:
            findings.append(Finding(
                "INSTRUCTION_CLOSURE", "error",
                "instruction_target_to_replace is empty for a two-object task type",
            ))
        elif target_token not in instruction:
            findings.append(Finding(
                "INSTRUCTION_CLOSURE", "error",
                f"instruction_target_to_replace {target_token!r} does not appear in the instruction",
            ))
    for token, objects, label in (
        (obj_token, mains, "main object"), (target_token, targets, "target object"),
    ):
        if not token or not objects:
            continue
        config = objects[0]
        haystack = f"{config.get('name', '')} {config.get('category', '')}".lower().replace("_", " ")
        if token not in haystack and haystack.split()[0] not in token:
            findings.append(Finding(
                "INSTRUCTION_GROUNDING", "warning",
                f"instruction token {token!r} does not match the {label} "
                f"({config.get('name')!r}/{config.get('category')!r}); the instruction must name "
                "the asset that is actually in the scene",
                obj=str(config.get("name")),
            ))
    return findings


def check_scene(document: dict, regions: dict[tuple[str, str], dict[str, float]]) -> tuple[
    list[Finding], dict[str, float] | None, bool
]:
    supported = document.get("supported_scenes") or {}
    if not isinstance(supported, dict) or not supported:
        return [Finding("SCENE_MISSING", "error", "supported_scenes is empty")], None, False
    scene = next(iter(supported))
    supports = supported[scene] or []
    if not supports:
        return [Finding("SCENE_MISSING", "error", f"scene {scene!r} lists no support region")], None, False
    support = str(supports[0])
    key = (str(scene), support)
    findings = []
    if key not in regions:
        findings.append(Finding(
            "SCENE_REGION_UNKNOWN", "error",
            f"{scene}/{support} has no x_min/x_max/y_min/y_max rectangle in scenes.yaml, so "
            "relative_bbox_position cannot be resolved",
        ))
        return findings, None, False
    if key in UNSAFE_SCENE_REGIONS:
        findings.append(Finding(
            "SCENE_REGION_UNSAFE", "error",
            f"{scene}/{support} is excluded from generation: rendered review showed its "
            "configured rectangle is not a reliable support surface",
        ))
    return findings, regions[key], support in ELLIPTICAL_SUPPORTS


def validate(document: dict, *, task: str = "<draft>",
             regions: dict[tuple[str, str], dict[str, float]] | None = None,
             task_types: set[str] | None = None, profile: str = "generated") -> Report:
    """Run every host-checkable rule over one loaded task document.

    `profile` selects how scenes.yaml spawn rectangles are read: see PROFILES.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {', '.join(PROFILES)}")
    regions = load_regions() if regions is None else regions
    task_types = load_task_types() if task_types is None else task_types
    scene_findings, region, elliptical = check_scene(document, regions)
    findings = [
        *check_structure(document, task_types),
        *check_object_fields(document),
        *scene_findings,
        *check_upright(document),
        *check_support_geometry(document, region, elliptical),
        *check_overlap(document),
        *check_capacity(document),
        *check_relations(document),
        *check_target_state(document),
        *check_instruction_closure(document),
    ]
    if profile == "authored":
        for finding in findings:
            if finding.code in SUPPORT_GEOMETRY_CODES:
                finding.severity = "warning"
    return Report(task=task, findings=findings)


def validate_file(path: Path, **kwargs) -> Report:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return validate(document, task=path.parent.name, **kwargs)


def validate_family(directory: Path, profile: str = "generated") -> list[Report]:
    regions, task_types = load_regions(), load_task_types()
    return [
        validate_file(path, regions=regions, task_types=task_types, profile=profile)
        for path in sorted(directory.glob("*/default.yaml"))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path,
                        help="a family directory (validates */default.yaml) or one YAML file")
    parser.add_argument("--json", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument("--quiet", action="store_true", help="print only failing tasks")
    parser.add_argument("--profile", default="generated", choices=PROFILES,
                        help="generated: spawn rectangle is the support footprint (strict). "
                             "authored: it is only a sampling region, so support-geometry "
                             "findings are warnings.")
    args = parser.parse_args()
    reports = (
        [validate_file(args.target, profile=args.profile)] if args.target.is_file()
        else validate_family(args.target, profile=args.profile)
    )
    for report in reports:
        if args.quiet and report.ok:
            continue
        print(f"\n{report.task}: {'OK' if report.ok else f'{len(report.errors)} error(s)'}")
        for finding in report.findings:
            print(f"  {finding.line()}")
    failed = [report for report in reports if not report.ok]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} configs pass the host-checkable rules")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([report.as_dict() for report in reports], indent=2) + "\n", encoding="utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
