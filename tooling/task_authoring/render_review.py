"""Render + probe harness: S5 of the agentic pipeline. RUNS IN THE CONTAINER (GPU).

Host-side validation proves arithmetic. It cannot see mesh interiors, contact, or what an object
actually looks like once the sim has settled it. This harness closes that gap: it builds one task
config, settles it, probes the live poses, renders every camera the config enables (both external
views plus the wrist camera, when `multi_view` is on) at the authored pose and again after settling,
and writes a single JSON the vision stage reads.

Three properties this file is built around:

**Compute first, look second** (AGENTIC_PIPELINE.md section 4.1). Every number a reviewer could
want -- per-object settle drift (live pose before vs after settling), support heights -- is
measured here and written into the JSON. A model shown an image AND the numbers can be asked to
report only what the numbers do not already settle, which is the difference between a review loop
that converges and one that hallucinates.

**Exit codes are not trusted.** Isaac exits 0 on unhandled exceptions and segfaults at teardown on
passing runs, so this prints an explicit verdict and writes the verdict into the JSON. The caller
reads the JSON, never `$?`.

**One simulator per process.** `og.shutdown()` is process-terminal -- there is no second scene to
be had in a live interpreter -- so `--family` does not loop over configs in-process. It spawns one
child per config and reads back the JSON each child wrote. That also bounds the blast radius: a
child that segfaults at teardown takes its own config down, not the remaining 99, and a child that
hangs is killed by `--timeout` instead of stalling the sweep.

Run against a running Slurm allocation (`uv` does not exist in the container; the image's conda env
already has REALM on `PYTHONPATH`):

    ./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py \
        --task_cfg_path REALM_DROID100/<task>/default.yaml --out tmp/droid100/review

    ./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py \
        --family REALM_DROID100 --out tmp/droid100/review
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

# `realm.config.shared` imports only numpy, so the placement constant can be bound to the real
# value instead of restated here -- DROP_HEIGHT is a number that moves placements.
from realm.config.shared import DROP_HEIGHT as DROP_HEIGHT_M
# The policy gets control after REALM's warmup, so the settle is judged at the same step count.
from realm.config.shared import WARMUP_STEPS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

#: Per-object thresholds. Exceeded drift is a finding, not automatically a failure: a small
#: settle is expected, a large one means the authored layout was not physically stable.
DRIFT_WARN_M = 0.01
DRIFT_FAIL_M = 0.03
#: A settled lower face this far below the MEASURED support surface means the object is sunk in.
#: Wider than a contact gap because mesh AABBs include tessellation and collision-margin slop.
PENETRATION_TOL_M = 0.015
#: Scene-floor height. Every REALM scene is an `*_int` scene with its floor at z = 0, so a lower face
#: below this is sunk into the floor whatever the task surface is. It is the only penetration check
#: left when no task surface can be measured (the drawer scenes author no objects on the spawn plane).
FLOOR_Z_M = 0.0
#: When placement cannot pack an object it writes z = support + DROP_HEIGHT (0.10 flat) instead of
#: the authored convention z = support + bbox_z/2 + 0.05, and only logs at ERROR level. The two
#: readings collide for bbox_z = 0.10, so the signature also requires the object to have SETTLED
#: (or nearly) onto the support -- an authored z on the drop plane that is still airborne is a
#: floating bug, not a failed pack. `placement.py`'s own ERROR log names the object; this finding
#: is the durable on-disk record of the same event.
DROP_TOL_M = 0.005
#: Renders issued before reading camera buffers (see `render`).
RENDER_FLUSH_PASSES = 30
#: An object REALM authors above its support falls at least SUPPORT_CLEARANCE (5 cm) when physics
#: runs. If no movable object moved even this much during the settle, the sim did not step.
FROZEN_FALL_M = 0.005
#: A lower face starting more than this above the settled surface is a drop the object may not
#: survive upright (it bounces, rolls or tips before the policy starts).
HIGH_DROP_M = 0.06
#: Tilt of an object's up-axis between reset and settle beyond which it has fallen over.
TIPPED_DEG = 35.0
#: An object declared inside a container whose lower face ends less than this above the
#: container's own lower face has gone through the container floor. Render v4's pen and sunglasses
#: ended 0.4 and 0.3 mm BELOW their box's lower face; render v5's banana resting correctly on a
#: 1.8 mm storage-box floor ended 1.8 mm above it, which the earlier 3 mm threshold failed.
THROUGH_FLOOR_M = 0.001
#: An object declared on top of a support whose lower face ends less than this above the support's
#: own base has slid off it onto the surface the support stands on. v5's blocks rested 5 and 11 mm
#: above the base of a plate and a tray.
SLID_OFF_M = 0.003


def authored_names(cfg: dict) -> set:
    """Names the task config itself placed, as opposed to the scene's own fixtures.

    `env.cfg["objects"]` is main + target + distractors + immutables *plus* `scene_definition.yaml`'s
    objects, and `scene_setup.apply_scene_fixes_from_cfg` legitimately removes some of the latter
    (`to_remove`) and pins others (`to_fix`). A removed scene fixture is not an authored object that
    went missing, and reporting it as one would fail the config for nothing.
    """
    return set(authored_roles(cfg))


#: Roles whose objects the sim moves. Immutables are fixtures (a table support, a sink), often
#: pinned and modelled to pass through the floor, so the floor bound does not apply to them.
MOVABLE_ROLES = ("main_objects", "target_objects", "distractors")


def authored_roles(cfg: dict) -> dict:
    """Authored object name -> the config role that placed it."""
    roles = {}
    for role in ("main_objects", "target_objects", "distractors", "immutables"):
        roles.update({str(o["name"]): role for o in (cfg.get(role) or []) if o.get("name")})
    return roles


#: Objects authored more than this far from the spawn plane are not ON the task surface: a
#: floor-standing table support in the drawer scenes, a lamp hung above the table. They must not
#: pull the measured surface down to the floor (both drawer tasks measured z = -0.001).
SUPPORT_BAND_M = 0.30


def on_task_surface(row: dict, support_z: float | None) -> bool:
    authored = row.get("authored_position")
    if support_z is None or not authored:
        return True
    return abs(float(authored[2]) - float(support_z)) <= SUPPORT_BAND_M


def measured_support_z(rows: list[dict], support_z: float | None = None,
                       resting: set | None = None) -> float | None:
    """The support surface as the scene actually has it: the resting height of the authored objects.

    scenes.yaml's `z` is the SPAWN height, not the table top. REALM authors objects at
    spawn z + relative z and lets them fall onto the real surface, which sits about 0.2 m lower in
    some scenes. Using `z` as the table height reported every object in every task as sunk
    18-20 cm into the table. The live surface is read off the settled objects instead: the lower
    quartile of their world-AABB lower faces, so objects stacked on others (higher) and a single
    genuinely sunk object (lower) cannot move it. Needs at least two measured objects.

    Objects the config declares resting on or in another object (`resting`) are not on the surface
    and are left out: a chocolate bar lying in a sink basin, with the sink the only other object,
    made the basin floor "the surface" and flagged the sink as sunk 16 cm below it.
    """
    resting = resting or set()
    lows = sorted(
        float(row["aabb_low_z"]) for row in rows
        if row.get("present") and row.get("authored", True) and row.get("aabb_low_z") is not None
        and on_task_surface(row, support_z) and row.get("name") not in resting
    )
    if len(lows) < 2:
        return None
    return lows[len(lows) // 4] if len(lows) >= 4 else lows[len(lows) // 2]


def declared_relations(task_cfg_path: str) -> list[tuple[str, str, str]]:
    """(predicate, subject, object) for every initial_state entry of the task YAML."""
    import yaml

    path = PROJECT_ROOT / "realm" / "config" / "tasks" / task_cfg_path
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    return [(str(entry.get("predicate")), str(entry.get("subject")), str(entry.get("object")))
            for entry in document.get("initial_state") or []
            if isinstance(entry, dict) and entry.get("subject") and entry.get("object")]


def _up_axis(quaternion) -> np.ndarray:
    x, y, z, w = (float(v) for v in quaternion)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def tilt_deg(start, end) -> float | None:
    """Angle between an object's up-axis at reset and after the settle."""
    if not start or not end:
        return None
    cosine = float(np.clip(np.dot(_up_axis(start), _up_axis(end)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def relation_findings(rows: list[dict], relations: list[tuple[str, str, str]]) -> list[dict]:
    """Start-state failures the drift numbers cannot see.

    TIPPED: a receiver or fixture -- the cup, holder, bowl or box a task is about -- fell over during
    the settle (v4: a knife knocked its cup over, a spatula its holder, a lid its bowl). Only targets
    and immutables are judged: in v5 every main object that "tipped" was fine -- markers and pens
    leaning against a cup rim, a rolling ball, a chip bag standing up in its sink -- and a fallen
    distractor is still clutter. LEFT_CONTAINER: an object declared inside something ended outside
    its footprint or through its floor (v4: a pen sank through a box floor). LEFT_SUPPORT: an object
    declared on top of something ended outside it or down at its base (a lid slid off a bowl).
    """
    findings = []
    by_name = {row.get("name"): row for row in rows if row.get("present")}
    for row in by_name.values():
        if row.get("role") in ("target_objects", "immutables"):
            tilt = tilt_deg(row.get("start_orientation"), row.get("settled_orientation"))
            if tilt is not None and tilt > TIPPED_DEG:
                findings.append({"code": "TIPPED", "object": row["name"],
                                 "reason": f"tilted {tilt:.0f} degrees during the settle (fell over)"})
    for predicate, subject, container in relations:
        inner, outer = by_name.get(subject), by_name.get(container)
        if not inner or not outer:
            continue
        low, high = outer.get("settled_aabb_low"), outer.get("settled_aabb_high")
        centre, inner_low = inner.get("settled_aabb_center"), inner.get("aabb_low_z")
        if not low or not high or not centre or inner_low is None:
            continue
        inside_xy = low[0] <= centre[0] <= high[0] and low[1] <= centre[1] <= high[1]
        if predicate == "inside":
            if not inside_xy:
                findings.append({"code": "LEFT_CONTAINER", "object": subject,
                                 "reason": f"declared inside {container!r} but settled outside its footprint"})
            elif inner_low < low[2] + THROUGH_FLOOR_M:
                findings.append({"code": "LEFT_CONTAINER", "object": subject,
                                 "reason": (f"declared inside {container!r} but its lower face "
                                            f"{inner_low:.4f} is at or below the container's own "
                                            f"lower face {low[2]:.4f}: it went through the floor")})
        elif predicate == "on_top_of":
            # Base, not half-height: a block resting on a rimmed tray (v5: 0.756 on a 0.745-0.771
            # tray) or plate sits well below half the support's height and is still on it.
            if not inside_xy or inner_low < low[2] + SLID_OFF_M:
                findings.append({"code": "LEFT_SUPPORT", "object": subject,
                                 "reason": f"declared on top of {container!r} but slid off it"})
    return findings


def grasp_findings(rows: list[dict], main_names: list[str], overhanging: set | None = None) -> list[dict]:
    """The final doability check: after settling, can the gripper close on the main object at all?

    Judged from the SETTLED world AABB, so it also catches an object that tipped flat or slid down
    into a slot. Same threshold as the host check (validation.GRASP_MIN_HEIGHT_M).
    """
    from tooling.task_authoring.validation import GRASP_MIN_HEIGHT_M

    findings = []
    for row in rows:
        if row.get("name") not in main_names or not row.get("present") or row.get("name") in (overhanging or set()):
            continue
        low, high = row.get("settled_aabb_low"), row.get("settled_aabb_high")
        if not low or not high:
            continue
        height = float(high[2]) - float(low[2])
        if height < GRASP_MIN_HEIGHT_M:
            findings.append({"code": "GRASP_INFEASIBLE", "object": row["name"],
                             "reason": (f"settled only {height * 100:.2f} cm high: the fingertips meet "
                                        f"the surface before the object; the task cannot be done")})
    return findings


def declared_resting(task_cfg_path: str) -> set:
    """Names the task YAML declares on/in another object (initial_state subjects)."""
    import yaml

    path = PROJECT_ROOT / "realm" / "config" / "tasks" / task_cfg_path
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()
    return {str(entry.get("subject")) for entry in document.get("initial_state") or []
            if isinstance(entry, dict) and entry.get("subject")}


def _explicitly_placed(obj_cfg: dict, spawn_bbox) -> bool:
    """True when the live config still holds the authored pose (relative + spawn origin).

    placement.py overwrites the position of an object it re-places, falling back to
    spawn z + DROP_HEIGHT when it cannot pack it. An object AUTHORED at relative z 0.10 (a spoon on
    a plate, a cube on a cube) has the same height, so the z alone cannot tell the two apart.
    """
    relative = obj_cfg.get("relative_bbox_position")
    position = obj_cfg.get("position")
    if spawn_bbox is None or not relative or not position or len(relative) != 3:
        return False
    origin = (float(spawn_bbox[0]), float(spawn_bbox[2]), float(spawn_bbox[4]))
    return all(abs(float(position[i]) - (float(relative[i]) + origin[i])) < 1e-6 for i in range(3))


def _numpy(value):
    value = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    return np.asarray(value, dtype=float)


def live_pose(obj) -> dict:
    """One object's live pose, read two ways: the asset origin and the world-AABB centre.

    The origin is wherever the asset author put it -- often the base, sometimes nowhere near the
    geometry -- so it is not comparable with the authored `position`, which is a BOUNDING-BOX CENTRE.
    Comparing the two made 324 of 354 DROID100 objects look sunk below their own geometry and
    reported a teddy bear sitting in its box as 1.04 m adrift. Drift is therefore only ever taken
    between two readings of the SAME quantity (`settle_drift`).
    """
    pos, ori = obj.get_position_orientation(frame="scene")
    pose = {"origin": [round(float(v), 5) for v in _numpy(pos)],
            "orientation": [round(float(v), 5) for v in _numpy(ori)]}
    try:
        low, high = obj.aabb
        low, high = _numpy(low), _numpy(high)
        pose["aabb_center"] = [round(float(v), 5) for v in (low + high) / 2]
        pose["aabb_low_z"] = round(float(low[2]), 5)
        pose["aabb_low"] = [round(float(v), 5) for v in low]
        pose["aabb_high"] = [round(float(v), 5) for v in high]
    except Exception:
        pass
    return pose


def sync_stage() -> None:
    """Bring the rendered stage (AABBs, camera buffers) up to the physics state.

    Physics runs with `render_on_step(False)` during the settle, so nothing on the render side
    moves until renders are issued; REALM's own warmup flushes 30 of them for the same reason.
    """
    try:
        import omnigibson as og
    except ImportError:
        return
    for _ in range(RENDER_FLUSH_PASSES):
        og.sim.render()


def snapshot(env) -> dict:
    """`live_pose` of every configured object present in the scene, keyed by name."""
    poses = {}
    for obj_cfg in env.cfg.get("objects", []):
        name = obj_cfg.get("name")
        try:
            obj = env.omnigibson_env.scene.object_registry("name", name)
        except Exception:
            obj = None
        if obj is not None:
            poses[name] = live_pose(obj)
    return poses


def settle_drift(before: dict | None, after: dict) -> dict:
    """How far the object moved while settling, like for like: origin before vs origin after.

    The origin comes from the physics engine and is always current. The world AABB is computed from
    the rendered stage, which is only synced on a render: in v2 every object's AABB read exactly the
    same before and after the settle (493 objects, drift 0.0) while the physics poses had fallen.
    Where the origin sits on the asset does not matter when both readings use it. The AABB centre
    is only the fallback when an origin is missing.
    """
    if not before:
        return {}
    key = "origin" if before.get("origin") and after.get("origin") else "aabb_center"
    start, end = before.get(key), after.get(key)
    if not start or not end:
        return {}
    return {
        "drift_reference": key,
        "drift_xy": round(float(np.hypot(end[0] - start[0], end[1] - start[1])), 5),
        "drift_z": round(float(end[2] - start[2]), 5),
    }


def probe(env, before: dict | None = None) -> list[dict]:
    """Read the live scene after settling: one row per configured object.

    `before` is the `snapshot()` taken right after reset, before the settle. Drift is the movement
    between that snapshot and now (`settle_drift`), never live pose minus authored `position`: the
    authored value is a bbox centre, the live pose an asset origin, and the two differ by an amount
    that depends only on how the asset was modelled. The authored position is still recorded, for
    the checks that are about the AUTHORED pose (DROPPED, FLOATING). Nothing here is model-derived.
    """
    rows = []
    roles = authored_roles(env.cfg)
    authored = set(roles)
    before = before or {}
    for obj_cfg in env.cfg.get("objects", []):
        name = obj_cfg.get("name")
        try:
            obj = env.omnigibson_env.scene.object_registry("name", name)
        except Exception:
            obj = None
        if obj is None:
            rows.append({"name": name, "present": False, "authored": name in authored})
            continue
        pose = live_pose(obj)
        authored_pos = obj_cfg.get("position") or []
        bbox = obj_cfg.get("bounding_box") or obj_cfg.get("scale") or []
        row = {
            "name": name,
            "present": True,
            "authored": name in authored,
            "role": roles.get(name),
            "category": getattr(obj, "category", None),
            "settled_position": pose["origin"],
            "settled_orientation": pose["orientation"],
            "settled_aabb_center": pose.get("aabb_center"),
            "start_aabb_center": (before.get(name) or {}).get("aabb_center"),
            "start_aabb_low_z": (before.get(name) or {}).get("aabb_low_z"),
            "start_origin": (before.get(name) or {}).get("origin"),
            "start_orientation": (before.get(name) or {}).get("orientation"),
            "settled_aabb_low": pose.get("aabb_low"),
            "settled_aabb_high": pose.get("aabb_high"),
            "authored_position": [round(float(v), 5) for v in authored_pos]
            if authored_pos else None,
            "bbox": [round(float(v), 5) for v in bbox] if bbox else None,
        }
        row.update(settle_drift(before.get(name), pose))
        if pose.get("aabb_low_z") is not None:
            # World-frame AABB of the live object: its true lowest point, with no assumption about
            # where the asset's origin sits relative to its bounding box.
            row["aabb_low_z"] = pose["aabb_low_z"]
        try:
            row["mass"] = round(float(obj.root_link.mass), 5)
        except Exception:
            pass
        row["explicitly_placed"] = _explicitly_placed(obj_cfg, getattr(env, "spawn_bbox", None))
        rows.append(row)
    return rows


def stability_findings(rows: list[dict], support_z: float, resting: set | None = None) -> list[dict]:
    """Computed findings from the probe, before any image is shown to a model.

    `support_z` is scenes.yaml's spawn `z`: the CONFIG-frame plane that authored and fallback
    positions are expressed against, used for DROPPED and FLOATING. It is not the table top, so
    PENETRATION is judged against `measured_support_z(rows)` instead.

    `resting` names objects the config declares on/in another object (initial_state subjects): a lid
    on a pot or a marker in a mug sits high by design, so FLOATING does not apply to them.

    Only objects the task authored are judged (`authored`, defaulting to True so a hand-built row
    is still judged): scene fixtures the config never placed are reported but not held against the
    config, because `scene_setup` removes and pins some of them deliberately.
    """
    findings = []
    resting = resting or set()
    surface = measured_support_z(rows, support_z, resting)
    for row in rows:
        judged = row.get("authored", True)
        if not row.get("present"):
            if judged:
                findings.append({
                    "code": "MISSING", "object": row["name"],
                    "reason": "authored in the config but absent from the built scene",
                })
            continue
        drift_xy = row.get("drift_xy")
        drift_z = row.get("drift_z")
        authored_pos = row.get("authored_position")
        # A failed pack is authored at z = support + DROP_HEIGHT and then falls; by probe time its
        # drift reads NEGATIVE, so the +Z signature is the authored height, not the drift. Resting
        # on the support is what distinguishes it from a genuinely floating object authored at the
        # same height (bbox_z = 0.10 makes the two authored poses identical).
        dropped = (authored_pos and support_z is not None and drift_z is not None
                   and not row.get("explicitly_placed", False)
                   and abs((authored_pos[2] - support_z) - DROP_HEIGHT_M) <= DROP_TOL_M
                   and drift_z < -DROP_TOL_M)
        if not judged:
            continue
        if dropped:
            findings.append({
                "code": "DROPPED", "object": row["name"],
                "reason": (
                    f"authored {authored_pos[2] - support_z:.4f} m above the support -- placement.py's "
                    f"{DROP_HEIGHT_M} m drop height, which it uses when it cannot find a "
                    f"collision-free slot -- and fell {abs(drift_z):.4f} m to come to rest. The "
                    f"layout has no room for this footprint: re-solve with more clearance or a "
                    f"smaller object."
                ),
            })
        elif (drift_xy is not None and drift_xy > DRIFT_FAIL_M
              and row.get("role", "main_objects") != "distractors" and row["name"] not in resting):
            # Clutter sliding a few cm, or an object moving about INSIDE the container it starts in
            # (a mug in a sink basin), is not a broken start state; tipping or leaving the
            # container is, and relation_findings judges that directly.
            findings.append({
                "code": "UNSTABLE", "object": row["name"],
                "reason": (
                    f"settled {drift_xy:.4f} m from its authored XY, past the {DRIFT_FAIL_M} m "
                    f"threshold: the layout is not resting where it was placed"
                ),
            })
        elif drift_xy is not None and drift_xy > DRIFT_WARN_M:
            findings.append({
                "code": "DRIFT", "object": row["name"],
                "reason": f"settled {drift_xy:.4f} m from its authored XY",
            })
        start_low = row.get("start_aabb_low_z")
        if (start_low is not None and surface is not None and on_task_surface(row, support_z)
                and row["name"] not in resting):
            row["start_gap_m"] = round(float(start_low) - surface, 5)
            if row["start_gap_m"] > HIGH_DROP_M:
                findings.append({
                    "code": "HIGH_DROP", "object": row["name"],
                    "reason": (f"starts with its lower face {row['start_gap_m']:.3f} m above the "
                               f"settled surface at z={surface:.4f}; it is dropped that far before "
                               f"the policy starts"),
                })
        bbox = row.get("bbox")
        lowest = row.get("aabb_low_z")
        if (lowest is not None and surface is not None and on_task_surface(row, support_z)
                and lowest < surface - PENETRATION_TOL_M):
            findings.append({
                "code": "PENETRATION", "object": row["name"],
                "reason": (
                    f"lower face at z={lowest:.4f} is {surface - lowest:.4f} m below the measured "
                    f"support surface at z={surface:.4f}"
                ),
            })
        elif (lowest is not None and lowest < FLOOR_Z_M - PENETRATION_TOL_M
              and row.get("role", "main_objects") in MOVABLE_ROLES):
            findings.append({
                "code": "PENETRATION", "object": row["name"],
                "reason": f"lower face at z={lowest:.4f} is below the floor at z={FLOOR_Z_M}",
            })
        if row.get("authored_position") and bbox and support_z is not None and row["name"] not in resting:
            authored_low = row["authored_position"][2] - float(bbox[2]) / 2
            if authored_low > support_z + 0.10:
                findings.append({
                    "code": "FLOATING", "object": row["name"],
                    "reason": (
                        f"authored lower face sits {authored_low - support_z:.4f} m above the "
                        f"support with nothing holding it up"
                    ),
                })
    return findings


def frozen_physics_findings(rows: list[dict]) -> list[dict]:
    """PHYSICS_FROZEN when no movable object moved during the settle.

    REALM authors every object above its resting height and lets it drop, so a settle that leaves
    them all exactly where they spawned means the simulator never stepped -- and every other
    "clean" number in the record is then meaningless.
    """
    moved = [abs(float(row.get("drift_z") or 0.0)) + float(row.get("drift_xy") or 0.0)
             for row in rows if row.get("present") and row.get("role") in MOVABLE_ROLES
             and row.get("drift_z") is not None]
    if moved and max(moved) < FROZEN_FALL_M:
        return [{
            "code": "PHYSICS_FROZEN", "object": "*",
            "reason": (f"no movable object moved more than {FROZEN_FALL_M} m during the settle "
                       f"(largest {max(moved):.5f} m): the simulator did not step"),
        }]
    return []


def collect_rgb(obs, prefix: str = "") -> list[tuple[str, object]]:
    """Every (camera-name, array) pair whose obs leaf is an `rgb` image.

    Obs is **nested**, keyed `obs['external']['external_sensor0']['rgb']` and
    `obs[<robot>]['<robot>:<link>:Camera:<i>']['rgb']` -- see `inference.utils.extract_from_obs`. A
    flat scan of the top-level keys for a name starting with `rgb` matches nothing, which is how this
    harness ended up writing zero images for every config.
    """
    found: list[tuple[str, object]] = []
    if isinstance(obs, dict):
        for key, value in obs.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if key == "rgb":
                # Name the frame after the camera, not the leaf.
                found.append((prefix or path, value))
            else:
                found.extend(collect_rgb(value, path))
    return found


def _to_uint8(array):
    array = array.detach().cpu().numpy() if hasattr(array, "detach") else np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype("uint8")
    return array


def render(env, out_dir: Path, stem: str, timestep: str) -> tuple[list[str], list[str]]:
    """Render every configured view to PNG. Returns (paths written, errors).

    Errors are returned, not swallowed: a harness that writes no images and reports no failure is
    indistinguishable from a scene that rendered fine, and the vision stage downstream would then be
    reviewing nothing while reporting a clean pass.
    """
    written, errors = [], []
    # PIL, not cv2: PIL is already a hard dependency of the sim image (realm_logging encodes video
    # through it), whereas opencv is declared nowhere.
    try:
        from PIL import Image
    except ImportError:
        return written, ["PIL is not importable; no images could be written"]
    try:
        import omnigibson as og
        # Sensor buffers are only refreshed on an explicit render under render-on-demand, and the
        # RTX pipeline hands back frames a few renders late. One render after a settle run with
        # `render_on_step(False)` returned the PRE-settle frame: every t060 image showed the
        # objects still in the air at their spawn height. REALM's own warmup flushes 30 renders.
        for _ in range(RENDER_FLUSH_PASSES):
            og.sim.render()
    except Exception as error:
        errors.append(f"og.sim.render() failed: {type(error).__name__}: {error}")
    try:
        obs, _ = env.omnigibson_env.get_obs()
    except Exception as error:
        errors.append(f"get_obs failed: {type(error).__name__}: {error}")
        return written, errors

    frames = collect_rgb(obs)
    if not frames:
        errors.append(f"obs carried no rgb leaf; top-level keys were {sorted(obs or {})}")
    for name, array in frames:
        try:
            slug = name.replace("/", "_").replace(":", "_")
            path = out_dir / f"{stem}_{slug}_{timestep}.png"
            Image.fromarray(_to_uint8(array)).save(path)
            written.append(str(path))
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    return written, errors


#: Pixels of the main object a camera must see for it to count as visible in that view.
VISIBLE_MIN_PX = 60
#: Ring around the object's mask, in pixels, that stands in for "what it is seen against".
CONTRAST_RING_PX = 6


def luminance(rgb) -> np.ndarray:
    """Rec. 709 luma of an HxWx3 (or x4) uint8 image, 0-255."""
    rgb = np.asarray(rgb, dtype=float)[..., :3]
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def mask_contrast(rgb, mask) -> dict:
    """How visible a masked object is: its pixel count and its luma against a ring around it.

    `contrast` is |median luma of the object - median luma of the ring| / 255. A black marker on a
    black table reads near 0 however many pixels it covers; pi05 failed exactly those scenes.
    """
    mask = np.asarray(mask, dtype=bool)
    pixels = int(mask.sum())
    if pixels == 0:
        return {"pixels": 0, "contrast": None}
    luma = luminance(rgb)
    ring = _dilate(mask, CONTRAST_RING_PX) & ~mask
    if not ring.any():
        return {"pixels": pixels, "contrast": None}
    inside, around = float(np.median(luma[mask])), float(np.median(luma[ring]))
    return {"pixels": pixels, "object_luma": round(inside, 1), "surround_luma": round(around, 1),
            "contrast": round(abs(inside - around) / 255.0, 3)}


#: Smallest per-pixel change (max over RGB, 0-255) counted as the object, whatever the noise floor.
CHANGE_MIN_LEVEL = 10
#: Margin added to the noise floor measured by rendering the same scene twice.
CHANGE_NOISE_MARGIN = 3


def _open3(mask: np.ndarray) -> np.ndarray:
    """Morphological opening with a 3x3 square: drops specks thinner than 3 px, keeps blob corners."""
    padded = np.pad(mask, 1)
    h, w = mask.shape
    shifts = [padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]
    eroded = np.logical_and.reduce(shifts)
    padded = np.pad(eroded, 1)
    return np.logical_or.reduce([padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)])


def change_mask(with_rgb, without_rgb, repeat_rgb=None) -> tuple[np.ndarray, int]:
    """Pixels that change when the object is hidden: the object as that camera sees it.

    `repeat_rgb` is the with-object frame rendered a second time; their difference is the renderer's
    own noise, and the threshold sits above it. A 3x3 opening removes isolated speckle, so noise
    can never add up to an object's worth of pixels.
    """
    a = np.asarray(with_rgb, dtype=np.int16)[..., :3]
    b = np.asarray(without_rgb, dtype=np.int16)[..., :3]
    diff = np.abs(a - b).max(axis=-1)
    threshold = CHANGE_MIN_LEVEL
    if repeat_rgb is not None:
        noise = np.abs(a - np.asarray(repeat_rgb, dtype=np.int16)[..., :3]).max(axis=-1)
        threshold = max(threshold, int(np.percentile(noise, 99.5)) + CHANGE_NOISE_MARGIN)
    mask = diff > threshold
    return _open3(mask) & mask, threshold


def object_visibility(with_rgb, without_rgb, repeat_rgb=None) -> dict:
    """Pixel count and contrast of one object in one camera, from a with/without pair of frames.

    `contrast` is the ring measure of `mask_contrast` (object vs the pixels around it).
    `contrast_behind` compares the object with what the camera sees in its place once it is hidden:
    the background it is actually seen against.
    """
    mask, threshold = change_mask(with_rgb, without_rgb, repeat_rgb)
    out = mask_contrast(with_rgb, mask)
    out["threshold"] = threshold
    if out["pixels"]:
        inside = float(np.median(luminance(with_rgb)[mask]))
        behind = float(np.median(luminance(without_rgb)[mask]))
        out["contrast_behind"] = round(abs(inside - behind) / 255.0, 3)
    return out


def _frames(env) -> dict:
    """Every camera's rgb, rendered the same way `render()` does (flush, then read)."""
    import omnigibson as og

    for _ in range(RENDER_FLUSH_PASSES):
        og.sim.render()
    obs, _ = env.omnigibson_env.get_obs()
    return {name: _to_uint8(array) for name, array in collect_rgb(obs)}


def _set_cast_shadows(obj, cast: bool) -> bool:
    """Turn the object's shadow off (or back on) so hiding it changes only its own pixels.

    Without this, the shadow it casts on the table counts as "the object", and an object hidden
    behind a box but casting a visible shadow would read as visible. Returns False if the primvar
    could not be set; the probe then still runs, shadow included, and says so.
    """
    try:
        import contextlib

        import omnigibson as og
        from pxr import Sdf, Usd, UsdGeom

        gprims = [prim for prim in Usd.PrimRange(obj.prim) if prim.IsA(UsdGeom.Gprim)]
        # OG 3.9.1 (Fabric Scene Delegate) only accepts raw USD writes inside editing_usd(): a
        # notice listener records an edit made outside it and re-raises it from a LATER call, which
        # is how the first version of this probe lost every measurement. The context must not be
        # nested, so nothing in here may call an OmniGibson setter (obj.visible opens its own).
        editing = getattr(og.sim, "editing_usd", None)
        with (editing() if editing is not None else contextlib.nullcontext()):
            for prim in gprims:
                primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool)
                primvar.Set(not cast)
        return True
    except Exception:
        return False


def _save_overlay(shown, hidden, repeat, path: Path) -> None:
    from PIL import Image

    mask, _ = change_mask(shown, hidden, repeat)
    overlay = np.asarray(shown, dtype=np.uint8)[..., :3].copy()
    overlay[mask] = (overlay[mask] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def visibility_probe(env, names: list[str], mask_dir: Path | None = None) -> tuple[dict, list[str]]:
    """Per camera and per named object: visible pixel count and contrast, from the RGB renders only.

    Each object is hidden for one render and the frames are compared with it shown. This replaced
    a `seg_instance` probe: adding that modality to the external cameras at runtime segfaulted Isaac
    inside `og.sim.render()` on a third of the DROID100 scenes (deterministically, per scene), which
    lost the task's JSON. Hiding a prim and rendering is what every task already does, so no new
    render products or annotators are created. The scene is restored afterwards (visibility and
    shadows); physics is not stepped. Errors are returned, not raised. With `mask_dir`, each
    camera's found pixels are saved as a red overlay on its frame, to check the probe by eye.
    """
    result: dict = {}
    errors: list[str] = []
    for name in names:
        try:
            obj = env.omnigibson_env.scene.object_registry("name", name)
        except Exception:
            obj = None
        if obj is None:
            errors.append(f"{name}: not in the scene")
            continue
        shadows_off = False
        try:
            shadows_off = _set_cast_shadows(obj, False)
            shown = _frames(env)
            repeat = _frames(env)
            obj.visible = False
            hidden = _frames(env)
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
            continue
        finally:
            try:
                obj.visible = True
            except Exception as error:
                errors.append(f"{name}: could not show it again: {error}")
            if shadows_off:
                _set_cast_shadows(obj, True)
        if not shadows_off:
            errors.append(f"{name}: shadow could not be turned off; pixel counts include it")
        for cam in sorted(set(shown) & set(hidden)):
            try:
                result.setdefault(name, {})[cam] = object_visibility(shown[cam], hidden[cam], repeat.get(cam))
                if mask_dir is not None:
                    _save_overlay(shown[cam], hidden[cam], repeat.get(cam),
                                  Path(mask_dir) / f"vis_{name}_{cam.replace('/', '_').replace(':', '_')}.png")
            except Exception as error:
                errors.append(f"{cam}: {type(error).__name__}: {error}")
    if names:
        try:
            sync_stage()
        except Exception:
            pass
    return result, errors


def visibility_findings(visibility: dict, main: str, min_contrast: float | None = None) -> list[dict]:
    """NOT_VISIBLE when no external camera sees the main object; LOW_CONTRAST when every camera that
    sees it sees it against a surround of nearly the same brightness. Both are reported, not
    failed, until the thresholds are calibrated against policy results."""
    views = visibility.get(main) or {}
    external = {cam: v for cam, v in views.items() if cam.startswith("external")}
    seen = {cam: v for cam, v in views.items() if (v.get("pixels") or 0) >= VISIBLE_MIN_PX}
    findings = []
    if not external:
        # A probe that measured nothing must not read as "visible": the smoke test of 916d13b
        # passed both tasks with an empty visibility block.
        return [{"code": "VISIBILITY_UNMEASURED", "object": main,
                 "reason": "no external camera has a visibility measurement; see visibility_errors"}]
    if external and not any((v.get("pixels") or 0) >= VISIBLE_MIN_PX for v in external.values()):
        findings.append({"code": "NOT_VISIBLE", "object": main,
                         "reason": f"fewer than {VISIBLE_MIN_PX} px in every external camera"})
    contrasts = [v["contrast"] for v in seen.values() if v.get("contrast") is not None]
    if min_contrast is not None and contrasts and max(contrasts) < min_contrast:
        findings.append({"code": "LOW_CONTRAST", "object": main,
                         "reason": f"best contrast against its surround {max(contrasts):.3f} "
                                   f"< {min_contrast} in every camera that sees it"})
    return findings


#: Calibrate from data before enforcing; None records the numbers without a finding.
LOW_CONTRAST_MIN = None


#: Findings that fail a task. PHYSICS_FROZEN is reported, not failed: hand-authored configs
#: (REALM_DROID10's drawers, switch, pour) place every object already at rest.
HARD_FINDINGS = {"MISSING", "UNSTABLE", "PENETRATION", "DROPPED", "TIPPED", "LEFT_CONTAINER",
                 "LEFT_SUPPORT", "GRASP_INFEASIBLE"}


def _verdict(record: dict) -> str:
    # No images at all is a harness failure, not a clean scene: never let it read as a pass.
    record["no_images"] = not record["renders"]
    failed = any(finding["code"] in HARD_FINDINGS for finding in record.get("compute_findings") or [])
    return "FAIL" if (failed or record["no_images"]) else "PASS"


def review_one(task_cfg_path: str, out_dir: Path, *, robot: str = "DROID_mounted",
               steps: int = WARMUP_STEPS, save_masks: bool = False) -> dict:
    """Build, settle, probe and render ONE task config.

    Process lifetime is the caller's problem, not this function's: `og.shutdown()` is terminal, so a
    function that called it could never be called twice. `main()` therefore gives each config its own
    process. The JSON is written before anything else can go wrong, including a raise in this
    function's own teardown, because that file is the only trustworthy output.
    """
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.environments.perturbations._helpers import settle
    from realm.sim_config import set_sim_config

    stem = Path(task_cfg_path).parent.name
    record: dict = {"task": stem, "task_cfg_path": task_cfg_path, "renders": [],
                    "render_errors": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{stem}.json"

    set_sim_config(robot=robot)
    # `perturbations` is a list, not `perturbation` -- the constructor iterates it, so a bare string
    # was both a TypeError and, before that, silently the wrong shape. `multi_view` is what makes the
    # config's cam2 exist at all: without it `_apply_camera_cfg` deletes the second external sensor,
    # so a "four view" review would have had one view to look at.
    env = RealmEnvironmentDynamic(
        task_cfg_path=task_cfg_path, perturbations=["Default"], robot=robot,
        rendering_mode="rt", no_rendering=False, multi_view=True,
    )
    try:
        env.reset()
        # Live poses before the settle: drift is measured against these, never against the authored
        # bbox-centre `position` (see `settle_drift`).
        sync_stage()
        before = snapshot(env)
        task_dir = out_dir / stem
        task_dir.mkdir(parents=True, exist_ok=True)
        paths, errors = render(env, task_dir, stem, "t000")
        record["renders"] += paths
        record["render_errors"] += errors
        # REALM's own settle: hold the arm at reset_qpos with the gripper open, skipping camera work.
        # Raw `og.sim.step()` leaves the arm unactuated, so it sags into the scene and knocks objects
        # over -- an unstable-layout finding the policy would never actually see.
        settle(env, steps=steps)
        region = getattr(env, "spawn_bbox", None)
        support_z = float(region[4]) if region is not None else None
        # Without this the AABBs (and so PENETRATION and the measured surface) are the spawn pose.
        sync_stage()
        rows = probe(env, before)
        record["objects"] = rows
        record["support_z"] = support_z
        resting = declared_resting(task_cfg_path)
        record["support_z_measured"] = measured_support_z(rows, support_z, resting)
        # Say which penetration check actually ran, so a null surface is not read as a clean one.
        record["penetration_check"] = (
            "measured task surface and floor" if record["support_z_measured"] is not None
            else "floor only: fewer than two authored objects rest on the spawn plane")
        record["settle_steps"] = steps
        record["compute_findings"] = stability_findings(rows, support_z, resting)
        record["compute_findings"] += relation_findings(rows, declared_relations(task_cfg_path))
        record["compute_findings"] += frozen_physics_findings(rows)
        paths, errors = render(env, task_dir, stem, "t060")
        record["renders"] += paths
        record["render_errors"] += errors
        main_names = [str(o.get("name")) for o in env.cfg.get("main_objects") or []]
        # A main object declared on top of something and still on it (no LEFT_SUPPORT) is held off
        # the table by that support, so its edge is reachable however thin it is.
        # ...but only when it overhangs that support (a plate across a bowl); a bar lying on a wide
        # tray is as ungraspable as on the table.
        left = {f["object"] for f in record["compute_findings"] if f["code"] == "LEFT_SUPPORT"}
        bboxes = {row["name"]: row.get("bbox") for row in rows if row.get("bbox")}
        on_support = {subject for predicate, subject, support in declared_relations(task_cfg_path)
                      if predicate == "on_top_of" and subject not in left
                      and bboxes.get(subject) and bboxes.get(support)
                      and min(bboxes[subject][:2]) > min(bboxes[support][:2])}
        record["compute_findings"] += grasp_findings(rows, main_names, on_support)
        # Everything measured so far is written BEFORE the visibility probe, so a crash inside it
        # (it renders with objects hidden) still leaves a reviewable record and its images.
        record["visibility"] = {}
        record["visibility_errors"] = ["visibility probe did not finish: the process died inside it"]
        record["verdict"] = _verdict(record)
        out_json.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        record["visibility"], record["visibility_errors"] = visibility_probe(
            env, main_names, mask_dir=task_dir if save_masks else None)
        for name in main_names:
            record["compute_findings"] += visibility_findings(record["visibility"], name, LOW_CONTRAST_MIN)
        record["verdict"] = _verdict(record)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
        record["verdict"] = "ERROR"

    out_json.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return record


def _child_command(target: str, out: Path, args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-u", str(Path(__file__).resolve()),
            "--task_cfg_path", target, "--out", str(out),
            "--robot", args.robot, "--steps", str(args.steps)] + (["--save-masks"] if args.save_masks else [])


def _read_record(out: Path, target: str) -> dict:
    """The child's JSON is the only output worth trusting; its exit code is not evidence."""
    stem = Path(target).parent.name
    path = out / f"{stem}.json"
    if not path.is_file():
        return {"task": stem, "task_cfg_path": target, "verdict": "NO_JSON",
                "error": f"child wrote no {path.name} (crash, OOM or timeout)"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"task": stem, "task_cfg_path": target, "verdict": "BAD_JSON",
                "error": f"{type(error).__name__}: {error}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_cfg_path", default=None,
                        help="one task config, e.g. REALM_DROID100/<task>/default.yaml")
    parser.add_argument("--family", default=None,
                        help="a task family directory or name; every */default.yaml under it is "
                             "reviewed, each in its own process")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "tmp" / "droid100" / "review")
    parser.add_argument("--robot", default="DROID_mounted")
    parser.add_argument("--steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--save-masks", action="store_true",
                        help="save each camera's main-object pixels as a red overlay (vis_*.png)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-config wall clock for --family children")
    args = parser.parse_args(argv)

    if args.task_cfg_path and args.family:
        print("give one of --task_cfg_path or --family, not both", file=sys.stderr)
        return 2

    if args.task_cfg_path:
        record = review_one(args.task_cfg_path, args.out, robot=args.robot, steps=args.steps,
                            save_masks=args.save_masks)
        _report(record)
        return 0

    if args.family:
        base = Path(args.family)
        if not base.is_dir():
            base = PROJECT_ROOT / "realm" / "config" / "tasks" / args.family
        if not base.is_dir():
            print(f"no such family directory: {base}", file=sys.stderr)
            return 2
        targets = sorted(str(path.relative_to(PROJECT_ROOT / "realm" / "config" / "tasks"))
                         for path in base.glob("*/default.yaml"))
        if not targets:
            print(f"no */default.yaml under {base}", file=sys.stderr)
            return 2
        results = []
        for index, target in enumerate(targets, start=1):
            print(f"\n=== [{index}/{len(targets)}] {target} ===", flush=True)
            # One process per config: og.shutdown() is terminal, and a config that segfaults or hangs
            # must not take the other 99 with it.
            try:
                subprocess.run(_child_command(target, args.out, args),
                               timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired:
                print(f"  killed after {args.timeout}s", flush=True)
            record = _read_record(args.out, target)
            results.append(record)
            _report(record)
        _summary(results, args.out)
        return 0

    print("give --task_cfg_path or --family", file=sys.stderr)
    return 2


def _report(record: dict) -> None:
    print(f"VERDICT {record['verdict']}  renders={len(record.get('renders') or [])}", flush=True)
    for finding in record.get("compute_findings") or []:
        print(f"  {finding['code']:<12} [{finding['object']}] {finding['reason']}", flush=True)
    for error in record.get("render_errors") or []:
        print(f"  {'RENDER':<12} {error}", flush=True)
    for error in record.get("visibility_errors") or []:
        print(f"  {'VISIBILITY':<12} {error}", flush=True)
    if record.get("error"):
        print(f"  ERROR {record['error']}", flush=True)


def _summary(results: list[dict], out: Path) -> None:
    tally: dict[str, int] = {}
    for record in results:
        tally[record["verdict"]] = tally.get(record["verdict"], 0) + 1
    print("\n" + "=" * 70, flush=True)
    print("  ".join(f"{verdict}={count}" for verdict, count in sorted(tally.items())), flush=True)
    for record in results:
        if record["verdict"] != "PASS":
            print(f"  {record['verdict']:<9} {record['task']}", flush=True)
    summary = out / "_summary.json"
    summary.write_text(json.dumps({"results": [
        {k: r.get(k) for k in ("task", "task_cfg_path", "verdict", "renders", "render_errors",
                               "error", "support_z", "support_z_measured",
                               "penetration_check")}
        for r in results], "tally": tally}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"summary -> {summary}", flush=True)
    print("Read the JSON, not the exit code: Isaac exits 0 on unhandled exceptions.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
