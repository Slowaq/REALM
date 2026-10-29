"""Grasp-and-lift test for each task's main object with DROID, cuRobo and physical grasping. CONTAINER (GPU).

    ./scripts/run_apptainer.sh python -u tmp/feasibility/grasp_test.py \
        --task_cfg_path REALM_DROID10/pick_spoon/default.yaml
    ./scripts/run_apptainer.sh python -u tmp/feasibility/grasp_test.py --family REALM_DROID10

Needs tmp/feasibility/droid_curobo/droid_curobo.yml (+ .usda) from droid_curobo.py. See README.md.
One simulator per process: --family spawns one child per task and reads back the JSON each wrote.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

STROKE_USABLE = 0.080      # 2F-85 stroke 85 mm minus 5 mm margin
FRICTION_DEG = 20.0        # contact normal within this of the closing axis
APPROACH_MAX_DEG = 70.0    # approach direction within this of straight down
PAD_BACKOFF = 0.02         # TCP is this far behind the fingertip (droid_curobo.py)
MAX_PAD_SHIFT = 0.015      # TCP may be pulled back this far to keep fingertips off the support
TIP_CLEARANCE = 0.004      # fingertips stay this far above the support surface
PREGRASP = 0.10
LIFT = 0.10
LIFT_OK = 0.07
HOLD_DRIFT_OK = 0.02
OBSTACLE_RADIUS = 1.5
MAX_OBSTACLE_MESHES = 1800
GRASP_TYPES = {"pick", "put", "rotate", "stack", "pour", "pour_proxy"}
# Gripper command sign is MEASURED at start-up (calibrate_gripper): on REALM's DROID, -1 opens and +1
# closes (the gap between the inner fingers is ~11.6 cm after -1 and ~3.3 cm after +1).
OPEN, CLOSE = -1.0, 1.0
# finger and palm envelopes in the grasp frame (x = y cross z, y = closing, z = approach; origin = TCP),
# with the gripper fully open (pads at |y| = 42.5 mm)
FINGER_Y = (0.0425, 0.056)
FINGER_X = 0.012
FINGER_Z = (-0.045, PAD_BACKOFF)
PALM_Z = -0.05
FINGER_CORNERS = np.array([[x, y, z] for x in (-FINGER_X, FINGER_X) for y in (-FINGER_Y[1], FINGER_Y[1])
                           for z in FINGER_Z], float)
PALM_XY = (0.03, 0.06)

REC: dict = {"stages": {}}
OUT = Path("tmp/feasibility/grasp")


def np_(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def mat(pos, quat_xyzw):
    from scipy.spatial.transform import Rotation as R
    m = np.eye(4)
    m[:3, :3] = R.from_quat(np_(quat_xyzw)).as_matrix()
    m[:3, 3] = np_(pos)
    return m


def geom_trimesh(cm):
    import trimesh
    from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh
    if cm.geom_type == "Mesh":
        return trimesh.Trimesh(np_(cm.points).astype(float), np_(cm.faces).astype(int), process=False)
    return mesh_prim_shape_to_trimesh_mesh(cm.prim)


def world_pieces(obj):
    """Collision geoms of an object as world-frame trimesh pieces."""
    pieces = []
    for link in obj.links.values():
        for cm in link.collision_meshes.values():
            tm = geom_trimesh(cm)
            tm.vertices = tm.vertices * np_(cm.get_world_scale())
            tm.apply_transform(mat(*cm.get_position_orientation()))
            if len(tm.faces):
                pieces.append(tm)
    return pieces


# ---------------------------------------------------------------- grasp sampling (pure geometry)
def surface_samples(pieces, n_total=2500, seed=0):
    """Surface points + outward normals, dropping points buried inside another (convex) piece."""
    import trimesh
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    areas = np.array([p.area for p in pieces])
    hulls = []
    for p in pieces:
        try:
            hulls.append(ConvexHull(p.vertices).equations)
        except Exception:
            hulls.append(None)
    pts, nrm = [], []
    for i, p in enumerate(pieces):
        k = max(30, int(n_total * areas[i] / areas.sum()))
        s, f = trimesh.sample.sample_surface(p, k, seed=int(rng.integers(1 << 30)))
        n = p.face_normals[f]
        flip = np.einsum("ij,ij->i", s - p.vertices.mean(0), n) < 0
        n[flip] *= -1
        keep = np.ones(len(s), bool)
        for j, eq in enumerate(hulls):
            if j == i or eq is None:
                continue
            inside = np.all(s @ eq[:, :3].T + eq[:, 3] < -1e-3, axis=1)
            keep &= ~inside
        pts.append(s[keep]); nrm.append(n[keep])
    return np.concatenate(pts), np.concatenate(nrm)


def grasp_frame(center, closing, approach):
    y = closing / np.linalg.norm(closing)
    z = approach - (approach @ y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, :3] = np.c_[x, y, z]
    m[:3, 3] = center
    return m


def collides(G, pts):
    """Object surface points inside the open fingers or the palm, for grasp pose G (4x4, world)."""
    return int(collides_many(G[None], pts)[0])


def collides_many(Gs, pts):
    """Vectorised collides() for a stack of grasp poses Gs (k, 4, 4)."""
    R, t = Gs[:, :3, :3], Gs[:, :3, 3]
    local = np.einsum("kji,knj->kni", R, pts[None] - t[:, None])   # R^T (p - t)
    x, y, z = local[..., 0], local[..., 1], local[..., 2]
    ay = np.abs(y)
    fingers = (np.abs(x) <= FINGER_X) & (ay >= FINGER_Y[0]) & (ay <= FINGER_Y[1]) \
        & (z >= FINGER_Z[0]) & (z <= FINGER_Z[1])
    palm = (z <= PALM_Z) & (z >= PALM_Z - 0.06) & (np.abs(x) <= PALM_XY[0]) & (ay <= PALM_XY[1])
    return (fingers | palm).sum(axis=1)


def sample_grasps(pieces, support_z, n_keep=12, seed=0, max_pairs=600):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    pts, nrm = surface_samples(pieces, seed=seed)
    allv = np.concatenate([p.vertices for p in pieces])
    com = allv.mean(0)
    size = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    tree = cKDTree(pts)
    cos_f = np.cos(np.radians(FRICTION_DEG))
    cos_a = np.cos(np.radians(APPROACH_MAX_DEG))
    stats = {"samples": len(pts), "antipodal_pairs": 0, "rejected_collision": 0, "rejected_support": 0}
    # 1) antipodal pairs: best partner per sample, vectorised per sample
    pairs = []
    for i, nbrs in enumerate(tree.query_ball_point(pts, STROKE_USABLE)):
        nbrs = np.asarray(nbrs)
        if len(nbrs) == 0:
            continue
        d = pts[nbrs] - pts[i]
        w = np.linalg.norm(d, axis=1)
        ok = w >= 0.004
        if not ok.any():
            continue
        nbrs, d, w = nbrs[ok], d[ok], w[ok]
        u = d / w[:, None]
        c1, c2 = u @ -nrm[i], np.einsum("ij,ij->i", u, nrm[nbrs])
        good = (c1 >= cos_f) & (c2 >= cos_f)
        if not good.any():
            continue
        k = np.argmax(np.where(good, c1 + c2, -9))
        if nbrs[k] < i and good[k]:
            pass  # keep both directions; the pair (j, i) gives the same grasp, the dedupe below handles it
        pairs.append((i, nbrs[k], w[k], u[k]))
    stats["antipodal_pairs"] = len(pairs)
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    # 2) approach directions around each closing axis, vectorised over the 12 angles
    thetas = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cands = []
    for i, j, w, u in pairs:
        center = (pts[i] + pts[j]) / 2
        e1 = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.95 else np.cross(u, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        A = np.cos(thetas)[:, None] * e1 + np.sin(thetas)[:, None] * e2
        A = A[-A[:, 2] >= cos_a]
        if len(A) == 0:
            continue
        # lowest corner of the two OPEN fingers must clear the support; the TCP may back off along -a
        G0 = np.stack([grasp_frame(center, u, a) for a in A])
        low = (G0[:, 2, :3] @ FINGER_CORNERS.T).min(axis=1) + center[2]
        shift = np.maximum(0.0, support_z + TIP_CLEARANCE - low) / np.maximum(-A[:, 2], 1e-3)
        bad = shift > MAX_PAD_SHIFT
        stats["rejected_support"] += int(bad.sum())
        A, shift = A[~bad], shift[~bad]
        if len(A) == 0:
            continue
        Gs = np.stack([grasp_frame(center - a * s_, u, a) for a, s_ in zip(A, shift)])
        local_pts = pts[tree.query_ball_point(center, 0.16)]
        hits = collides_many(Gs, local_pts)
        stats["rejected_collision"] += int((hits > 2).sum())
        central = 1.0 - min(1.0, np.linalg.norm((center - com)[:2]) / max(size, 1e-3))
        for G, a, s_, h in zip(Gs, A, shift, hits):
            if h > 2:
                continue
            down = float(-a[2])
            cands.append({"G": G, "width": float(w), "shift": float(s_),
                          "score": 1.5 * down + central + 0.5 * (1 - w / STROKE_USABLE),
                          "topdown_deg": float(np.degrees(np.arccos(np.clip(down, -1, 1))))})
    cands.sort(key=lambda c: -c["score"])
    chosen = []
    for c in cands:  # diversity: >= 1 cm or >= 25 deg from every chosen grasp
        if all(np.linalg.norm(c["G"][:3, 3] - k["G"][:3, 3]) > 0.01
               or np.degrees(np.arccos(np.clip(c["G"][:3, 2] @ k["G"][:3, 2], -1, 1))) > 25 for k in chosen):
            chosen.append(c)
        if len(chosen) >= n_keep:
            break
    stats["candidates"] = len(cands)
    stats["kept"] = len(chosen)
    return chosen, stats, pts


def baseline_grasp(pieces, support_z):
    """OmniGibson's own pose (AABB top-centre, straight down), closing across the narrower side."""
    allv = np.concatenate([p.vertices for p in pieces])
    lo, hi = allv.min(0), allv.max(0)
    ext = hi - lo
    closing = np.array([1.0, 0, 0]) if ext[0] <= ext[1] else np.array([0, 1.0, 0])
    top = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
    a = np.array([0, 0, -1.0])
    # fingertips go PAD_BACKOFF below the top, but never into the support
    tcp = top.copy()
    tip_z = tcp[2] - PAD_BACKOFF
    if tip_z < support_z + TIP_CLEARANCE:
        tcp[2] += support_z + TIP_CLEARANCE - tip_z
    return {"G": grasp_frame(tcp, closing, a), "width": float(min(ext[0], ext[1])), "score": None,
            "shift": 0.0, "topdown_deg": 0.0, "baseline": True}


# ---------------------------------------------------------------- sim helpers
def link_mat(robot, name):
    return mat(*robot.links[name].get_position_orientation())


def step_hold(env, q_arm, steps, gripper):
    import omnigibson as og
    with og.sim.render_on_step(False):
        for _ in range(steps):
            env.omnigibson_env.step(np.concatenate([q_arm, [gripper]]))


def build_world(cmg, robot, ignore=()):
    from curobo.geom.types import Mesh, WorldConfig
    from scipy.spatial.transform import Rotation as R
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as E
    base = link_mat(robot, "panda_link0")
    inv_b = np.linalg.inv(base)
    found = []
    for obj in robot.scene.objects:
        if obj is robot or obj in ignore or getattr(obj, "visual_only", False):
            continue
        lo, hi = (np_(v) for v in obj.aabb)
        dist = float(np.linalg.norm(np.clip(base[:3, 3], lo, hi) - base[:3, 3]))
        if dist > OBSTACLE_RADIUS:
            continue
        for link in obj.links.values():
            for cm in link.collision_meshes.values():
                found.append((dist, cm))
    found.sort(key=lambda x: x[0])
    meshes = []
    for _, cm in found[:MAX_OBSTACLE_MESHES]:
        tm = geom_trimesh(cm)
        P = inv_b @ mat(*cm.get_position_orientation())
        q = R.from_matrix(P[:3, :3]).as_quat()
        meshes.append(Mesh(name=cm.prim_path, pose=[*P[:3, 3].tolist(), q[3], q[0], q[1], q[2]],
                           vertices=np.asarray(tm.vertices, float), faces=np.asarray(tm.faces, int),
                           scale=np_(cm.get_world_scale()).tolist()))
    cmg.mg[E.DEFAULT].update_world(WorldConfig(mesh=meshes).get_collision_check_world())
    return len(meshes)


def plan(cmg, G, timeout=4.0):
    import torch as th
    from scipy.spatial.transform import Rotation as R
    quat = R.from_matrix(G[:3, :3]).as_quat()
    succ, paths = cmg.compute_trajectories(
        target_pos={"grasp_frame": th.tensor(G[:3, 3], dtype=th.float32)},
        target_quat={"grasp_frame": th.tensor(quat, dtype=th.float32)},
        is_local=False, skip_obstacle_update=True, max_attempts=8, timeout=timeout)
    if succ is None or not len(succ) or not bool(succ[0]):
        return None
    return np_(cmg.path_to_joint_trajectory(paths[0]))


def tcp_world(robot, tcp_in_eef):
    return (link_mat(robot, "panda_link8") @ np.r_[tcp_in_eef, 1.0])[:3]


def finger_gap(robot, fingers):
    a, b = (np_(robot.links[f].get_position_orientation()[0]) for f in fingers)
    return float(np.linalg.norm(a - b))


def supports_of(robot, obj):
    """Scene objects the main object rests on: overlapping in xy, top within 3 cm of its bottom."""
    lo, hi = (np_(v) for v in obj.aabb)
    out = []
    for o in robot.scene.objects:
        if o is obj or o is robot:
            continue
        l2, h2 = (np_(v) for v in o.aabb)
        if l2[0] <= hi[0] and h2[0] >= lo[0] and l2[1] <= hi[1] and h2[1] >= lo[1] and abs(h2[2] - lo[2]) < 0.03:
            out.append(o)
    return out


def calibrate_gripper(env):
    """Command +1 then -1 and keep whichever leaves the fingers wider apart as OPEN."""
    global OPEN, CLOSE
    robot = env.robot
    fingers = list(robot.finger_link_names[robot.default_arm])
    q = np.asarray(env.reset_qpos[:7], float)
    step_hold(env, q, 25, 1.0)
    gap_pos = finger_gap(robot, fingers)
    step_hold(env, q, 25, -1.0)
    gap_neg = finger_gap(robot, fingers)
    OPEN, CLOSE = (1.0, -1.0) if gap_pos > gap_neg else (-1.0, 1.0)
    REC["gripper_calibration"] = {"gap_after_plus1": round(gap_pos, 4), "gap_after_minus1": round(gap_neg, 4),
                                  "open_command": OPEN, "close_command": CLOSE,
                                  "gap_open": round(max(gap_pos, gap_neg), 4),
                                  "gap_closed_empty": round(min(gap_pos, gap_neg), 4)}


def settle_open(env, steps=20):
    step_hold(env, np.asarray(env.reset_qpos[:7], float), steps, OPEN)


def attempt(env, cmg, obj, cand, arm_idx, tcp_in_eef, fingers):
    """Pre-grasp -> approach -> close -> lift -> hold. Returns a record."""
    robot = env.robot
    r = {"baseline": cand.get("baseline", False), "width": round(cand["width"], 4),
         "topdown_deg": round(cand["topdown_deg"], 1), "shift": round(cand["shift"], 4),
         "tcp": np.round(cand["G"][:3, 3], 4).tolist()}
    obj0 = np_(obj.get_position_orientation()[0])
    G = cand["G"]
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])  # same grasp, rotated 180 deg about the approach axis
    pre = None
    for variant, Gv in (("as_sampled", G), ("flipped", G @ flip)):
        P = Gv.copy()
        P[:3, 3] -= Gv[:3, 2] * PREGRASP
        build_world(cmg, robot)
        pre = plan(cmg, P)
        if pre is not None:
            r["variant"] = variant
            G = Gv
            break
    if pre is None:
        r["result"] = "PLAN_PREGRASP_FAIL"
        return r, False
    r["executed"] = True
    for q in pre[:, arm_idx]:
        step_hold(env, q, 1, OPEN)
    # Final approach: the object itself is not an obstacle. Fingertips end a few mm above the support,
    # which cuRobo's buffers count as contact, so if that refuses, retry without the support too.
    build_world(cmg, robot, ignore=(obj,))
    app = plan(cmg, G)
    supports = supports_of(robot, obj)
    r["supports"] = [o.name for o in supports]
    if app is None and supports:
        build_world(cmg, robot, ignore=(obj, *supports))
        app = plan(cmg, G)
        r["approach_ignored_support"] = app is not None
    if app is None:
        r["result"] = "PLAN_APPROACH_FAIL"
        return r, True
    for q in app[:, arm_idx]:
        step_hold(env, q, 1, OPEN)
    q_grasp = app[-1, arm_idx]
    r["tcp_err_at_grasp"] = round(float(np.linalg.norm(tcp_world(robot, tcp_in_eef) - G[:3, 3])), 4)
    obj_before = np_(obj.get_position_orientation()[0])
    r["obj_moved_on_approach"] = round(float(np.linalg.norm(obj_before - obj0)), 4)
    step_hold(env, q_grasp, 25, CLOSE)
    r["finger_gap_closed"] = round(finger_gap(robot, fingers), 4)
    L = G.copy()
    L[2, 3] += LIFT
    build_world(cmg, robot, ignore=(obj, *supports))   # straight up off the support
    lift = plan(cmg, L)
    if lift is None:
        r["result"] = "PLAN_LIFT_FAIL"
        return r, True
    z0 = float(np_(obj.get_position_orientation()[0])[2])
    for q in lift[:, arm_idx]:
        step_hold(env, q, 1, CLOSE)
    step_hold(env, lift[-1, arm_idx], 30, CLOSE)
    p1 = np_(obj.get_position_orientation()[0])
    rise = float(p1[2] - z0)
    rel0 = obj_before - G[:3, 3]
    rel1 = p1 - tcp_world(robot, tcp_in_eef)
    drift = float(np.linalg.norm(rel1 - rel0))
    r.update(rise=round(rise, 4), hold_drift=round(drift, 4),
             is_grasping=str(robot.is_grasping(candidate_obj=obj)))
    ok = rise >= LIFT_OK and drift <= HOLD_DRIFT_OK
    r["result"] = "LIFTED" if ok else ("DROPPED" if rise < 0.02 else "SLIPPED")
    return r, True


def stage(name):
    def wrap(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                out = fn(*a, **k)
                REC["stages"][name] = f"PASS ({time.time() - t0:.1f}s)"
                return out
            except Exception as error:
                REC["stages"][name] = f"FAIL: {type(error).__name__}: {error}"
                REC[f"{name}_traceback"] = traceback.format_exc()
                print(REC[f"{name}_traceback"])
                return None
            finally:
                save()
        return run
    return wrap


def save():
    OUT.mkdir(parents=True, exist_ok=True)
    name = REC.get("task_name", "task")
    (OUT / f"{name}.json").write_text(json.dumps(REC, indent=2, default=str) + "\n")


@stage("S1_env")
def s1_env(args):
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.sim_config import set_sim_config
    set_sim_config(robot=args.robot)
    env = RealmEnvironmentDynamic(task_cfg_path=args.task_cfg_path, perturbations=["Default"],
                                  robot=args.robot, rendering_mode="rt", no_rendering=False)
    env.reset()
    calibrate_gripper(env)
    settle_open(env, 30)
    return env


@stage("S2_curobo")
def s2_curobo(env, args):
    from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
    d = Path(args.curobo_dir).resolve()
    kw = dict(robot_cfg_path=str(d / "droid_curobo.yml"), robot_usd_path=str(d / "droid_curobo.usda"),
              use_default_embodiment_only=True, batch_size=2)
    try:
        cmg = CuRoboMotionGenerator(env.robot, **kw)
        REC["cuda_graph"] = True
    except AssertionError:
        cmg = CuRoboMotionGenerator(env.robot, use_cuda_graph=False, **kw)
        REC["cuda_graph"] = False
    import yaml
    k = yaml.safe_load((d / "droid_curobo.yml").read_text())["robot_cfg"]["kinematics"]
    tcp = np.array(k["extra_links"]["grasp_frame"]["fixed_transform"][:3], float)
    return cmg, tcp


@stage("S3_grasps")
def s3_grasps(env, args):
    obj = env.main_objects[0]
    pieces = world_pieces(obj)
    lo, hi = (np_(v) for v in obj.aabb)
    support_z = float(lo[2])
    cands, stats, _ = sample_grasps(pieces, support_z, n_keep=args.max_exec * 3)
    REC["object"] = {"name": obj.name, "category": getattr(obj, "category", None),
                     "model": getattr(obj, "model", None), "aabb_extent": np.round(hi - lo, 4).tolist(),
                     "collision_pieces": len(pieces)}
    REC["sampling"] = stats
    return obj, pieces, support_z, cands


@stage("S4_attempts")
def s4_attempts(env, cmg, tcp, obj, pieces, support_z, cands, args):
    robot = env.robot
    arm_idx = [cmg.robot_joint_names.index(j) for j in robot.arm_joint_names[robot.default_arm]]
    fingers = list(robot.finger_link_names[robot.default_arm])
    REC["finger_gap_open"] = round(finger_gap(robot, fingers), 4)
    attempts, executed = [], 0
    queue = [baseline_grasp(pieces, support_z)] + cands
    T0 = mat(*obj.get_position_orientation())
    for i, cand in enumerate(queue):
        if not cand.get("baseline") and executed >= args.max_exec:
            break
        D = mat(*obj.get_position_orientation()) @ np.linalg.inv(T0)
        moved = dict(cand, G=D @ cand["G"])
        r, ran = attempt(env, cmg, obj, moved, arm_idx, tcp, fingers)
        r["obj_offset_since_sampling"] = round(float(np.linalg.norm(D[:3, 3])), 4)
        r["index"] = i
        attempts.append(r)
        REC["attempts"] = attempts
        save()
        if ran and not cand.get("baseline"):
            executed += 1
        if r["result"] == "LIFTED" and not cand.get("baseline"):
            break
        if ran:  # put the scene back before the next try
            env.reset()
            settle_open(env, 20)
            obj = env.main_objects[0]
    REC["attempts"] = attempts


def verdict():
    att = REC.get("attempts") or []
    base = [a for a in att if a.get("baseline")]
    samp = [a for a in att if not a.get("baseline")]
    REC["baseline_result"] = base[0]["result"] if base else None
    REC["sampled_lifted"] = any(a["result"] == "LIFTED" for a in samp)
    if any(not v.startswith("PASS") for v in REC["stages"].values()):
        REC["verdict"] = "ERROR"
    elif REC.get("sampling", {}).get("kept", 0) == 0 and REC["baseline_result"] != "LIFTED":
        REC["verdict"] = "NO_CANDIDATES"
    elif REC["sampled_lifted"] or REC["baseline_result"] == "LIFTED":
        REC["verdict"] = "GRASP_OK"
    else:
        REC["verdict"] = "GRASP_FAIL"


def run_one(args) -> int:
    REC["task"] = args.task_cfg_path
    REC["task_name"] = Path(args.task_cfg_path).parent.name
    env = s1_env(args)
    if env is not None:
        res = s2_curobo(env, args)
        g = s3_grasps(env, args)
        if res and g:
            s4_attempts(env, *res, *g, args)
    verdict()
    save()
    short = {k: REC.get(k) for k in ("task_name", "stages", "object", "sampling", "gripper_calibration",
                                     "baseline_result", "sampled_lifted", "verdict")}
    short["attempts"] = [{k: a.get(k) for k in ("index", "baseline", "result", "width", "topdown_deg",
                                                "variant", "rise", "hold_drift", "is_grasping",
                                                "finger_gap_closed", "tcp_err_at_grasp")}
                         for a in REC.get("attempts") or []]
    print("=" * 30 + " SUMMARY " + "=" * 30)
    print(json.dumps(short, indent=1, default=str))
    print(f"VERDICT {REC['verdict']}  ->  {OUT / (REC['task_name'] + '.json')}")
    sys.stdout.flush()
    os._exit(0)


def run_family(args) -> int:
    import yaml
    root = Path(__file__).resolve().parents[2] / "realm/config/tasks"
    tasks = []
    for cfg in sorted((root / args.family).glob("*/default.yaml")):
        t = str(yaml.safe_load(cfg.read_text()).get("task_type", "")).strip('"')
        if t in GRASP_TYPES:
            tasks.append(f"{args.family}/{cfg.parent.name}/default.yaml")
    if args.limit:
        tasks = tasks[: args.limit]
    rows = []
    for t in tasks:
        name = Path(t).parent.name
        print(f"--- {name}", flush=True)
        cmd = [sys.executable, "-u", __file__, "--task_cfg_path", t, "--out", str(OUT),
               "--curobo_dir", args.curobo_dir, "--max_exec", str(args.max_exec), "--robot", args.robot]
        try:
            subprocess.run(cmd, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"    timeout after {args.timeout}s")
        f = OUT / f"{name}.json"
        rec = json.loads(f.read_text()) if f.exists() else {}
        att = rec.get("attempts") or []
        win = next((a for a in att if a.get("result") == "LIFTED" and not a.get("baseline")), None)
        rows.append({"task": name, "object": (rec.get("object") or {}).get("category"),
                     "verdict": rec.get("verdict", "NO_REPORT"), "baseline": rec.get("baseline_result"),
                     "tries": len([a for a in att if a.get("executed") and not a.get("baseline")]),
                     "win_width": win and win.get("width"), "win_topdown_deg": win and win.get("topdown_deg")})
        (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("=" * 30 + " FAMILY SUMMARY " + "=" * 30)
    for r in rows:
        print(f"{r['verdict']:14s} baseline={str(r['baseline']):20s} tries={r['tries']}  "
              f"{r['task']} ({r['object']})  win: w={r['win_width']} tilt={r['win_topdown_deg']}")
    ok = sum(r["verdict"] == "GRASP_OK" for r in rows)
    print(f"GRASP_OK {ok}/{len(rows)}  ->  {OUT / 'summary.json'}")
    return 0


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_cfg_path")
    ap.add_argument("--family")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--curobo_dir", default="tmp/feasibility/droid_curobo")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max_exec", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    OUT = Path(args.out).resolve()
    if args.family:
        return run_family(args)
    if not args.task_cfg_path:
        ap.error("give --task_cfg_path or --family")
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
