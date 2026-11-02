"""DROID cuRobo model: build it from the live robot, then verify it. RUNS IN THE CONTAINER (GPU).

    ./scripts/run_apptainer.sh python -u tmp/feasibility/droid_curobo.py \
        --task_cfg_path REALM_DROID10/pick_spoon/default.yaml --out tmp/feasibility/droid_curobo

Writes <out>/droid_curobo.yml, <out>/droid_curobo.usda (the kinematic copy of droid.usd that cuRobo
reads) and <out>/report.json. The JSON is rewritten after every stage, so a crash keeps what ran.

Stages:
  S1 env        build the REALM task and settle
  S2 robot      joints, links, arm/gripper split, USD articulation root, USD joint list
  S3 usd        kinematic copy of droid.usd: joints cuRobo's USD parser cannot read (D6/closed-loop,
                excludeFromArticulation) deactivated
  S4 spheres    per-link collision meshes -> fitted spheres; Franka's hand-tuned spheres reused for a
                panda link when they cover our mesh as well; coverage per link
  S5 tcp        grasp frame between the finger pads, from finger mesh geometry
  S6 yml        write droid_curobo.yml (gripper joints locked; OmniGibson refreshes them from sim)
  S7 load       CuRoboMotionGenerator(robot, yml); self/world collision at the reset pose
  S8 fk_sweep   move each arm joint +0.3 rad in sim; cuRobo FK vs sim link poses (catches axis flips)
  S9 plan_exec  plan the TCP to 12 cm above the main object, and 15 cm to the side of that; execute
                through REALM's joint controller; tracking, TCP error, scene-object drift
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

PAD_BACKOFF = 0.02        # TCP sits this far back from the fingertip, along the approach axis
SPHERE_BUFFER = 0.004     # collision_sphere_buffer, also the coverage tolerance
OBSTACLE_RADIUS = 1.5     # metres from the robot base; scene objects further away are not loaded
MAX_OBSTACLE_MESHES = 1800
FRANKA_LINKS = [f"panda_link{i}" for i in range(8)]

REC: dict = {"stages": {}}
OUT = Path("tmp/feasibility/droid_curobo")


def save():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(REC, indent=2, default=str) + "\n")


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


def np_(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def mat(pos, quat_xyzw):
    from scipy.spatial.transform import Rotation as R
    m = np.eye(4)
    m[:3, :3] = R.from_quat(np_(quat_xyzw)).as_matrix()
    m[:3, 3] = np_(pos)
    return m


def link_mat(robot, name):
    return mat(*robot.links[name].get_position_orientation())


def geom_trimesh(cm):
    import trimesh
    from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh
    if cm.geom_type == "Mesh":
        tm = trimesh.Trimesh(np_(cm.points).astype(float), np_(cm.faces).astype(int), process=False)
    else:
        tm = mesh_prim_shape_to_trimesh_mesh(cm.prim)
    return tm


def link_mesh(robot, name):
    """All collision geoms of a link, in that link's frame."""
    import trimesh
    link = robot.links[name]
    inv_l = np.linalg.inv(link_mat(robot, name))
    parts = []
    for cm in link.collision_meshes.values():
        tm = geom_trimesh(cm)
        tm.vertices = tm.vertices * np_(cm.get_world_scale())
        tm.apply_transform(inv_l @ mat(*cm.get_position_orientation()))
        parts.append(tm)
    return trimesh.util.concatenate(parts) if parts else None


def greedy_spheres(mesh, target=0.97, max_n=24, n_samples=2500, overhang=0.008):
    """Spheres chosen greedily until `target` of the surface lies within radius + buffer.

    Candidates are surface samples pushed inward along the face normal; a candidate's radius is its
    distance to the nearest surface sample plus at most `overhang`, so no sphere sticks out of the
    mesh by more than that (spheres are conservative: they may cover a little air, never miss metal).
    """
    import trimesh
    from scipy.spatial import cKDTree
    samples, faces = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[faces]
    if mesh.is_volume and mesh.volume < 0:
        normals = -normals
    tree = cKDTree(samples)
    depth_scale = max(float(mesh.extents.min()), 0.004)
    cands, radii = [], []
    for frac in (0.1, 0.2, 0.35, 0.5, 0.75):
        pts = samples - normals * depth_scale * frac
        d, idx = tree.query(pts)
        inside = np.einsum("ij,ij->i", samples[idx] - pts, normals[idx]) > 0  # nearest surface faces away
        keep = inside & (d > 0.002)
        for extra in (0.0, 0.5 * overhang, overhang):
            cands.append(pts[keep]); radii.append(d[keep] + extra)
    cands, radii = np.concatenate(cands), np.concatenate(radii)
    if len(cands) == 0:
        return np.zeros((0, 3)), np.zeros(0), 0.0
    sel = np.random.default_rng(0).choice(len(cands), size=min(len(cands), 6000), replace=False)
    cands, radii = cands[sel], radii[sel]
    cover = [set(tree.query_ball_point(c, r + SPHERE_BUFFER)) for c, r in zip(cands, radii)]
    covered, chosen = set(), []
    while len(chosen) < max_n and len(covered) < target * len(samples):
        gains = [len(c - covered) for c in cover]
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        chosen.append(best)
        covered |= cover[best]
    return cands[chosen], radii[chosen], len(covered) / len(samples)


def coverage(samples, centers, radii):
    if len(centers) == 0:
        return 0.0, float("inf")
    d = np.min(np.linalg.norm(samples[:, None, :] - centers[None], axis=2) - radii[None], axis=1)
    return float((d <= SPHERE_BUFFER).mean()), float(d.max())


# ---------------------------------------------------------------------------------------------------
@stage("S1_env")
def s1_env(args):
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.environments.perturbations._helpers import settle
    from realm.sim_config import set_sim_config
    set_sim_config(robot=args.robot)
    env = RealmEnvironmentDynamic(task_cfg_path=args.task_cfg_path, perturbations=["Default"],
                                  robot=args.robot, rendering_mode="rt", no_rendering=False)
    env.reset()
    settle(env, steps=30)
    REC["task"] = args.task_cfg_path
    REC["main_object"] = env.main_objects[0].name if env.main_objects else None
    return env


@stage("S2_robot")
def s2_robot(env):
    from pxr import Usd, UsdPhysics
    robot = env.robot
    arm = robot.default_arm
    arm_joints = list(robot.arm_joint_names[arm])
    all_joints = list(robot.joints.keys())
    info = {
        "model": robot.model, "usd_path": robot.usd_path, "prim_path": robot.prim_path,
        "root_link": robot.root_link_name, "arm": arm, "arm_joints": arm_joints,
        "gripper_joints": [j for j in all_joints if j not in arm_joints], "all_joints": all_joints,
        "links": list(robot.links.keys()), "eef": robot.eef_link_names[arm],
        "fingers": list(robot.finger_link_names[arm]), "n_dof": int(robot.n_dof),
        "reset_joint_pos": np_(robot.reset_joint_pos).round(4).tolist(),
        "joint_pos": np_(robot.get_joint_positions()).round(4).tolist(),
        "joints": [{"name": n, "type": str(j.joint_type), "lower": float(j.lower_limit),
                    "upper": float(j.upper_limit), "body0": str(j.body0), "body1": str(j.body1)}
                   for n, j in robot.joints.items()],
    }
    stage_ = Usd.Stage.Open(robot.usd_path)
    default = stage_.GetDefaultPrim()
    info["usd_default_prim"] = str(default.GetPath()) if default else None
    info["usd_articulation_roots"] = [str(p.GetPath()) for p in stage_.Traverse()
                                      if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    usd_joints = []
    for p in stage_.Traverse():
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            usd_joints.append({
                "path": str(p.GetPath()), "type": str(p.GetTypeName()),
                "body0": [str(t) for t in j.GetBody0Rel().GetTargets()],
                "body1": [str(t) for t in j.GetBody1Rel().GetTargets()],
                "exclude": bool(j.GetExcludeFromArticulationAttr().Get() or False),
                "schemas": [str(s) for s in p.GetAppliedSchemas()],
            })
    info["usd_joints"] = usd_joints
    REC["robot"] = info
    return info


@stage("S3_usd")
def s3_usd(info):
    from pxr import Usd, UsdPhysics
    src = Usd.Stage.Open(info["usd_path"])
    stage_ = Usd.Stage.Open(src.Flatten())
    deactivated, children = [], {}
    joints = [p for p in stage_.Traverse() if p.IsA(UsdPhysics.Joint)]
    for p in joints:
        j = UsdPhysics.Joint(p)
        parseable = p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint) \
            or p.IsA(UsdPhysics.FixedJoint)
        excluded = bool(j.GetExcludeFromArticulationAttr().Get() or False)
        if not parseable or excluded:
            deactivated.append({"path": str(p.GetPath()), "type": str(p.GetTypeName()), "excluded": excluded})
            continue
        for t in j.GetBody1Rel().GetTargets():
            children.setdefault(t.name, []).append(str(p.GetPath()))
    for d in deactivated:
        stage_.GetPrimAtPath(d["path"]).SetActive(False)
    out = OUT / "droid_curobo.usda"
    stage_.GetRootLayer().Export(str(out))
    # parent/child link edges, as cuRobo will see them
    edges = []
    for p in stage_.Traverse():
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if b0 and b1:
                edges.append((b0[0].name, b1[0].name, p.GetName(), str(p.GetTypeName())))
    # cuRobo's USD parser places the child at T0 * inv(T1) and then rotates about the joint's axis
    # LETTER in the child frame, ignoring localRot1. The physical axis in the child frame is
    # R(localRot1) @ e_axis, so that is what we hand it (usd_flip_joints), with a negative direction
    # expressed as a flipped joint sign (usd_flip_joint_limits -> joint_offset -1).
    from scipy.spatial.transform import Rotation as R
    unit = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    axes = {}
    for p in stage_.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
            j = UsdPhysics.RevoluteJoint(p) if p.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(p)
            letter = str(j.GetAxisAttr().Get())
            q1 = j.GetLocalRot1Attr().Get()
            im = q1.GetImaginary()
            v = R.from_quat([im[0], im[1], im[2], q1.GetReal()]).apply(unit[letter])
            k = int(np.argmax(np.abs(v)))
            axes[p.GetName()] = {"usd_axis": letter, "child_axis": np.round(v, 4).tolist(),
                                 "curobo_axis": "XYZ"[k], "negative": bool(v[k] < 0),
                                 "aligned": bool(abs(v[k]) > 0.999),
                                 "pos1_norm": round(float(np.linalg.norm(np.array(j.GetLocalPos1Attr().Get()))), 6)}
    roots = info["usd_articulation_roots"]
    # cuRobo keeps joints whose path starts with usd_robot_root: use the default prim (joints are
    # often siblings of the articulation-root link rather than below it).
    usd_root = info["usd_default_prim"] or "/"
    REC["usd"] = {"file": str(out), "usd_robot_root": usd_root, "articulation_roots": roots,
                  "deactivated": deactivated, "edges": edges,
                  "links_with_several_parent_joints": {k: v for k, v in children.items() if len(v) > 1},
                  "joint_axes": axes,
                  "misaligned_axes": [n for n, a in axes.items() if not a["aligned"]],
                  "joint_frame_offsets": {n: a["pos1_norm"] for n, a in axes.items() if a["pos1_norm"] > 1e-6}}
    return {"path": str(out), "root": usd_root, "edges": edges, "axes": axes}


@stage("S4_spheres")
def s4_spheres(robot, info):
    import trimesh
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    franka = load_yaml(join_path(get_robot_configs_path(), "spheres/franka_mesh.yml"))["collision_spheres"]
    spheres, report, meshes = {}, {}, {}
    for name in info["links"]:
        mesh = link_mesh(robot, name)
        if mesh is None or len(mesh.vertices) == 0:
            continue
        meshes[name] = mesh
        samples, _ = trimesh.sample.sample_surface(mesh, 3000)
        big = mesh.extents.max() > 0.06
        fitted = None
        try:
            # ~150 spheres in total keeps cuRobo on its fast self-collision kernel (< 16384 pair checks)
            # Start small, and only spend more spheres on a link that is still under-covered
            # (thin outer fingers, the base plate), so the total stays near 150.
            max_n, cap = (14, 20) if mesh.extents.max() > 0.15 else ((12, 16) if big else (5, 10))
            while True:
                c, r, cov_fit = greedy_spheres(mesh, target=0.95, max_n=max_n, overhang=0.012 if big else 0.004)
                if cov_fit >= 0.9 or max_n >= cap:
                    break
                max_n = min(cap, max_n + 3)
            if len(r):
                fitted = (c, r, "greedy_inscribed")
        except Exception as error:
            report.setdefault(name, {})["fit_error"] = f"{type(error).__name__}: {error}"
        entry = report.setdefault(name, {})
        entry["extents"] = np.round(mesh.extents, 4).tolist()
        choice = None
        if fitted is not None:
            cov, gap = coverage(samples, fitted[0], fitted[1])
            entry.update(fitted_n=len(fitted[1]), fitted_cov=round(cov, 3), fitted_gap=round(gap, 4),
                         fit_type=fitted[2])
            choice = ("fitted", fitted[0], fitted[1], cov)
        if name in franka and name in FRANKA_LINKS:
            fc = np.array([s["center"] for s in franka[name]], float)
            fr = np.array([s["radius"] for s in franka[name]], float)
            cov, gap = coverage(samples, fc, fr)
            entry.update(franka_n=len(fr), franka_cov=round(cov, 3), franka_gap=round(gap, 4))
            if cov >= 0.95 and (choice is None or cov >= choice[3] - 0.02):
                choice = ("franka", fc, fr, cov)
        if choice is None:
            entry["source"] = "none"
            continue
        entry["source"], entry["coverage"] = choice[0], round(choice[3], 3)
        entry["gap"] = entry.get("fitted_gap") if choice[0] == "fitted" else entry.get("franka_gap")
        spheres[name] = [{"center": np.round(c, 5).tolist(), "radius": round(float(r), 5)}
                         for c, r in zip(choice[1], choice[2])]
    REC["spheres"] = report
    REC["sphere_total"] = sum(len(v) for v in spheres.values())
    return spheres, meshes


@stage("S5_tcp")
def s5_tcp(robot, info, meshes):
    eef, fingers = info["eef"], info["fingers"]
    inv_e = np.linalg.inv(link_mat(robot, eef))
    per_finger = {}
    for f in fingers:
        m = meshes.get(f)
        if m is None:
            raise RuntimeError(f"finger link {f} has no collision mesh")
        v = (inv_e @ link_mat(robot, f) @ np.c_[m.vertices, np.ones(len(m.vertices))].T).T[:, :3]
        per_finger[f] = v
    pts = np.concatenate(list(per_finger.values()))
    centroid = pts.mean(0)
    axis = centroid / np.linalg.norm(centroid)
    tip = float((pts @ axis).max())
    lateral = centroid - (centroid @ axis) * axis
    tcp = lateral + axis * (tip - PAD_BACKOFF)
    cents = [v.mean(0) for v in per_finger.values()]
    closing = (cents[1] - cents[0]) if len(cents) == 2 else np.zeros(3)
    REC["tcp"] = {"eef": eef, "approach_axis_in_eef": np.round(axis, 4).tolist(),
                  "fingertip_dist": round(tip, 4), "tcp_in_eef": np.round(tcp, 4).tolist(),
                  "closing_axis_in_eef": np.round(closing / (np.linalg.norm(closing) or 1), 4).tolist(),
                  "finger_gap_now": round(float(np.linalg.norm(closing)), 4)}
    return tcp


def tree_ignore(edges, collision_links, eef):
    adj = {}
    for parent, child, _, _ in edges:
        adj.setdefault(parent, set()).add(child)
        adj.setdefault(child, set()).add(parent)
    def dist(a, b):
        seen, q = {a}, deque([(a, 0)])
        while q:
            n, d = q.popleft()
            if n == b:
                return d
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    q.append((m, d + 1))
        return 99
    # everything downstream of panda_link7 is the hand: no self-collision inside it
    down, q = {"panda_link7"}, deque(["panda_link7"])
    children = {}
    for parent, child, _, _ in edges:
        children.setdefault(parent, []).append(child)
    while q:
        for c in children.get(q.popleft(), []):
            if c not in down:
                down.add(c)
                q.append(c)
    hand = (down | {eef, "attached_object"})
    ignore = {}
    links = list(collision_links)
    for i, a in enumerate(links):
        for b in links[i + 1:]:
            if (a in hand and b in hand) or dist(a, b) <= 2:
                ignore.setdefault(a, []).append(b)
    return ignore, sorted(hand)


@stage("S6_yml")
def s6_yml(info, usd, spheres, tcp):
    import yaml
    eef = info["eef"]
    usd_links = {e[0] for e in usd["edges"]} | {e[1] for e in usd["edges"]}
    lockable = {e[2] for e in usd["edges"] if e[3] != "PhysicsFixedJoint"}
    collision_links = [l for l in info["links"] if l in spheres and l in usd_links]
    ignore, hand = tree_ignore(usd["edges"], collision_links + ["attached_object"], eef)
    locks = {j: None for j in info["gripper_joints"] if j in lockable}
    link_names = [l for l in ["panda_link2", "panda_link4", "panda_link6", eef] + info["fingers"]
                  if l in usd_links]
    base_link = "panda_link0" if "panda_link0" in usd_links else info["root_link"]
    cfg = {"robot_cfg": {"kinematics": {
        "use_usd_kinematics": True,
        "usd_path": usd["path"],
        "usd_robot_root": usd["root"],
        "usd_flip_joints": {n: a["curobo_axis"] for n, a in usd["axes"].items()},
        "usd_flip_joint_limits": [n for n, a in usd["axes"].items() if a["negative"]],
        "base_link": base_link,
        "ee_link": "grasp_frame",
        "link_names": link_names,
        "collision_link_names": collision_links + ["attached_object"],
        "collision_spheres": {l: spheres[l] for l in collision_links},
        "collision_sphere_buffer": SPHERE_BUFFER,
        "extra_collision_spheres": {"attached_object": 4},
        "self_collision_ignore": ignore,
        "self_collision_buffer": {},
        "use_global_cumul": True,
        "mesh_link_names": collision_links,
        "lock_joints": locks,
        "extra_links": {
            "grasp_frame": {"parent_link_name": eef, "link_name": "grasp_frame",
                            "fixed_transform": [*np.round(tcp, 5).tolist(), 1.0, 0.0, 0.0, 0.0],
                            "joint_type": "FIXED", "joint_name": "grasp_frame_joint"},
            "attached_object": {"parent_link_name": "grasp_frame", "link_name": "attached_object",
                                "fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_type": "FIXED",
                                "joint_name": "attach_joint"},
        },
        "cspace": {"joint_names": info["arm_joints"], "retract_config": None,
                   "null_space_weight": [1.0] * len(info["arm_joints"]),
                   "cspace_distance_weight": [1.0] * len(info["arm_joints"]),
                   "max_acceleration": 15.0, "max_jerk": 500.0},
    }}}
    path = OUT / "droid_curobo.yml"
    path.write_text("# DROID (Panda + Robotiq 2F-85) for cuRobo, generated by tmp/feasibility/droid_curobo.py\n"
                    + yaml.safe_dump(cfg, sort_keys=False))
    REC["yml"] = {"path": str(path), "base_link": base_link, "collision_links": collision_links,
                  "hand_links": hand, "locked_joints": list(locks), "link_names": link_names}
    return str(path), base_link


def build_world(cmg, robot, base_link):
    import torch as th
    from curobo.geom.types import Mesh, WorldConfig
    from scipy.spatial.transform import Rotation as R
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as E
    base = link_mat(robot, base_link)
    inv_b = np.linalg.inv(base)
    found = []
    for obj in robot.scene.objects:
        if obj is robot or getattr(obj, "visual_only", False):
            continue
        lo, hi = (np_(v) for v in obj.aabb)
        nearest = np.clip(base[:3, 3], lo, hi)
        dist = float(np.linalg.norm(nearest - base[:3, 3]))
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
    world = WorldConfig(mesh=meshes).get_collision_check_world()
    cmg.mg[E.DEFAULT].update_world(world)
    return {"meshes": len(meshes), "candidates": len(found)}


def fk_check(cmg, robot, base_link, links):
    import torch as th
    from curobo.types.state import JointState
    from scipy.spatial.transform import Rotation as R
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as E
    mg = cmg.mg[E.DEFAULT]
    q = robot.get_joint_positions().unsqueeze(0)
    js = JointState(position=cmg._tensor_args.to_device(q), joint_names=cmg.robot_joint_names)
    cmg.update_locked_joints(js, E.DEFAULT)
    q_ord = js.get_ordered_joint_state(mg.kinematics.joint_names).position
    links = [l for l in links if l in mg.kinematics.link_names]
    pose = mg.kinematics.get_link_poses(q_ord, links)
    pos = np_(pose.position).reshape(-1, 3)
    quat = np_(pose.quaternion).reshape(-1, 4)  # wxyz
    inv_b = np.linalg.inv(link_mat(robot, base_link))
    out = {}
    for i, l in enumerate(links):
        sim = inv_b @ link_mat(robot, l)
        dr = (R.from_quat(quat[i][[1, 2, 3, 0]]).inv() * R.from_matrix(sim[:3, :3])).magnitude()
        out[l] = (float(np.linalg.norm(sim[:3, 3] - pos[i])), float(dr))
    return out


@stage("S7_load")
def s7_load(env, yml, usd, base_link, spheres):
    import torch as th
    from curobo.types.state import JointState
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as E
    from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
    robot = env.robot
    t0 = time.time()
    # batch_size=2 is OmniGibson's own default (the path its warmup asserts are exercised on). If the
    # post-warmup CUDA-graph asserts still fire, retry once without CUDA graphs and record why.
    try:
        cmg = CuRoboMotionGenerator(robot, robot_cfg_path=yml, robot_usd_path=usd["path"],
                                    use_default_embodiment_only=True, batch_size=2)
        REC["cuda_graph"] = True
    except AssertionError:
        REC["load_first_try_traceback"] = traceback.format_exc()
        print(REC["load_first_try_traceback"])
        cmg = CuRoboMotionGenerator(robot, robot_cfg_path=yml, robot_usd_path=usd["path"],
                                    use_default_embodiment_only=True, batch_size=2, use_cuda_graph=False)
        REC["cuda_graph"] = False
    REC["load_s"] = round(time.time() - t0, 1)
    mg = cmg.mg[E.DEFAULT]
    REC["curobo_joint_names"] = list(mg.kinematics.joint_names)
    REC["world"] = build_world(cmg, robot, base_link)
    q = robot.get_joint_positions().unsqueeze(0)
    js = JointState(position=cmg._tensor_args.to_device(q), joint_names=cmg.robot_joint_names)
    cmg.update_locked_joints(js, E.DEFAULT)
    q_ord = js.get_ordered_joint_state(mg.kinematics.joint_names)
    sph = mg.compute_kinematics(q_ord).robot_spheres.unsqueeze(1)
    with th.no_grad():
        self_c = float(mg.rollout_fn.robot_self_collision_constraint.forward(sph).max())
        world_c = float(mg.rollout_fn.primitive_collision_constraint.forward(sph).max())
    REC["reset_collision"] = {"self_cost": self_c, "world_cost": world_c}
    # independent check of the ignore list: sphere overlaps between non-ignored links, from sim poses
    import yaml
    ign = yaml.safe_load(Path(yml).read_text())["robot_cfg"]["kinematics"]["self_collision_ignore"]
    pairs = {(a, b) for a, bs in ign.items() for b in bs} | {(b, a) for a, bs in ign.items() for b in bs}
    world_sph = {}
    for l, ss in spheres.items():
        M = link_mat(robot, l)
        world_sph[l] = [((M @ np.r_[s["center"], 1.0])[:3], s["radius"]) for s in ss]
    overlaps = []
    names = list(world_sph)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (a, b) in pairs:
                continue
            worst = max(ra + rb - np.linalg.norm(ca - cb) for ca, ra in world_sph[a] for cb, rb in world_sph[b])
            if worst > 0:
                overlaps.append((a, b, round(float(worst), 4)))
    REC["reset_self_overlaps_not_ignored"] = sorted(overlaps, key=lambda x: -x[2])[:15]
    REC["fk_reset"] = {k: [round(v[0], 5), round(v[1], 4)]
                       for k, v in fk_check(cmg, robot, base_link, list(spheres)).items()}
    return cmg


def step_hold(env, q_arm, steps, gripper=-1.0):
    import omnigibson as og
    with og.sim.render_on_step(False):
        for _ in range(steps):
            env.omnigibson_env.step(np.concatenate([q_arm, [gripper]]))


@stage("S8_fk_sweep")
def s8_fk_sweep(env, cmg, info, base_link, spheres):
    robot = env.robot
    q_reset = np.asarray(env.reset_qpos[:7], float)
    lims = {j["name"]: (j["lower"], j["upper"]) for j in info["joints"]}
    links = [l for l in ["panda_link2", "panda_link4", "panda_link6", info["eef"]] + info["fingers"]]
    links += [l for l in spheres if l not in links]
    sweep, worst = {}, (0.0, 0.0)
    for k, j in enumerate(info["arm_joints"]):
        q = q_reset.copy()
        lo, hi = lims[j]
        q[k] = q[k] + 0.3 if q[k] + 0.3 < hi - 0.05 else q[k] - 0.3
        step_hold(env, q, 30)
        err = fk_check(cmg, robot, base_link, links)
        p = max(v[0] for v in err.values())
        r = max(v[1] for v in err.values())
        sweep[j] = {"pos_err_max": round(p, 5), "rot_err_max": round(r, 4),
                    "worst_link": max(err, key=lambda l: err[l][0])}
        worst = (max(worst[0], p), max(worst[1], r))
    step_hold(env, q_reset, 30)
    REC["fk_sweep"] = sweep
    REC["fk_worst"] = {"pos_m": round(worst[0], 5), "rot_rad": round(worst[1], 4)}


@stage("S9_plan_exec")
def s9_plan_exec(env, cmg, info, base_link, tcp):
    import torch as th
    from scipy.spatial.transform import Rotation as R
    robot = env.robot
    arm_idx = [cmg.robot_joint_names.index(j) for j in info["arm_joints"]]
    mo = env.main_objects[0]
    lo, hi = (np_(v) for v in mo.aabb)
    above = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 0.12])
    E_now = link_mat(robot, info["eef"])
    quat = R.from_matrix(E_now[:3, :3]).as_quat()  # keep the current hand orientation (xyzw)
    tracked = [o for o in env.main_objects + env.target_objects + env.distractors if o is not None]
    results = []
    for name, target in [("above_main", above), ("side_of_that", above + np.array([0.0, 0.15, 0.0]))]:
        r = {"goal": name, "target": np.round(target, 4).tolist()}
        REC["world"] = build_world(cmg, robot, base_link)
        t0 = time.time()
        succ, paths = cmg.compute_trajectories(
            target_pos={"grasp_frame": th.tensor(target, dtype=th.float32)},
            target_quat={"grasp_frame": th.tensor(quat, dtype=th.float32)},
            is_local=False, skip_obstacle_update=True, max_attempts=10, timeout=5.0)
        r["plan_s"] = round(time.time() - t0, 3)
        r["success"] = bool(succ[0]) if succ is not None and len(succ) else False
        if r["success"]:
            traj = np_(cmg.path_to_joint_trajectory(paths[0]))[:, arm_idx]
            before = {o.name: np_(o.get_position_orientation()[0]) for o in tracked}
            track = []
            for wp in traj:
                step_hold(env, wp, 1)
                track.append(float(np.abs(np_(robot.get_joint_positions())[arm_idx] - wp).max()))
            step_hold(env, traj[-1], 20)
            E_end = link_mat(robot, info["eef"])
            tcp_w = (E_end @ np.r_[tcp, 1.0])[:3]
            rot = (R.from_quat(quat).inv() * R.from_matrix(E_end[:3, :3])).magnitude()
            after = {o.name: np_(o.get_position_orientation()[0]) for o in tracked}
            r.update(waypoints=len(traj), track_err_max_rad=round(max(track), 4),
                     track_err_final_rad=round(float(np.abs(np_(robot.get_joint_positions())[arm_idx]
                                                            - traj[-1]).max()), 4),
                     tcp_err_m=round(float(np.linalg.norm(tcp_w - target)), 4), rot_err_rad=round(float(rot), 4),
                     object_drift_max_m=round(max(float(np.linalg.norm(after[k] - before[k])) for k in before), 4))
        results.append(r)
        save()
    REC["plans"] = results


def verdict():
    problems = []
    st = REC["stages"]
    for k, v in st.items():
        if not v.startswith("PASS"):
            problems.append(f"{k} failed")
    low = [l for l, e in REC.get("spheres", {}).items()
           if e.get("coverage", 0) < 0.85 or e.get("gap", 1) > 0.02]
    if low:
        problems.append(f"sphere coverage < 0.85 or gap > 2 cm on {low}")
    fk = REC.get("fk_worst")
    if fk and (fk["pos_m"] > 0.005 or fk["rot_rad"] > 0.02):
        problems.append(f"FK mismatch {fk}")
    rc = REC.get("reset_collision")
    if rc and rc["self_cost"] > 0:
        problems.append("self-collision at reset")
    plans = REC.get("plans") or []
    if plans and not any(p.get("success") and p.get("tcp_err_m", 1) < 0.015 for p in plans):
        problems.append("no plan reached its target within 1.5 cm")
    REC["problems"] = problems
    REC["verdict"] = "DROID_CUROBO_OK" if not problems else "DROID_CUROBO_NEEDS_WORK"


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_cfg_path", default="REALM_DROID10/pick_spoon/default.yaml")
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out).resolve()
    env = s1_env(args)
    if env is not None:
        info = s2_robot(env)
        usd = s3_usd(info) if info else None
        res = s4_spheres(env.robot, info) if info else None
        spheres, meshes = res if res else ({}, {})
        tcp = s5_tcp(env.robot, info, meshes) if meshes else None
        out = s6_yml(info, usd, spheres, tcp) if (usd and spheres and tcp is not None) else None
        cmg = None
        if out:
            yml, base_link = out
            cmg = s7_load(env, yml, usd, base_link, spheres)
        if cmg is not None:
            s8_fk_sweep(env, cmg, info, base_link, spheres)
            s9_plan_exec(env, cmg, info, base_link, tcp)
    verdict()
    save()
    keys = ["stages", "usd", "tcp", "sphere_total", "reset_collision", "reset_self_overlaps_not_ignored",
            "fk_reset", "fk_worst", "plans", "problems", "verdict"]
    summary = {k: REC.get(k) for k in keys}
    summary["cuda_graph"] = REC.get("cuda_graph")
    for k in [k for k in REC if k.endswith("traceback")]:
        summary[k + "_tail"] = REC[k].strip().splitlines()[-6:]
    summary["sphere_table"] = {l: f"{e.get('source')} n={e.get('fitted_n') if e.get('source') == 'fitted' else e.get('franka_n')}"
                                  f" cov={e.get('coverage')} gap={e.get('gap')} (fitted {e.get('fitted_cov')}, franka {e.get('franka_cov')})"
                               for l, e in (REC.get("spheres") or {}).items()}
    if summary.get("usd"):
        summary["usd"] = {k: summary["usd"].get(k) for k in
                          ("usd_robot_root", "articulation_roots", "deactivated", "links_with_several_parent_joints",
                           "misaligned_axes", "joint_frame_offsets")}
        summary["arm_axes"] = {n: f"{a['usd_axis']} -> {'-' if a['negative'] else '+'}{a['curobo_axis']}"
                               for n, a in (REC.get("usd", {}).get("joint_axes") or {}).items() if n.startswith("panda_joint")}
    print("=" * 30 + " SUMMARY " + "=" * 30)
    print(json.dumps(summary, indent=1, default=str))
    print(f"VERDICT {REC['verdict']}  ->  {OUT / 'report.json'}")
    os._exit(0)  # skip Isaac teardown (it can segfault after a passing run)


if __name__ == "__main__":
    raise SystemExit(main())
