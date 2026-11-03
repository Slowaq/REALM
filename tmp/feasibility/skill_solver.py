"""Scripted privileged solver for REALM's skills, scored by REALM's own progression ladder. CONTAINER (GPU).

    ./scripts/run_apptainer.sh python -u tmp/feasibility/skill_solver.py \
        --task_cfg_path REALM_DROID10/open_drawer/default.yaml
    ./scripts/run_apptainer.sh python -u tmp/feasibility/skill_solver.py --family REALM_DROID10

Privileged: it reads object meshes, poses and the drawer joint straight from the simulator, plans with
cuRobo (tmp/feasibility/droid_curobo/droid_curobo.yml) and executes through REALM's own env.step,
exactly as an evaluation does: env.reset() -> env.warmup() -> env.step(7 joint targets + gripper).
The score is env.step's task_progression (fraction of the task_progressions.yaml ladder reached).

Skills:
  pick          grasp, lift 15 cm
  put / stack   grasp, lift, carry over the target, lower to 3 cm (put) / 1 cm (stack) above its top,
                open, back off
  rotate        grasp, lift 3 cm, turn the wrist (joint 7) by ~1.6 rad
  push          press a closed fingertip into the toggle marker along the switch's outward normal
  open_drawer   grasp a handle on the drawer front (horizontal approach), pull slowly along the slide axis
  close_drawer  push the drawer front with a closed fist, slowly along the slide axis
Grasping skills try OmniGibson's AABB pose first, then mesh-sampled grasps (grasp_test.py); each failed
attempt resets the task. Reports whether the solve fits REALM's default 500-step policy budget.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_test as gt  # noqa: E402  (same folder: sampling, planning world, gripper calibration)
from grasp_test import mat, np_, sync_stage  # noqa: E402

SKILLS = {"pick", "put", "rotate", "push", "stack", "open_drawer", "close_drawer"}
POLICY_BUDGET = 500          # REALM's default --max_steps
MAX_ATTEMPTS = 4
PICK_LIFT = 0.15
PUT_CLEARANCE = {"put": 0.03, "stack": 0.01}
CARRY_HEIGHT = 0.10          # carry this far above the drop pose
ROTATE_LIFT = 0.03
ROTATE_ANGLE = 1.6           # rad; REALM's ROTATED needs > 1.1
SLOW_JOINT_STEP = 0.002      # rad per control step for drawer pulls/pushes (the drawer is damped, kd=1000)
PUSH_DEPTHS = (0.004, 0.009, 0.014)   # fingertip depth into the toggle marker, tried in turn
DRAWER_FRONT_BAND = 0.03     # handle contacts must lie within this of the front-most point of the drawer
DRAWER_OVERSHOOT = 0.01

REC: dict = {"stages": {}}
OUT = Path("tmp/feasibility/solver")


def save():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{REC.get('task_name', 'task')}.json").write_text(json.dumps(REC, indent=2, default=str) + "\n")


class Solver:
    def __init__(self, env, cmg, tcp):
        self.env, self.cmg, self.tcp = env, cmg, tcp
        r = env.robot
        self.robot = r
        self.arm_joints = list(r.arm_joint_names[r.default_arm])
        self.arm_idx = [cmg.robot_joint_names.index(j) for j in self.arm_joints]
        self.fingers = list(r.finger_link_names[r.default_arm])
        self.limits = [(float(r.joints[j].lower_limit), float(r.joints[j].upper_limit)) for j in self.arm_joints]
        self.steps, self.best, self.log = 0, 0.0, []

    # ------------------------------------------------------------------ episode / stepping
    def new_episode(self):
        import omnigibson as og
        obs, _ = self.env.reset()
        with og.sim.render_on_step(False):
            self.env.warmup(obs)                 # REALM's own warmup: settles, opens, captures references
        sync_stage(30)
        self.q = np.asarray(self.env.reset_qpos[:7], float).copy()
        self.g = gt.OPEN
        self.steps, self.best, self.log = 0, 0.0, []

    def step(self, q=None, g=None, n=1):
        import omnigibson as og
        if q is not None:
            self.q = np.asarray(q, float)
        if g is not None:
            self.g = g
        with og.sim.render_on_step(False):   # as eval's render-on-demand; proprio and states stay live
            for _ in range(n):
                _, prog, _, _, _ = self.env.step(np.concatenate([self.q, [self.g]]))
                self.steps += 1
                self.best = max(self.best, float(prog))

    def stages(self):
        return dict(self.env.task_progression or {})

    def note(self, what, **kw):
        self.log.append({"what": what, "step": self.steps, "score": round(self.best, 3), **kw})

    # ------------------------------------------------------------------ planning
    def plan(self, G, ignore=(), constraint=None, timeout=4.0):
        import torch as th
        from scipy.spatial.transform import Rotation as R
        gt.build_world(self.cmg, self.robot, ignore=tuple(ignore))
        quat = R.from_matrix(G[:3, :3]).as_quat()
        kw = {}
        if constraint is not None:
            kw["motion_constraint"] = list(constraint)
        succ, paths = self.cmg.compute_trajectories(
            target_pos={"grasp_frame": th.tensor(G[:3, 3], dtype=th.float32)},
            target_quat={"grasp_frame": th.tensor(quat, dtype=th.float32)},
            is_local=False, skip_obstacle_update=True, max_attempts=8, timeout=timeout, **kw)
        if succ is None or not len(succ) or not bool(succ[0]):
            return None
        return np_(self.cmg.path_to_joint_trajectory(paths[0]))[:, self.arm_idx]

    def run(self, traj, g=None, slow=None, watch=None):
        """Execute a joint trajectory; `slow` caps the per-step joint change; `watch()` -> True stops early."""
        if traj is None:
            return False
        pts = [traj[0]]
        for a, b in zip(traj[:-1], traj[1:]):
            k = 1 if slow is None else max(1, int(np.ceil(np.abs(b - a).max() / slow)))
            pts += [a + (b - a) * (i / k) for i in range(1, k + 1)]
        for q in pts:
            self.step(q, g)
            if watch is not None and watch():
                return True
        return True

    def hold(self, n, g=None):
        self.step(None, g, n)

    def tcp_pose(self):
        E = gt.link_mat(self.robot, "panda_link8")
        T = E.copy()
        T[:3, 3] = (E @ np.r_[self.tcp, 1.0])[:3]
        return T

    def near(self, point, radius=0.06, exclude=()):
        """Scene objects whose AABB comes within `radius` of `point` (walls behind a switch, etc.)."""
        out = []
        for o in self.robot.scene.objects:
            if o is self.robot or o in exclude:
                continue
            lo, hi = (np_(v) for v in o.aabb)
            if np.linalg.norm(np.clip(point, lo, hi) - point) <= radius:
                out.append(o)
        return out

    # ------------------------------------------------------------------ grasping skills
    def grasp(self, obj, cand):
        """Pre-grasp, approach, close. Returns a failure string or None."""
        G = cand["G"]
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        pre = None
        for Gv in (G, G @ flip):
            P = Gv.copy()
            P[:3, 3] -= Gv[:3, 2] * gt.PREGRASP
            pre = self.plan(P)
            if pre is not None:
                G = Gv
                break
        if pre is None:
            return "PLAN_PREGRASP_FAIL"
        self.run(pre, gt.OPEN)
        app = self.plan(G, ignore=(obj,))
        if app is None:
            app = self.plan(G, ignore=(obj, *gt.supports_of(self.robot, obj)))
        if app is None:
            return "PLAN_APPROACH_FAIL"
        self.run(app, gt.OPEN)
        self.hold(25, gt.CLOSE)
        self.note("closed", gap=round(gt.finger_gap(self.robot, self.fingers), 4))
        return None

    def lift(self, obj, dz):
        T = self.tcp_pose()
        T[2, 3] += dz
        z0 = float(np_(obj.get_position_orientation()[0])[2])
        traj = self.plan(T, ignore=(obj, *gt.supports_of(self.robot, obj)))
        if traj is None:
            return None
        self.run(traj, gt.CLOSE)
        self.hold(15)
        return float(np_(obj.get_position_orientation()[0])[2]) - z0

    def place_over(self, obj, target, clearance):
        sync_stage(10)
        T = self.tcp_pose()
        olo, ohi = (np_(v) for v in obj.aabb)
        tlo, thi = (np_(v) for v in target.aabb)
        off_xy = (olo[:2] + ohi[:2]) / 2 - T[:2, 3]
        bottom_off = olo[2] - T[2, 3]
        drop = T.copy()
        drop[:2, 3] = (tlo[:2] + thi[:2]) / 2 - off_xy
        drop[2, 3] = thi[2] + clearance - bottom_off
        carry = drop.copy()
        carry[2, 3] = max(T[2, 3], drop[2, 3] + CARRY_HEIGHT)
        if self.run(self.plan(carry, ignore=(obj,)), gt.CLOSE) is False:
            return "PLAN_CARRY_FAIL"
        down = self.plan(drop, ignore=(obj,))
        if down is None:
            down = self.plan(drop, ignore=(obj, target))
        if down is None:
            return "PLAN_LOWER_FAIL"
        self.run(down, gt.CLOSE)
        self.hold(10)
        self.hold(30, gt.OPEN)
        up = self.tcp_pose()
        up[2, 3] += 0.10
        self.run(self.plan(up, ignore=(obj, target)), gt.OPEN)
        self.hold(20)
        return None

    def rotate_wrist(self):
        lo, hi = self.limits[6]
        q = self.q.copy()
        sign = 1.0 if hi - q[6] >= q[6] - lo else -1.0
        target = np.clip(q[6] + sign * ROTATE_ANGLE, lo + 0.05, hi - 0.05)
        for k in range(1, 61):
            qq = q.copy()
            qq[6] = q[6] + (target - q[6]) * k / 60
            self.step(qq, gt.CLOSE)
        self.hold(20)
        return float(target - q[6])

    def solve_grasping(self, task_type, args):
        env = self.env
        obj = env.main_objects[0]
        target = env.target_objects[0] if env.target_objects else None
        attempts = []
        for k in range(MAX_ATTEMPTS):
            self.new_episode()
            obj = env.main_objects[0]
            target = env.target_objects[0] if env.target_objects else None
            pieces = gt.world_pieces(obj)
            support_z = float(np_(obj.aabb[0])[2])
            # attempt 0: OmniGibson's AABB pose; then mesh-sampled grasps, sampled on THIS episode's pose
            # (REALM's reset can re-place the object) and taken in rank order
            if k == 0:
                cand = gt.baseline_grasp(pieces, support_z)
            else:
                sampled, stats, _ = gt.sample_grasps(pieces, support_z, n_keep=MAX_ATTEMPTS * 2)
                if not sampled:
                    attempts.append({"attempt": k, "result": "NO_CANDIDATES", "sampling": stats})
                    break
                cand = sampled[min(k - 1, len(sampled) - 1)]
            a = {"attempt": k, "baseline": bool(cand.get("baseline")), "width": round(cand["width"], 4),
                 "tilt_deg": round(cand["topdown_deg"], 1)}
            fail = self.grasp(obj, cand)
            if fail is None:
                dz = self.lift(obj, ROTATE_LIFT if task_type == "rotate" else (
                    PICK_LIFT if task_type == "pick" else 0.12))
                a["rise"] = None if dz is None else round(dz, 4)
                if dz is None:
                    fail = "PLAN_LIFT_FAIL"
                elif dz < (0.02 if task_type == "rotate" else 0.05):
                    fail = "NOT_HELD"
                elif task_type in ("put", "stack"):
                    fail = self.place_over(obj, target, PUT_CLEARANCE[task_type])
                elif task_type == "rotate":
                    a["wrist_turn_rad"] = round(self.rotate_wrist(), 3)
            a.update(result=fail or "DONE", score=round(self.best, 3), steps=self.steps,
                     stages=self.stages(), log=self.log)
            attempts.append(a)
            REC["attempts"] = attempts
            save()
            if self.best >= 1.0:
                break
        return attempts

    # ------------------------------------------------------------------ push (light switch)
    def solve_push(self, args):
        from omnigibson.object_states import ToggledOn
        attempts = []
        for k in range(2):
            self.new_episode()
            sw = self.env.main_objects[0]
            marker = np_(sw.states[ToggledOn].visual_marker.get_position_orientation()[0])
            pieces = gt.world_pieces(sw)
            verts = np.concatenate([p.vertices for p in pieces])
            B = mat(*sw.get_position_orientation())
            local = (np.linalg.inv(B) @ np.c_[verts, np.ones(len(verts))].T).T
            ext = local[:, :3].max(0) - local[:, :3].min(0) if len(local) else np.ones(3)
            n = B[:3, int(np.argmin(ext))]
            base = gt.link_mat(self.robot, "panda_link0")[:3, 3]
            if n @ (base - marker) < 0:
                n = -n                                  # outward: toward the robot
            a = -n
            y = np.cross([0, 0, 1.0], a)
            y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-3 else np.array([0, 1.0, 0])
            tip_off = self.closed_tip - self.tcp[2]     # fingertip beyond the TCP with the gripper closed
            rec = {"attempt": k, "marker": np.round(marker, 4).tolist(), "normal": np.round(n, 3).tolist(),
                   "closed_tip_beyond_tcp": round(float(tip_off), 4)}
            self.hold(20, gt.CLOSE)
            pre = gt.grasp_frame(marker + n * (0.08 + tip_off), y, a)
            if not self.run(self.plan(pre), gt.CLOSE):
                rec["result"] = "PLAN_PREPRESS_FAIL"
            else:
                rec["result"] = "NOT_TOGGLED"
                wall = self.near(marker, 0.06)
                for depth in PUSH_DEPTHS:
                    G = gt.grasp_frame(marker + n * (tip_off - depth), y, a)
                    traj = self.plan(G, ignore=(sw, *wall), constraint=[1, 1, 1, 1, 1, 0])
                    if traj is None:
                        traj = self.plan(G, ignore=(sw, *wall))
                    if traj is None:
                        rec["result"] = "PLAN_PRESS_FAIL"
                        break
                    self.run(traj, gt.CLOSE, slow=SLOW_JOINT_STEP * 2)
                    self.hold(12)
                    if sw.states[ToggledOn].get_value():
                        rec["result"] = "DONE"
                        rec["depth"] = depth
                        break
                self.run(self.plan(pre, ignore=(sw, *wall)), gt.CLOSE)
                self.hold(10)
            rec.update(score=round(self.best, 3), steps=self.steps, stages=self.stages(), log=self.log)
            attempts.append(rec)
            REC["attempts"] = attempts
            save()
            if self.best >= 1.0:
                break
        return attempts

    # ------------------------------------------------------------------ drawers
    def drawer_geometry(self):
        """Drawer link pieces, cabinet obstacle points, opening direction (world) and travel left."""
        from scipy.spatial.transform import Rotation as R
        env = self.env
        cab, joint = env.main_objects[0], env.mo_joint
        path2link = {l.prim_path: l for l in cab.links.values()}
        drawer_link = path2link[joint.body1]
        parent = path2link.get(joint.body0)
        q0 = joint.get_attribute("physics:localRot0")
        im = q0.GetImaginary()
        R0 = R.from_quat([im[0], im[1], im[2], q0.GetReal()]).as_matrix()
        unit = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[str(joint.axis)]
        Rp = gt.link_mat(self.robot if parent is None else cab, parent.name)[:3, :3] if parent is not None \
            else mat(*cab.get_position_orientation())[:3, :3]
        axis = Rp @ R0 @ np.array(unit, float)
        sync_stage(10)
        drawer_pieces = []
        for cm in drawer_link.collision_meshes.values():
            tm = gt.geom_trimesh(cm)
            tm.vertices = tm.vertices * np_(cm.get_world_scale())
            tm.apply_transform(mat(*cm.get_position_orientation()))
            drawer_pieces.append(tm)
        rest = [p for l in cab.links.values() if l is not drawer_link
                for p in self._link_pieces(l)]
        cabinet_pts = gt.surface_samples(rest, n_total=4000)[0] if rest else np.zeros((0, 3))
        base = gt.link_mat(self.robot, "panda_link0")[:3, 3]
        dverts = np.concatenate([p.vertices for p in drawer_pieces])
        toward_robot = float(axis @ (base - dverts.mean(0))) > 0
        q = float(np_(joint.get_state()[0])[0])
        return {"cabinet": cab, "joint": joint, "pieces": drawer_pieces, "cabinet_pts": cabinet_pts,
                "open_dir": axis, "q": q, "lower": float(joint.lower_limit), "upper": float(joint.upper_limit),
                "open_dir_toward_robot": toward_robot, "dverts": dverts}

    @staticmethod
    def _link_pieces(link):
        out = []
        for cm in link.collision_meshes.values():
            tm = gt.geom_trimesh(cm)
            tm.vertices = tm.vertices * np_(cm.get_world_scale())
            tm.apply_transform(mat(*cm.get_position_orientation()))
            out.append(tm)
        return out

    def openness(self):
        return float(self.env.get_mo_joint_openness_fraction())

    def solve_drawer(self, task_type, args):
        attempts = []
        for k in range(MAX_ATTEMPTS):
            self.new_episode()
            d = self.drawer_geometry()
            cab = d["cabinet"]
            o = d["open_dir"]
            front = float((d["dverts"] @ o).max())
            rec = {"attempt": k, "open_dir": np.round(o, 3).tolist(),
                   "open_dir_toward_robot": d["open_dir_toward_robot"],
                   "openness_start": round(self.openness(), 3), "joint_q": round(d["q"], 4),
                   "joint_range": [round(d["lower"], 4), round(d["upper"], 4)]}
            if task_type == "open_drawer":
                travel = d["upper"] - d["q"] + DRAWER_OVERSHOOT
                cands, stats, _ = gt.sample_grasps(
                    d["pieces"], None, n_keep=MAX_ATTEMPTS * 2, seed=k, approach_ref=-o, approach_max_deg=40,
                    contact_mask=lambda P: P @ o >= front - DRAWER_FRONT_BAND, obstacle_pts=d["cabinet_pts"])
                rec["handle_sampling"] = {kk: stats[kk] for kk in ("antipodal_pairs", "candidates", "kept")}
                if not cands:
                    rec.update(result="NO_HANDLE", score=round(self.best, 3), steps=self.steps)
                    attempts.append(rec)
                    break
                cand = cands[min(k, len(cands) - 1)]
                G = cand["G"]
                rec["handle_width"] = round(cand["width"], 4)
                P = G.copy()
                P[:3, 3] -= G[:3, 2] * gt.PREGRASP
                if not self.run(self.plan(P), gt.OPEN):
                    rec["result"] = "PLAN_PREGRASP_FAIL"
                else:
                    app = self.plan(G, ignore=(cab,))
                    if app is None:
                        rec["result"] = "PLAN_APPROACH_FAIL"
                    else:
                        self.run(app, gt.OPEN)
                        self.hold(25, gt.CLOSE)
                        pull = G.copy()
                        pull[:3, 3] += o * travel
                        traj = self.plan(pull, ignore=(cab,), constraint=[1, 1, 1, 1, 1, 0])
                        if traj is None:
                            traj = self.plan(pull, ignore=(cab,))
                        if traj is None:
                            rec["result"] = "PLAN_PULL_FAIL"
                        else:
                            self.run(traj, gt.CLOSE, slow=SLOW_JOINT_STEP, watch=lambda: self.openness() > 0.97)
                            self.hold(15)
                            rec["result"] = "DONE" if self.openness() > 0.95 else "PULL_STALLED"
                            self.hold(20, gt.OPEN)
                            back = self.tcp_pose()
                            back[:3, 3] -= back[:3, 2] * 0.08
                            self.run(self.plan(back, ignore=(cab,)), gt.OPEN)
            else:  # close_drawer: push the front face with a closed fist
                travel = d["q"] - d["lower"] + DRAWER_OVERSHOOT
                face = d["dverts"][(d["dverts"] @ o) >= front - 0.005]
                c = face.mean(0)
                a = -o
                y = np.cross([0, 0, 1.0], a)
                y = y / np.linalg.norm(y)
                tip_off = self.closed_tip - self.tcp[2]
                rec.update(face_center=np.round(c, 4).tolist(), closed_tip_beyond_tcp=round(float(tip_off), 4))
                self.hold(20, gt.CLOSE)
                pre = gt.grasp_frame(c + o * (tip_off + 0.06), y, a)
                if not self.run(self.plan(pre), gt.CLOSE):
                    rec["result"] = "PLAN_PREPUSH_FAIL"
                else:
                    end = gt.grasp_frame(c + o * tip_off - o * travel, y, a)
                    traj = self.plan(end, ignore=(cab,), constraint=[1, 1, 1, 1, 1, 0])
                    if traj is None:
                        traj = self.plan(end, ignore=(cab,))
                    if traj is None:
                        rec["result"] = "PLAN_PUSH_FAIL"
                    else:
                        self.run(traj, gt.CLOSE, slow=SLOW_JOINT_STEP, watch=lambda: self.openness() < 0.03)
                        self.hold(15)
                        rec["result"] = "DONE" if self.openness() < 0.05 else "PUSH_STALLED"
                        self.run(self.plan(pre, ignore=(cab,)), gt.CLOSE)
            rec.update(openness_end=round(self.openness(), 3), score=round(self.best, 3), steps=self.steps,
                       stages=self.stages(), log=self.log)
            attempts.append(rec)
            REC["attempts"] = attempts
            save()
            if self.best >= 1.0:
                break
        return attempts

    # ------------------------------------------------------------------ calibration
    def measure_closed_tip(self):
        """How far the fingertips reach along panda_link8's z with the gripper CLOSED."""
        import omnigibson as og
        q = np.asarray(self.env.reset_qpos[:7], float)
        with og.sim.render_on_step(False):
            for _ in range(25):
                self.env.omnigibson_env.step(np.concatenate([q, [gt.CLOSE]]))
        sync_stage(10)
        E = gt.link_mat(self.robot, "panda_link8")
        inv_e = np.linalg.inv(E)
        best = 0.0
        for f in self.fingers:
            for p in self._link_pieces(self.robot.links[f]):
                v = (inv_e @ np.c_[p.vertices, np.ones(len(p.vertices))].T).T
                best = max(best, float(v[:, 2].max()))
        self.closed_tip = best if best > 0 else float(self.tcp[2] + gt.PAD_BACKOFF)
        return self.closed_tip


def run_one(args) -> int:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.sim_config import set_sim_config
    REC["task"] = args.task_cfg_path
    REC["task_name"] = Path(args.task_cfg_path).parent.name
    gt.REC = REC  # grasp_test helpers record into this report
    gt.OUT = OUT
    t0 = time.time()
    try:
        set_sim_config(robot=args.robot)
        env = RealmEnvironmentDynamic(task_cfg_path=args.task_cfg_path, perturbations=["Default"],
                                      robot=args.robot, rendering_mode="rt", no_rendering=False)
        env.reset()
        gt.calibrate_gripper(env)
        REC["task_type"] = env.task_type
        REC["ladder"] = list((env.task_progression or {}).keys())
        REC["stages"]["S1_env"] = f"PASS ({time.time() - t0:.1f}s)"
        args_ns = argparse.Namespace(curobo_dir=args.curobo_dir)
        cmg, tcp = gt.s2_curobo(env, args_ns)
        solver = Solver(env, cmg, tcp)
        REC["closed_tip_in_eef_z"] = round(solver.measure_closed_tip(), 4)
        save()
        t1 = time.time()
        if env.task_type in ("pick", "put", "stack", "rotate"):
            att = solver.solve_grasping(env.task_type, args)
        elif env.task_type == "push":
            att = solver.solve_push(args)
        elif env.task_type in ("open_drawer", "close_drawer"):
            att = solver.solve_drawer(env.task_type, args)
        else:
            att = []
            REC["skipped"] = f"task_type {env.task_type} is not one of the 7 skills"
        REC["attempts"] = att
        REC["stages"]["S3_solve"] = f"PASS ({time.time() - t1:.1f}s)"
    except Exception as error:
        REC["stages"]["ERROR"] = f"{type(error).__name__}: {error}"
        REC["traceback"] = traceback.format_exc()
        print(REC["traceback"])
    att = REC.get("attempts") or []
    best = max(att, key=lambda a: (a.get("score", 0), -a.get("steps", 1e9)), default=None)
    REC["best_score"] = best.get("score") if best else None
    REC["best_steps"] = best.get("steps") if best else None
    REC["within_policy_budget"] = bool(best and best.get("score") == 1.0 and best.get("steps", 1e9) <= POLICY_BUDGET)
    REC["verdict"] = ("SKIPPED" if REC.get("skipped") else "ERROR" if "ERROR" in REC["stages"] else
                      "SOLVED" if REC["best_score"] == 1.0 else "PARTIAL" if (REC["best_score"] or 0) > 0 else "FAILED")
    save()
    short = {k: REC.get(k) for k in ("task_name", "task_type", "ladder", "stages", "gripper_calibration",
                                     "closed_tip_in_eef_z", "best_score", "best_steps", "within_policy_budget",
                                     "verdict")}
    short["attempts"] = [{k: v for k, v in a.items() if k != "log"} for a in att]
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
        if t in SKILLS:
            tasks.append((f"{args.family}/{cfg.parent.name}/default.yaml", t))
    if args.limit:
        tasks = tasks[: args.limit]
    rows = []
    for t, ttype in tasks:
        name = Path(t).parent.name
        print(f"--- {name} ({ttype})", flush=True)
        cmd = [sys.executable, "-u", __file__, "--task_cfg_path", t, "--out", str(OUT),
               "--curobo_dir", args.curobo_dir, "--robot", args.robot]
        try:
            subprocess.run(cmd, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"    timeout after {args.timeout}s")
        f = OUT / f"{name}.json"
        rec = json.loads(f.read_text()) if f.exists() else {}
        att = rec.get("attempts") or []
        rows.append({"task": name, "skill": ttype, "verdict": rec.get("verdict", "NO_REPORT"),
                     "score": rec.get("best_score"), "steps": rec.get("best_steps"),
                     "budget_ok": rec.get("within_policy_budget"), "attempts": len(att),
                     "last_result": att[-1].get("result") if att else None})
        (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("=" * 30 + " FAMILY SUMMARY " + "=" * 30)
    for r in rows:
        print(f"{r['verdict']:8s} score={r['score']}  steps={r['steps']} budget_ok={r['budget_ok']}  "
              f"attempts={r['attempts']} last={r['last_result']}  {r['task']} ({r['skill']})")
    solved = sum(r["verdict"] == "SOLVED" for r in rows)
    print(f"SOLVED {solved}/{len(rows)}  ->  {OUT / 'summary.json'}")
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
    ap.add_argument("--timeout", type=int, default=1500)
    args = ap.parse_args()
    OUT = Path(args.out).resolve()
    if args.family:
        return run_family(args)
    if not args.task_cfg_path:
        ap.error("give --task_cfg_path or --family")
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
