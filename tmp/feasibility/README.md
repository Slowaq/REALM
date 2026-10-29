# tmp/feasibility: privileged planning checks (one-off)

These scripts check whether REALM tasks can be solved with privileged access: object poses known, a
classical planner, no learned policy. They are one-off tests. They live in `tmp/` (gitignored and
force-added) and are not meant to ship. Drop the commits before the PR.

Run everything on the GPU machine, from the repo root, through the container:

```
./scripts/run_apptainer.sh python -u tmp/feasibility/<script>.py ...
```

| Script | Where | Time | What it answers |
|---|---|---|---|
| `check_curobo.py` | container | ~1 min | Does cuRobo run on this GPU at all? |
| `grasp_width_audit.py` | host | seconds | Which main objects fit the 2F-85 jaw by bounding box? |
| `droid_curobo.py` | container | ~5 min | Builds and verifies a cuRobo model of DROID |
| `grasp_test.py` | container | ~5 min per task | Can DROID physically grasp and lift each main object? |

Every container script writes a JSON report and prints a `VERDICT` line. Isaac's exit code is not
trusted (it exits 0 on exceptions and can segfault at teardown after a pass), so read the JSON, not `$?`.

---

## cuRobo in this stack

### What is installed
- **Package:** the base image `stanfordvl/behavior:3.9.1` is built with BEHAVIOR-1K's `--primitives`
  flag. That installs `nvidia-curobo` from `StanfordVL/curobo@78612f4` into the `behavior` conda env.
- **Precompiled for one arch list:** `TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;10.0;12.0+PTX'`. The CUDA
  toolkit is removed afterwards, so the kernels cannot be rebuilt in the container.
  - Works: RTX 20/30/40/50, A100, L40.
  - Fails with "no kernel image is available": V100 (7.0), H100/H200 (9.0).
  - elmo's RTX 5090 (12.0) works.
- **OmniGibson's wrapper:** `omnigibson.action_primitives.curobo.CuRoboMotionGenerator`. It reads
  the robot's kinematics from its USD (`use_usd_kinematics=True`). It refreshes locked joints
  (the gripper) from the live sim before every plan. It converts world-frame targets into the robot
  base frame.

Smoke test:

```
./scripts/run_apptainer.sh python -u tmp/feasibility/check_curobo.py --out tmp/feasibility/curobo.json
```

This prints the GPU and its compute capability, then runs four levels:
- L0: imports
- L1: CUDA forward-kinematics kernels
- L2: a Franka MotionGen plan around a box
- L3: whether DROID has a cuRobo config

### The DROID model (`droid_curobo.py`)
REALM's DROID (Panda + Robotiq 2F-85, `realm/robots/panda_robotiq/droid.usd`) ships no cuRobo
config: there is no `curobo_path` in `realm/robots/definitions/*`. `droid_curobo.py` builds one from
the robot as the simulator loads it, then checks it against the simulator.

```
./scripts/run_apptainer.sh python -u tmp/feasibility/droid_curobo.py \
    --task_cfg_path REALM_DROID10/pick_spoon/default.yaml --out tmp/feasibility/droid_curobo
```

Outputs in `tmp/feasibility/droid_curobo/`:
- `droid_curobo.yml`: the cuRobo robot config. All paths in it are absolute (`/app/...`).
- `droid_curobo.usda`: a flattened copy of `droid.usd`, which cuRobo reads for kinematics.
- `report.json`: every measurement below.

What goes into the yml:

| Field | Value | Why |
|---|---|---|
| `usd_robot_root` | `/panda` | The robot's default prim; all joints live below it. |
| `base_link` | `panda_link0` | |
| `ee_link` | `grasp_frame` | Extra link on `panda_link8` at z = 0.1606 m, 2 cm back from the fingertip (0.1806 m), between the pads. Same orientation as `panda_link8`: z = approach, y = closing axis. |
| `usd_flip_joints` | every Panda joint → `Z` | See gotcha 2. |
| `lock_joints` | all six gripper joints (null) | OmniGibson fills them from the sim on every plan. Open or closed, the finger spheres sit where the fingers really are. |
| `collision_spheres` | ~150 spheres, fitted to DROID's own collision meshes | See gotcha 3. |
| `extra_links.attached_object` | 4 placeholder spheres on `grasp_frame` | Reserved for planning with an object attached. |
| `self_collision_ignore` | link pairs ≤ 2 joints apart, and every pair inside the hand (`panda_link7` and below) | |

Verified on elmo, 2026-10-29 (`DROID_CUROBO_OK`):
- cuRobo FK equals sim link poses to rounding (< 5 µm, < 1e-4 rad). This holds at reset and with each
  arm joint moved 0.3 rad.
- No self or world collision at reset.
- Plans to 12 cm above the spoon and 15 cm to the side take 0.3–0.4 s with CUDA graphs on. Executed
  through REALM's `CustomJointController`, the max tracking error is 0.035 rad, the final TCP error
  is ~0, and no object moves.
- Spheres cover ≥ 89% of every link's surface with gaps ≤ 1.7 cm. The fixed base, `panda_link0`, is
  the lowest; every other link is ≥ 91%.

### Using the model

```python
import numpy as np, torch as th, omnigibson as og
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator

D = "/app/tmp/feasibility/droid_curobo"
cmg = CuRoboMotionGenerator(env.robot, robot_cfg_path=f"{D}/droid_curobo.yml",
                            robot_usd_path=f"{D}/droid_curobo.usda",
                            use_default_embodiment_only=True, batch_size=2)
# load obstacles yourself (see gotcha 5), then:
ok, paths = cmg.compute_trajectories(
    target_pos={"grasp_frame": th.tensor(pos_world)},         # (3,)
    target_quat={"grasp_frame": th.tensor(quat_world_xyzw)},  # (4,) xyzw
    is_local=False, skip_obstacle_update=True, max_attempts=10, timeout=5.0)
arm_idx = [cmg.robot_joint_names.index(j) for j in env.robot.arm_joint_names[env.robot.default_arm]]
traj = cmg.path_to_joint_trajectory(paths[0]).cpu().numpy()[:, arm_idx]   # one row per env step
with og.sim.render_on_step(False):
    for q in traj:
        env.omnigibson_env.step(np.r_[q, -1.0])  # gripper on DROID: -1 open, +1 close
```

The interpolation dt is the sim step (1/15 s for DROID), so a trajectory has one waypoint per
`env.step`. REALM's arm action is 7 absolute joint positions followed by 1 binary gripper value.

### Gotchas, all hit while building this
1. **Use absolute paths.** cuRobo joins a relative `usd_path` onto its own `content/assets` folder.
2. **cuRobo's USD parser ignores the joint frame's rotation (`localRot1`).** It places a child link
   at `localPose0 · localPose1⁻¹`, then rotates it about the joint's axis letter in the child frame.
   - Isaac's URDF importer gives every Panda joint axis `X` plus a rotated joint frame, so without a
     fix cuRobo turns each joint about the wrong axis: FK was off by up to 69 cm and ~180°.
   - Fix: `usd_flip_joints[j] = dominant axis of R(localRot1) · e_axis`. A negative direction goes in
     `usd_flip_joint_limits`, which flips the joint's sign.
   - For DROID this is `X → +Z` on all seven arm joints. It matches what cuRobo's own `franka.yml`
     does by hand.
3. **Keep spheres to ~150.**
   - At ~280 spheres, cuRobo switches to its slow self-collision kernel ("Self Collision checks are
     greater than 16384"), and OmniGibson's warmup then fails `assert opt.cu_opt_init`.
   - `droid_curobo.py` retries with `use_cuda_graph=False` if that assert fires. That works, but
     planning is slower.
   - cuRobo's stock Franka spheres do **not** fit DROID's meshes (2% coverage on link0, 41% on link6),
     so the spheres are fitted. The fit is greedy and coverage-driven. A sphere may overhang the mesh
     by up to 12 mm on big links and 4 mm on gripper parts.
4. **Use `batch_size=2`.** That's OmniGibson's default; `batch_size=1` is not a path it exercises.
5. **Don't use `CuRoboMotionGenerator.update_obstacles()` on REALM scenes.** It asserts that every
   collision geom is a `Mesh`, and it loads the whole house. The scripts build the world themselves:
   - only objects within 1.5 m of the robot base
   - primitive shapes converted to meshes
   - at most 1800 meshes
   - then `cmg.mg[DEFAULT].update_world(...)`, and plan with `skip_obstacle_update=True`.
6. **The gripper sign on DROID: −1 opens, +1 closes.** This is measured, not taken from the docs:
   the gap between the inner finger links is ~11.6 cm after −1 and ~3.3 cm after +1. OmniGibson's
   controller docstring says "grasp is negative", but with the Robotiq's joint direction and
   `inverted: false`, the effect on DROID is the reverse. REALM's `settle()` holds −1, so after a
   settle the gripper is open. `grasp_test.py` re-measures the sign at start-up
   (`gripper_calibration` in its JSON), so a change in the robot config cannot silently break it.
7. **The API differs from upstream cuRobo docs.** In this fork, `MotionGen.compute_kinematics()`
   returns `ee_pos_seq` / `ee_quat_seq`, not `ee_quaternion`.
8. **The scene load dominates run time** (200–325 s for `pick_spoon`). Everything cuRobo does here
   takes well under a minute.

### Not usable as-is: OmniGibson's starter primitives
`StarterSemanticActionPrimitives._grasp` refuses unless `grasping_mode` is "sticky" or "assisted".
REALM's DROID is "physical". Its grasp pose generator (`get_grasp_poses_for_object_sticky`) gives one
top-down pose at the AABB top-centre. BEHAVIOR-1K assets carry **no grasp annotations**, so
`grasp_test.py` samples grasps on the collision mesh instead.

---

## Grasp test (`grasp_test.py`)

For each grasped main object (task types pick, put, rotate, stack, pour, pour_proxy) it runs:

1. **Candidates.** Antipodal grasps sampled on the object's collision mesh:
   - contact normals within 20° of the closing axis
   - width ≤ 80 mm (85 mm stroke minus 5 mm margin)
   - approach within 70° of straight down
   - open fingers and palm not intersecting the object; every corner of both open fingers above the
     support surface
   - ranked by verticality, centrality and width margin
   - plus OmniGibson's AABB top-centre pose as a baseline
2. **Execute** each candidate in turn:
   - plan to a pre-grasp 10 cm back with the object as an obstacle
   - approach with the object removed from the world
   - close the gripper, plan a 10 cm lift, hold. The open/close sign is measured at start-up.
3. **Judge.** Success means the object rose ≥ 7 cm and stayed within 2 cm of the TCP. The report also
   records OmniGibson's `robot.is_grasping(candidate_obj=...)`.
4. **Reset and retry.** After a failure the task resets and the next candidate runs. The baseline gets
   one attempt; sampled candidates get up to `--max_exec`.

```
# one task
./scripts/run_apptainer.sh python -u tmp/feasibility/grasp_test.py \
    --task_cfg_path REALM_DROID10/pick_spoon/default.yaml
# every grasping task in a family, one child process per task (one simulator per process)
./scripts/run_apptainer.sh python -u tmp/feasibility/grasp_test.py --family REALM_DROID10
```

It needs `tmp/feasibility/droid_curobo/droid_curobo.yml` from `droid_curobo.py`. Results go to
`tmp/feasibility/grasp/<task>.json` and `tmp/feasibility/grasp/summary.json`.
