"""cuRobo smoke test. RUNS IN THE CONTAINER (GPU). Takes about a minute.

    ./scripts/run_apptainer.sh python -u tmp/feasibility/check_curobo.py --out tmp/feasibility/curobo.json
    # or, inside `source ./scripts/run_docker.sh`:
    python -u tmp/feasibility/check_curobo.py --out tmp/feasibility/curobo.json

The base image (stanfordvl/behavior:3.9.1) installs OmniGibson with the `primitives` extra, i.e.
nvidia-curobo from StanfordVL/curobo@78612f4, PRECOMPILED with
TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;10.0;12.0+PTX' and the CUDA toolkit removed afterwards, so the
kernels cannot be rebuilt at runtime. A GPU whose compute capability is not in that list (V100 = 7.0,
H100/H200 = 9.0) is expected to fail at L1 with "no kernel image is available".

Levels, each gated on the previous one:
  L0  imports: torch + CUDA device, curobo package, OmniGibson's CuRoboMotionGenerator wrapper
  L1  CUDA kernels: batched forward kinematics on curobo's bundled Franka model
  L2  motion generation: MotionGen warmup + plan_single around a box obstacle (Franka, panda hand)
  L3  REALM robot: does the DROID robot definition carry a curobo config? (it currently does not)

Exit codes are not trusted downstream; read the verdict in the JSON.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

COMPILED_ARCHS = {(7, 5), (8, 0), (8, 6), (8, 9), (10, 0), (12, 0)}


def level0(rec: dict) -> None:
    import torch
    rec["torch"] = torch.__version__
    rec["torch_cuda"] = torch.version.cuda
    rec["cuda_available"] = torch.cuda.is_available()
    if not rec["cuda_available"]:
        raise RuntimeError("torch.cuda.is_available() is False (no GPU visible in the container?)")
    cap = torch.cuda.get_device_capability(0)
    rec["gpu"] = torch.cuda.get_device_name(0)
    rec["compute_capability"] = f"{cap[0]}.{cap[1]}"
    rec["arch_in_compiled_list"] = cap in COMPILED_ARCHS or cap[0] >= 12
    import curobo
    rec["curobo_file"] = curobo.__file__
    rec["curobo_version"] = getattr(curobo, "__version__", "unknown")
    try:
        from omnigibson.action_primitives.curobo import CuRoboMotionGenerator  # noqa: F401
        rec["og_curobo_wrapper_import"] = "ok"
    except Exception as error:  # not fatal for L1/L2: they use curobo directly
        rec["og_curobo_wrapper_import"] = f"{type(error).__name__}: {error}"


def level1(rec: dict) -> None:
    import torch
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_path, join_path, load_yaml

    tensor_args = TensorDeviceType()
    cfg = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
    kin = CudaRobotModel(RobotConfig.from_dict(cfg, tensor_args).kinematics)
    q = torch.zeros((64, kin.get_dof()), **tensor_args.as_torch_dict())
    t0 = time.time()
    state = kin.get_state(q)
    torch.cuda.synchronize()
    rec["fk_ms"] = round((time.time() - t0) * 1000, 2)
    rec["fk_ee_pos_q0"] = [round(float(v), 4) for v in state.ee_position[0]]


def level2(rec: dict) -> None:
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    tensor_args = TensorDeviceType()
    world = WorldConfig(cuboid=[
        Cuboid(name="table", pose=[0.5, 0.0, -0.02, 1, 0, 0, 0], dims=[1.2, 1.2, 0.04]),
        Cuboid(name="wall", pose=[0.45, 0.0, 0.25, 1, 0, 0, 0], dims=[0.05, 0.30, 0.50]),
    ])
    cfg = MotionGenConfig.load_from_robot_config("franka.yml", world, tensor_args,
                                                 interpolation_dt=0.02, num_trajopt_seeds=4)
    mg = MotionGen(cfg)
    t0 = time.time()
    mg.warmup()
    rec["warmup_s"] = round(time.time() - t0, 1)
    start = JointState.from_position(
        tensor_args.to_device([[0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]]),
        joint_names=[f"panda_joint{i}" for i in range(1, 8)])
    # Keep the start orientation (from FK) so the test does not depend on panda_hand's frame convention.
    start_quat = mg.compute_kinematics(start).ee_quat_seq.reshape(-1, 4)[:1].clone()
    rec["plans"] = []
    # one goal in front of the wall, one behind it (forces the planner around the obstacle)
    for name, pos in [("open_space", [0.35, 0.30, 0.25]), ("behind_wall", [0.60, 0.0, 0.15])]:
        goal = Pose(position=tensor_args.to_device([pos]), quaternion=start_quat)
        t0 = time.time()
        res = mg.plan_single(start, goal)
        rec["plans"].append({
            "goal": name, "success": bool(res.success.item()),
            "status": str(getattr(res, "status", None)), "solve_s": round(time.time() - t0, 3),
            "waypoints": int(res.get_interpolated_plan().position.shape[0]) if res.success.item() else 0,
        })
    if not any(p["success"] for p in rec["plans"]):
        raise RuntimeError("MotionGen ran but no plan succeeded")


def level3(rec: dict) -> None:
    import yaml
    repo = Path(__file__).resolve().parents[2]
    found = {}
    for path in sorted((repo / "realm/robots/definitions").glob("*/*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        found[path.parent.name] = data.get("curobo_path") or None
    rec["realm_robot_curobo_paths"] = found
    rec["droid_has_curobo_cfg"] = bool(found.get("droid") or found.get("droid_mounted"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/feasibility/curobo.json")
    args = ap.parse_args()
    rec: dict = {"levels": {}}
    for name, fn in [("L0_imports", level0), ("L1_kernels", level1),
                     ("L2_motion_gen", level2), ("L3_realm_robot", level3)]:
        try:
            fn(rec)
            rec["levels"][name] = "PASS"
        except Exception as error:
            rec["levels"][name] = f"FAIL: {type(error).__name__}: {error}"
            rec[f"{name}_traceback"] = traceback.format_exc()
            if name != "L3_realm_robot":
                break
    core = [rec["levels"].get(k, "") for k in ("L0_imports", "L1_kernels", "L2_motion_gen")]
    rec["verdict"] = "CUROBO_OK" if all(v == "PASS" for v in core) else "CUROBO_BROKEN"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("gpu", "compute_capability", "arch_in_compiled_list",
                                              "levels", "plans", "droid_has_curobo_cfg", "verdict")},
                     indent=2))
    print(f"VERDICT {rec['verdict']}  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
