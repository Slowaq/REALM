"""Host-side grasp audit of task configs (no GPU, no sim). Seconds to run.

    python3 tmp/feasibility/grasp_width_audit.py --families REALM_DROID10 REALM_DROID100 \
        --out tmp/feasibility/grasp_width_audit.json

What it answers: does the grasp OmniGibson/BEHAVIOR-1K gives you for free fit the DROID gripper?
BEHAVIOR-1K assets carry no per-object grasp annotations. The only object grasp generator in
OmniGibson 3.9.1 (utils/grasping_planning_utils.get_grasp_poses_for_object_sticky) returns ONE
top-down pose at the AABB top-centre, and the starter primitives refuse to grasp unless the robot runs
"sticky" or "assisted" grasping. REALM's DROID runs "physical" grasping with a Robotiq 2F-85
(85 mm stroke), so that pose only works when the object's narrower horizontal extent fits the jaw.

Classes per grasped main object (world AABB from the authored bounding_box and orientation):
  PINCH_OK   narrower horizontal extent <= 80 mm (85 mm stroke - 5 mm margin), height >= 7 mm
  EDGE_ONLY  too wide to pinch across, but height <= 80 mm, so a rim/edge grasp (bowl wall, plate lip)
             is geometrically possible; needs a mesh-based grasp sampler, not the AABB pose
  TOO_WIDE   neither: only a part grasp (handle, lid knob) can work; needs a mesh/part sampler
  TOO_FLAT   height < 7 mm (same threshold as GRASP_INFEASIBLE)
  NO_GRASP   task type does not grasp (push, open_drawer, close_drawer, turn_faucet)
This is a necessary-condition screen on boxes, not a proof: PINCH_OK still has to be confirmed in sim.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

STROKE = 0.085
MARGIN = 0.005
MIN_HEIGHT = 0.007
GRASPING_TYPES = {"pick", "put", "rotate", "stack", "pour", "pour_proxy"}


def quat_xyzw_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_extents(bbox, quat) -> np.ndarray:
    return np.abs(quat_xyzw_to_mat(quat or [0, 0, 0, 1])) @ np.asarray(bbox, dtype=float)


def classify(ext: np.ndarray) -> str:
    narrow, height = float(min(ext[0], ext[1])), float(ext[2])
    if height < MIN_HEIGHT:
        return "TOO_FLAT"
    if narrow <= STROKE - MARGIN:
        return "PINCH_OK"
    if height <= STROKE - MARGIN:
        return "EDGE_ONLY"
    return "TOO_WIDE"


def audit(root: Path, families: list[str]) -> list[dict]:
    rows = []
    for fam in families:
        for cfg_path in sorted((root / fam).glob("*/default.yaml")):
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            ttype = str(cfg.get("task_type", "")).strip('"')
            for obj in cfg.get("main_objects") or []:
                row = {"family": fam, "task": cfg_path.parent.name, "task_type": ttype,
                       "object": obj.get("name"), "category": obj.get("category"),
                       "model": obj.get("model")}
                if ttype not in GRASPING_TYPES:
                    row["class"] = "NO_GRASP"
                elif not obj.get("bounding_box"):
                    row["class"] = "NO_BBOX"
                else:
                    ext = world_extents(obj["bounding_box"], obj.get("orientation"))
                    row["extents_m"] = [round(float(v), 4) for v in ext]
                    row["class"] = classify(ext)
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["REALM_DROID10", "REALM_DROID100"])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "realm/config/tasks"))
    ap.add_argument("--out", default="tmp/feasibility/grasp_width_audit.json")
    args = ap.parse_args()
    rows = audit(Path(args.root), args.families)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    for fam in args.families:
        fam_rows = [r for r in rows if r["family"] == fam]
        print(f"{fam}: {len(fam_rows)} main objects  {dict(Counter(r['class'] for r in fam_rows))}")
    for r in rows:
        if r["class"] not in ("PINCH_OK", "NO_GRASP"):
            print(f"  {r['class']:9s} {r['family']}/{r['task']}  {r['category']}  {r.get('extents_m')}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
