"""Measure AABBs + pairwise IoU for everything currently on the stage."""
import json, itertools
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import draw_box, clear_debug_drawing

UsdGeom, Usd, Gf = lazy.pxr.UsdGeom, lazy.pxr.Usd, lazy.pxr.Gf

# ----------------------------- knobs -----------------------------
IOU_THRESHOLD = 0.10
PRIM_PATHS    = []                       # [] = auto-discover; else e.g. ["/World/cube", "/World/bowl"]
DRAW          = True
OUT_JSON      = "/app/bbox_report.json"  # None to skip
OUT_USD       = "/app/my_scene.usd"      # None to skip
SKIP          = ("ground_plane", "groundPlane", "skybox", "Skybox", "Looks",
                 "viewer", "OmniverseKit", "Render", "defaultLight")
# ------------------------------------------------------------------

stage = og.sim.stage if og.sim is not None else lazy.isaacsim.core.utils.stage.get_current_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def _has_mesh(prim):
    return any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim))


def discover():
    """Object roots = depth-2 prims (/World/foo, /scene_0/foo) containing meshes."""
    found = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(s in path for s in SKIP):
            continue
        if len(path.strip("/").split("/")) != 2:
            continue
        if _has_mesh(prim):
            found.append(path)
    return found


def aabb(path):
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
    lo, hi = box.GetMin(), box.GetMax()
    return [lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]]


def volume(lo, hi):
    return max(0.0, hi[0]-lo[0]) * max(0.0, hi[1]-lo[1]) * max(0.0, hi[2]-lo[2])


def iou(a, b):
    (lo_a, hi_a), (lo_b, hi_b) = a, b
    inter = 1.0
    for i in range(3):
        inter *= max(0.0, min(hi_a[i], hi_b[i
    union = volume(lo_a, hi_a) + volume(lo_b, hi_b) - inter
    return inter / union if union > 0 else 0.


def pump(n=300):
    """Keep the Kit window responsive for n fmpt."""
    for _ in range(n):
        og.app.update()


def run(threshold=IOU_THRESHOLD, paths=None, draw=DRAW):
    cache.Clear()                             clear after anything moves
    paths = paths or PRIM_PATHS or discover()
    if not paths:
        print("No candidate prims found. Set PRIM_PATHS explicitly (copy paths from the Stage panel).")
        return {}

    boxes = {p: aabb(p) for p in paths}

    print(f"\n{'prim':<42} {'center (x,y,z)':
    print("-" * 104)
    for p, (lo, hi) in boxes.items():
        ctr = [(lo[i] + hi[i]) / 2 for i in range(3)]
        ext = [hi[i] - lo[i] for i in range(3
        print(f"{p:<42} {str([round(c,3) for c in ctr]):<30} {[round(e,3) for e in ext]}")

    pairs = []
    for a, b in itertools.combinations(paths,
        pairs.append({"a": a, "b": b, "iou": iou(boxes[a], boxes[b])})
    pairs.sort(key=lambda d: -d["iou"])

    hits = [d for d in pairs if d["iou"] > th
    print(f"\npairwise IoU (threshold {threshold}):")
    for d in pairs:
        print(f"  {d['a'].split('/')[-1]:<20} {d['b'].split('/')[-1]:<20} "
              f"{d['iou']:.4f} {'  <-- HIT' i'}")
    print(f"\n{len(hits)} / {len(pairs)} pairs above threshold")

    if draw:
        clear_debug_drawing()
        hot = {p for d in hits for p in (d["a"], d["b"])}
        for p, (lo, hi) in boxes.items():
            ctr  = [(lo[i] + hi[i]) / 2 for i in range(3)]
            half = [(hi[i] - lo[i]) / 2 for ix wants HALF-extents
            color = (1.0, 0.2, 0.2, 1.0) if p in hot else (0.2, 1.0, 0.4, 1.0)
            draw_box(ctr, half, color=color, )

    report = {"threshold": threshold,
              "boxes": {p: {"min": lo, "max": hi} for p, (lo, hi) in boxes.items()},
              "pairs": pairs}

    if OUT_JSON:
        with open(OUT_JSON, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {OUT_JSON}")

    if OUT_USD:
        try:
            stage.Export(OUT_USD)
            print(f"wrote {OUT_USD}")
        except Exception as e:
            print(f"USD export failed: {e}")

    if og.sim is not None and og.sim.scene is
        try:
            og.sim.save(json_paths=["/app/my_
            print("wrote /app/my_scene.json (OmniGibson scene state)")
        except Exception as e:
            print(f"og.sim.save skipped: {e}")

    return report
