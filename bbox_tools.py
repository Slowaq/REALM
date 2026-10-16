"""Axis-aligned bounding boxes, pairwise intersection/IoU, and debug drawing for OmniGibson objects.

Usage (inside the container, `conda activate behavior`, cwd /app):
    import omnigibson as og
    env = og.Environment(configs=...)
    from bbox_tools import *
    report = bbox_report()               # all objects in the scene, prints a table, draws boxes
    inter, iou = intersection(obj_a, obj_b)
    show(300)                            # keep the window responsive for 300 frames
"""
import itertools

import torch as th

import omnigibson as og
from omnigibson.utils.ui_utils import clear_debug_drawing, draw_box

SKIP_CATEGORIES = {"floors", "walls", "ceilings", "background", "driveway", "lawn"}
GREEN = (0.2, 1.0, 0.4, 1.0)
RED = (1.0, 0.2, 0.2, 1.0)


def aabb(obj):
    """(lo, hi) world-frame AABB as two 3-element tensors (computed from collision meshes)."""
    lo, hi = obj.aabb
    return lo.float(), hi.float()


def volume(lo, hi):
    return th.clamp(hi - lo, min=0).prod().item()


def intersection(a, b):
    """Return (intersection_volume, IoU) of two objects' (or two (lo, hi) tuples') AABBs."""
    lo_a, hi_a = a if isinstance(a, tuple) else aabb(a)
    lo_b, hi_b = b if isinstance(b, tuple) else aabb(b)
    inter = volume(th.maximum(lo_a, lo_b), th.minimum(hi_a, hi_b))
    union = volume(lo_a, hi_a) + volume(lo_b, hi_b) - inter
    return inter, (inter / union if union > 0 else 0.0)


def scene_objects(include_skipped=False):
    objs = og.sim.scenes[0].objects
    return [o for o in objs if include_skipped or o.category not in SKIP_CATEGORIES]


def draw(objs, highlight=(), size=2.0):
    """Draw wireframe AABBs; objects in `highlight` are red, others green."""
    clear_debug_drawing()
    for o in objs:
        lo, hi = aabb(o)
        draw_box(((lo + hi) / 2).tolist(), ((hi - lo) / 2).tolist(), color=RED if o in highlight else GREEN, size=size)


def bbox_report(objs=None, threshold=0.0, do_draw=True):
    """Print every object's AABB and all pairwise intersections; returns a list of dicts sorted by IoU."""
    objs = objs or scene_objects()
    boxes = {o: aabb(o) for o in objs}

    print(f"\n{'object':<35} {'center':<28} extent")
    for o, (lo, hi) in boxes.items():
        print(f"{o.name:<35} {str([round(v, 3) for v in ((lo + hi) / 2).tolist()]):<28} "
              f"{[round(v, 3) for v in (hi - lo).tolist()]}")

    pairs = []
    for a, b in itertools.combinations(objs, 2):
        inter, iou = intersection(boxes[a], boxes[b])
        if inter > 0:
            pairs.append({"a": a.name, "b": b.name, "intersection_m3": inter, "iou": iou})
    pairs.sort(key=lambda d: -d["iou"])

    print(f"\noverlapping pairs ({len(pairs)}):")
    for d in pairs:
        flag = "  <-- above threshold" if d["iou"] > threshold else ""
        print(f"  {d['a']:<25} {d['b']:<25} inter={d['intersection_m3']:.5f} m^3  IoU={d['iou']:.4f}{flag}")

    if do_draw:
        hot = {n for d in pairs if d["iou"] > threshold for n in (d["a"], d["b"])}
        draw(objs, highlight=[o for o in objs if o.name in hot])
    return pairs


def show(n=300):
    """Render n frames so the window stays responsive and the boxes are visible."""
    for _ in range(n):
        og.sim.step() if og.sim.is_playing() else og.sim.render()


# ---------------------------------------------------------------------------
# Raw USD prims (things added by hand in the GUI: Create menu, drag-and-drop)
# ---------------------------------------------------------------------------
import omnigibson.lazy as lazy

USD_SKIP = ("groundPlane", "ground_plane", "Skybox", "skybox", "Looks", "OmniverseKit", "Render", "defaultLight", "viewer", "physicsScene")


def _stage():
    return og.sim.stage


def usd_prims(roots=("/", "/World")):
    """Children of `roots` that contain geometry (the objects you placed by hand).

    Kit drops dragged-in assets at the stage root (/mug), while OmniGibson puts objects under /World,
    so both are searched by default.
    """
    UsdGeom, Usd = lazy.pxr.UsdGeom, lazy.pxr.Usd
    out = []
    for root in roots:
        root_prim = _stage().GetPrimAtPath(root)
        if not root_prim.IsValid():
            continue
        for child in root_prim.GetChildren():
            path = str(child.GetPath())
            if path in roots or any(s in path for s in USD_SKIP):
                continue
            if any(p.IsA(UsdGeom.Gprim) for p in Usd.PrimRange(child)):
                out.append(path)
    return out


def usd_aabb(path):
    """(lo, hi) world-frame AABB of a USD prim (from its render geometry)."""
    UsdGeom, Usd = lazy.pxr.UsdGeom, lazy.pxr.Usd
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(_stage().GetPrimAtPath(path)).ComputeAlignedBox()
    return th.tensor(list(box.GetMin()), dtype=th.float32), th.tensor(list(box.GetMax()), dtype=th.float32)


def usd_report(paths=None, threshold=0.0, do_draw=True, roots=("/", "/World")):
    """Like bbox_report(), but for raw USD prim paths. Defaults to everything under `root`."""
    paths = paths or usd_prims(roots)
    if not paths:
        print(f"No geometry found under {roots}. Pass paths=[...] copied from the Stage panel.")
        return []
    boxes = {p: usd_aabb(p) for p in paths}

    print(f"\n{'prim':<40} {'center':<28} extent")
    for p, (lo, hi) in boxes.items():
        print(f"{p:<40} {str([round(v, 3) for v in ((lo + hi) / 2).tolist()]):<28} "
              f"{[round(v, 3) for v in (hi - lo).tolist()]}")

    pairs = []
    for a, b in itertools.combinations(paths, 2):
        inter, iou = intersection(boxes[a], boxes[b])
        if inter > 0:
            pairs.append({"a": a, "b": b, "intersection_m3": inter, "iou": iou})
    pairs.sort(key=lambda d: -d["iou"])

    print(f"\noverlapping pairs ({len(pairs)}):")
    for d in pairs:
        flag = "  <-- above threshold" if d["iou"] > threshold else ""
        print(f"  {d['a']:<30} {d['b']:<30} inter={d['intersection_m3']:.5f} m^3  IoU={d['iou']:.4f}{flag}")

    if do_draw:
        clear_debug_drawing()
        hot = {n for d in pairs if d["iou"] > threshold for n in (d["a"], d["b"])}
        for p, (lo, hi) in boxes.items():
            draw_box(((lo + hi) / 2).tolist(), ((hi - lo) / 2).tolist(), color=RED if p in hot else GREEN, size=2.0)
    return pairs


def save_stage(path="/app/my_scene.usd"):
    """Save the hand-built stage so it survives a restart (re-open later with og.sim.stage / File > Open)."""
    _stage().Export(path)
    print(f"saved {path}  (on the host: ~/sedlam/REALM/{path.split('/app/')[-1]})")
