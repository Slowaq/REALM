"""Place objects, measure AABB IoU, save — all driven by keypresses in the viewport.

Keys (click the viewport first so it has focus):
  B  measure bounding boxes + pairwise IoU, draw them
  P  print current object poses as a paste-ready OBJECTS list
  M  save scene JSON + IoU report
  X  quit cleanly
"""
import os, json, itertools
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.utils.ui_utils import KeyboardEventHandler, draw_box, clear_debug_drawing

# ----------------------------- knobs -----------------------------
OBJECTS = [
    dict(name="apple0", category="apple", model="agveuv", position=[0.00, 0.0, 1.0]),
    dict(name="apple1", category="apple", model="agveuv", position=[0.06, 0.0, 1.0]),
    dict(name="bowl0",  category="bowl",  model=None,     position=[0.40, 0.0, 1.0]),
]
IOU_THRESHOLD = 0.10
SCENE_JSON    = "/app/my_scene.json"
REPORT_JSON   = "/app/bbox_report.json"
LOAD_EXISTING = True          # reload SCENE_JSON if it exists, instead of OBJECTS
# ------------------------------------------------------------------

running = True


def build():
    og.launch()
    if LOAD_EXISTING and os.path.exists(SCENE_JSON):
        og.clear()
        og.sim.restore(scene_files=[SCENE_JSON])
        print(f"restored {SCENE_JSON}")
    else:
        og.sim.import_scene(Scene(use_floor_plane=True))
        for spec in OBJECTS:
            obj = DatasetObject(name=spec["name"], category=spec["category"], model=spec["model"])
            og.sim.scene.add_object(obj)
            obj.set_position_orientation(position=spec["position"])
        print(f"built scene with {len(OBJECTS)} objects")
    og.sim.play()
    for _ in range(60):
        og.sim.step()


def measure(threshold=IOU_THRESHOLD, draw=True):
    objs = list(og.sim.scene.objects)
    boxes = {}
    for o in objs:
        lo, hi = o.aabb
        boxes[o.name] = (lo, hi)

    print(f"\n{'object':<16} {'center':<32} extent")
    print("-" * 80)
    for name, (lo, hi) in boxes.items():
        ctr = ((lo + hi) / 2).tolist()
        ext = (hi - lo).tolist()
        print(f"{name:<16} {str([round(c,3) for c in ctr]):<32} {[round(e,3) for e in ext]}")

    pairs = []
    for a, b in itertools.combinations(boxes, 2):
        lo_a, hi_a = boxes[a]
        lo_b, hi_b = boxes[b]
        overlap = (lazy.torch.minimum(hi_a, hi_b) - lazy.torch.maximum(lo_a, lo_b)).clamp(min=0.0)
        inter = float(overlap.prod())
        union = float((hi_a - lo_a).prod()) + float((hi_b - lo_b).prod()) - inter
        pairs.append({"a": a, "b": b, "iou": inter / union if union > 0 else 0.0})
    pairs.sort(key=lambda d: -d["iou"])

    hits = [d for d in pairs if d["iou"] > threshold]
    print(f"\npairwise IoU (threshold {threshold}):")
    for d in pairs:
        print(f"  {d['a']:<16} {d['b']:<16} {d['iou']:.4f}{'   <-- HIT' if d['iou'] > threshold else ''}")
    print(f"{len(hits)} / {len(pairs)} pairs above threshold")

    if draw:
        clear_debug_drawing()
        hot = {n for d in hits for n in (d["a"], d["b"])}
        for name, (lo, hi) in boxes.items():
            ctr  = ((lo + hi) / 2).tolist()
            half = ((hi - lo) / 2).tolist()            # draw_box wants HALF-extents
            color = (1.0, 0.2, 0.2, 1.0) if name in hot else (0.2, 1.0, 0.4, 1.0)
            draw_box(ctr, half, color=color, size=2.0 if name in hot else 1.0)

    return {"threshold": threshold, "pairs": pairs,
            "boxes": {n: {"min": lo.tolist(), "max": hi.tolist()} for n, (lo, hi) in boxes.items()}}


def print_poses():
    print("\nOBJECTS = [")
    for o in og.sim.scene.objects:
        pos, orn = o.get_position_orientation()
        print(f"    dict(name={o.name!r}, category={o.category!r}, model={getattr(o, 'model', None)!r}, "
              f"position={[round(float(v), 4) for v in pos]}, orientation={[round(float(v), 4) for v in orn]}),")
    print("]")


def save():
    og.sim.save(json_paths=[SCENE_JSON])
    print(f"wrote {SCENE_JSON}")
    with open(REPORT_JSON, "w") as f:
        json.dump(measure(draw=False), f, indent=2)
    print(f"wrote {REPORT_JSON}")


def quit_tool():
    global running
    running = False


def main():
    global running
    build()
    og.sim.enable_viewer_camera_teleoperation()

    K = lazy.carb.input.KeyboardInput
    KeyboardEventHandler.add_keyboard_callback(K.B, measure)
    KeyboardEventHandler.add_keyboard_callback(K.P, print_poses)
    KeyboardEventHandler.add_keyboard_callback(K.M, save)
    KeyboardEventHandler.add_keyboard_callback(K.X, quit_tool)

    print("\n" + "=" * 70)
    print("READY.  Click the viewport, then:")
    print("  shift + left-drag : move an object")
    print("  B : measure bounding boxes + IoU      P : print poses")
    print("  M : save scene + report               X : quit")
    print("=" * 70 + "\n")

    while running:
        og.sim.step()

    save()
    og.shutdown()


if __name__ == "__main__":
    main()
