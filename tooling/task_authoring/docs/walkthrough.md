# Walkthrough: one instruction, end to end

[← Back to overview](../README.md) · [How it works](how-it-works.md) · **Walkthrough** · [Usage](usage.md) · [Results](results.md) · [Reference](reference.md) · [Limitations](limitations.md)

This page follows **"Put the marker in the cup"**, one of the most common DROID instructions, through every step. All outputs below are real; long JSON is trimmed with `…`. You can reproduce it from the repository root in about a minute, with no GPU.

> Tip: to try this without touching the real family, add `--sessions tmp/docs_demo/sessions` after `agent.cli`, as these examples do. Sessions then go to a scratch folder.

```sh
C="python -m tooling.task_authoring.agent.cli --sessions tmp/docs_demo/sessions"
```

## 1. Open a session

```sh
$C start "Put the marker in the cup" --ranking-id docs-demo --rank 1
```

```json
{
  "task_id": "508c253cb1fa",
  "instruction": "Put the marker in the cup",
  "hint": {"task_type": "put", "nouns": ["marker", "mug"], "note": "a regex guess; you own the decision"},
  "catalog": {"source": "dataset", "categories": 2430},
  "next": "call 508c253cb1fa search_assets ... for each noun, then propose_layout"
}
```

The `task_id` is a hash of the instruction and the ranking id, so the same pair always gives the same id. The hint is only a guess; the generator makes the actual decision.

## 2. Look up real assets

```sh
$C call 508c253cb1fa search_assets '{"query":"marker","role":"main"}'
```

```json
{
  "category": "marker", "found": true, "alternatives": [],
  "candidates": [
    {"category": "marker", "model": "qemval", "bbox": [0.10029, 0.01261, 0.0126]},
    {"category": "marker", "model": "kjxpdi", "bbox": [0.10839, 0.01488, 0.01279]},
    …
  ]
}
```

```sh
$C call 508c253cb1fa search_assets '{"query":"cup","role":"target"}'
```

```json
{
  "category": "soda_cup", "found": true,
  "alternatives": ["teacup", "paper_cup", "coffee_cup", "cup_of_ranch", "cup_of_yogurt", "measuring_cup"],
  "candidates": [
    {"category": "soda_cup", "model": "gnzegv", "bbox": [0.08208, 0.08207, 0.15727]},
    …
  ],
  "note": "use `category` and `model` verbatim in propose_layout"
}
```

Each candidate is a real model with its natural size in metres. "Cup" is ambiguous, so the tool lists the other categories it could mean. The generator picks a 15.7 cm deep soda cup, deep enough to hold the 10.8 cm marker lengthwise.

A word with no asset gets a clear answer instead of a guess:

```json
{"query": "position", "found": false, "candidates": [],
 "note": "no indexed category matches 'position'. It does not exist. Do not invent it: search a synonym, or call report_ungroundable."}
```

## 3. Propose roles; the solver does the geometry

```sh
$C call 508c253cb1fa propose_layout '{
  "task_type": "put",
  "instruction": "Put the marker in the cup",
  "main":   {"name": "marker", "category": "marker",   "model": "kjxpdi"},
  "target": {"name": "cup",    "category": "soda_cup", "model": "gnzegv"},
  "decisions": ["cup grounded as soda_cup: deep enough to take the marker lengthwise"]}'
```

```json
{
  "ok": true,
  "audit": {
    "region": {"scene": "office_cubicles_left", "support": "Office_Desk", "width": 0.40, "depth": 0.60, "z": 0.9},
    "resized_assets": [],
    "receiver_capacity": {"task_type": "put", "lengthwise_insertion": true, "uniform_scale": 1.0, …},
    "distractors": [{"name": "distractor_bottle_of_protein_powder", …}, …]
  },
  "note": "geometry solved. Positions, orientations and scales are final -- do not restate them. Call validate_draft next."
}
```

The model named two objects and a task type. Everything else came from the solver: which table, where each object goes, whether anything must shrink (nothing here), which three distractors join the scene, and the cameras. The model sees an **audit** of what was decided, not numbers it could edit.

## 4. Validate

```sh
$C call 508c253cb1fa validate_draft
```

```json
{"ok": true, "error_count": 0, "findings": [], "warnings": [], "round": 1, "note": "clean: call submit_task"}
```

### A validation error and its fix

Suppose the generator kept the word "bowl" in the sentence but grounded the target as a cup. The validator catches it:

```json
{
  "ok": false, "error_count": 2,
  "findings": [
    {"code": "INSTRUCTION_CLOSURE", "severity": "error",
     "message": "instruction_target_to_replace 'cup' does not appear in the instruction"},
    {"code": "INSTRUCTION_PHANTOM_OBJECT", "severity": "error",
     "message": "the instruction names ['bowl'] but no such object is in the scene. Either add it as a role … or rewrite the instruction so it promises only objects the scene contains.",
     "path": "instruction",
     "fix": {"remove_or_ground_words": ["bowl"]}}
  ],
  "note": "apply each error's `fix` through the tool that owns it, then validate again"
}
```

The `fix` field says exactly what to change. A policy would otherwise be asked to put the marker in a bowl that doesn't exist, and every evaluation of that task would be meaningless.

## 5. Submit, or decline

```sh
$C call 508c253cb1fa submit_task '{"decisions": ["cup grounded as soda_cup gnzegv"]}'
```

When an instruction can't be represented honestly, the generator declines it. That is a correct outcome, not a failure:

```sh
$C call <task_id> report_ungroundable '{
  "reason": "The ring-stacking toy is not in the catalogue: ring is a 1.5 cm jewellery ring and stand is furniture.",
  "blocking_terms": ["ring", "stand"]}'
```

```json
{"status": "recorded", "note": "the instruction is dropped from the family"}
```

## 6. The exported task file

After `cli export`, the family version of this task lives in `realm/config/tasks/REALM_DROID100/put_the_marker_in_the_cup_d31114/default.yaml`. It landed on a different table than the demo above because the family run used a different ranking id, so the task id and its seed differ. Abridged, with comments added:

```yaml
task_type: put                         # selects REALM's scoring rubric: reach, grasp, lift, move, place into
instruction: Put the marker in the cup
instruction_obj_to_replace: marker     # REALM's language perturbations rewrite these words
instruction_target_to_replace: cup
supported_scenes:
  Wainscott_0_int: [Dining_Table]      # chosen from this task's seed

main_objects:
- type: DatasetObject
  name: marker
  category: marker
  model: kjxpdi
  bounding_box: [0.10839, 0.01488, 0.01279]     # natural size, metres
  orientation: [0.0, 0.0, 0.0, 1.0]             # upright, no yaw
  relative_bbox_position: [0.35, 0.42, -0.0636] # from the solver: x/y inside the table region, z = 1.5 cm above the measured surface

target_objects:
- {type: DatasetObject, name: cup, category: soda_cup, model: gnzegv, …}

distractors:                            # sampled clutter, one per product family
- {name: distractor_bottle_of_tea_leaves, …}
- {name: distractor_jar_of_spaghetti_sauce, …}
- {name: distractor_sponge, …}

camera_extrinsics: {cam1: {…}, cam2: {…}}   # a DROID camera pair that looks at the table

target_state: {rubric: put, predicate: inside, subject: marker, object: cup}   # documentation; REALM scores by task_type

provenance:
  task_id: d311145a2995
  ranking_id: droid-clusters-1chunks-v1
  original_instruction: Put the marker in the cup
  decisions:
  - cup grounded as soda_cup gnzegv (literally a cup, deep enough for lengthwise insertion)
```

The same task appears as the first panel of the [layout figure](results.md#what-the-scenes-look-like).

## How instructions get edited

88 of the 104 tasks changed their wording from the original DROID text. The reasons are always recorded in `decisions`:

| DROID original | Final instruction | Recorded reason |
|---|---|---|
| Pick up the white cup and move it to the left | Pick up the cup | "move it to the left" is a motion REALM cannot score; "white" dropped (texture) |
| Place the green marker in the grey bowl | Place the marker in the bowl | colour words dropped: mesh textures are fixed, so colour can't be guaranteed |
| Put one blue block from the box inside the white and black bowl | Take the blue block out of the box and put it in the bowl | the block is a primitive whose colour *is* set, so "blue" stays; the box becomes a source the block starts inside |
| Pick up the bottle from the table and put it in the grey box, place the grey box on top of the black box | Put the bottle in the box | shortened to the one stage a single rubric can score |

## Next steps

- Run it on your own instructions: [Usage](usage.md#generate-tasks-from-your-own-instructions)
- See how the whole family turned out: [Results](results.md)
