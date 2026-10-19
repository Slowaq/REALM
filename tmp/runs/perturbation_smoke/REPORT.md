# Perturbation smoke test: pi05 on 4 REALM_DROID100 tasks × 15 perturbations

**Setup:**
- **Code:** feat/agentic-task-generation at 77a843a.
- **Configs:** `realm/config/tasks/REALM_DROID100` restored from origin/pilot-configs, giving 104 folders.
- **Policy:** the openpi server `pi05_full_droid_finetune` with checkpoint `pi05_droid_jointpos` on port 8000. It is the same server process as the pi05_validation sweep.
- **Invocation:** the same as the validation sweep, apart from `--max_steps 100` and `--perturbation_id`. One rollout each:

  ```
  tmp/gpu_session/run_in_container.sh examples/02_evaluate.py --task_cfg_path REALM_DROID100/<task>/default.yaml \
    --model_type openpi --model_name pi05 --port 8000 --experiment_name droid100 \
    --log_dir /app/tmp/runs/perturbation_smoke --perturbation_id <1..15> --repeats 1 --max_steps 100 --no-render_on_demand
  ```
- **Time:** 2026-10-19, 17:27 to 21:43.
- **Files:**
  - `logs/run_all.sh` drives the sweep, and the per-run logs are in `logs/runs/<task>__<pert>.log`.
  - `logs/summarize.py` builds `summary.json` and `frames/`.
  - `frames/<task>__<pert>.jpg` holds frame 0 of each run's video. The left half is the external camera and the right half is the wrist camera.
  - The Default frame for each task comes from the pi05_validation sweep, for comparison.

**How I judged:**
- **Ran** means the run wrote `reports/<task>_<pert>.csv`. Exit codes were ignored.
- **Errors** are lines containing Traceback, Exception, Error, Segmentation or Killed. I excluded lines every clean run prints: the gymnasium "Casting input x to numpy array" warning and the GLFW "DISPLAY missing" warning.
- **Instruction** is the report's `instruction` column. That is `env.instruction`, the same field `rollout.py` sends to the client on every step.
- **Scene check:** only frame 0. I did not judge success.

## Summary

- **39 of 60 runs loaded and ran to 100 steps**, with no error lines.
- **21 runs crashed during `reset()` with a traceback**, before the first step:
  - **20 runs (S-PROP, S-LANG, S-MO, S-AFF, S-INT on all four tasks):** `KeyError: 'cached_semantic_perturbations'`. No REALM_DROID100 config has that key (0/104), while all 11 REALM_DROID10 configs have it. These five perturbations read the instruction variants from that key in `perturbations/semantic.py:13`. S-LANG falls back to it because the configs have no `synonyms` either.
  - **1 run (SB-VRB on remove_the_banana_from_the_box):** `Exception: Failed to get rigid body transforms from backend`.
    - The task has no target object, so SB-VRB spawns a `receiver` object (`sb_vrb.py:73`, then `_spawn_receiver`) and rescales it.
    - `backfill_object_cfgs` (`_helpers.py:64`) then reads the receiver's pose and fails.
    - Just before that, Isaac warned that the receiver's collision prim `/World/scene_0/receiver/base_link/collisions/noiyxv__base_link__collision__0` "was deleted while being used by a shape in a tensor view class". Then it logged "Simulation view object is invalidated and cannot be used again to call getTransforms".
    - The other three tasks have a target object, so they skip the spawn, and SB-VRB ran on all three.
- **Scenes:** no first frame shows a floating object, an interpenetrating object, or an object off the table. Every object a perturbation adds is visible in at least one camera.
  - **One perturbation broke the scene setup:** VB-POSE on remove_the_banana_from_the_box placed the banana *outside* the box, on the table behind it, and the box is empty.
  - **The instruction-changing perturbations sometimes produce odd instructions:**
    - SB-NOUN gives "Remove the storage box from the box".
    - SB-VRB gives "put the bowl into the bowl" and "stack the pot on top of the crayon".

## Table

Abbreviations: **ext** = external camera (left half of the frame), **wrist** = wrist camera.

### remove_the_banana_from_the_box_859416 (pick from a container, no target)
Default instruction: "Remove the banana from the box".

| perturbation | ran | errors | instruction received | first frame |
|---|---|---|---|---|
| V-AUG | yes | none | Remove the banana from the box | sane. The image is darker (the augmentation shows in the recorded frames). |
| V-VIEW | yes | none | Remove the banana from the box | sane. The ext camera has moved; the box and banana are visible. |
| V-SC | yes | none | Remove the banana from the box | sane. A mug, a dark bottle, a pen and a pencil have been added, all on the table and visible. |
| V-LIGHT | yes | none | Remove the banana from the box | sane |
| S-PROP | **no** | KeyError: 'cached_semantic_perturbations' | (none, crashed in reset) | (none) |
| S-LANG | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-MO | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-AFF | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-INT | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| B-HOBJ | yes | none | Remove the banana from the box | sane |
| SB-NOUN | yes | none | **Remove the storage box from the box** | The scene is sane and unchanged. The swapped noun is the container itself, so the instruction says to remove the box from the box. |
| SB-VRB | **no** | Exception: Failed to get rigid body transforms from backend (in the receiver spawn) | (none) | (none) |
| VB-POSE | yes | none | Remove the banana from the box | **Not sane for this task.** The banana lies on the table behind and left of the box, and the box is empty. Nothing floats. |
| VB-MOBJ | yes | none | Remove the banana from the box | sane. A different banana sits in the box. |
| VSB-NOBJ | yes | none | Remove the pop case from the box | sane. The banana is replaced by a dark-blue pop case inside the box, visible in ext and wrist. |

### put_the_garlic_in_the_bowl_7b721a (put)
Default instruction: "Put the garlic in the bowl".

| perturbation | ran | errors | instruction received | first frame |
|---|---|---|---|---|
| V-AUG | yes | none | Put the garlic in the bowl | sane. The colour and contrast have changed. |
| V-VIEW | yes | none | Put the garlic in the bowl | sane, but the ext camera is low and close. It shows the garlic, the jar and the bottle; **the bowl is out of the ext view** (it is still visible in wrist). |
| V-SC | yes | none | Put the garlic in the bowl | sane. A spray bottle, a white appliance, a red marker, a pen and small green items have been added, all on the table and visible. |
| V-LIGHT | yes | none | Put the garlic in the bowl | sane, brighter |
| S-PROP | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-LANG | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-MO | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-AFF | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-INT | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| B-HOBJ | yes | none | Put the garlic in the bowl | sane |
| SB-NOUN | yes | none | Put the jar of bath salt in the bowl | sane. The jar of bath salt is the existing distractor. |
| SB-VRB | yes | none | stack the garlic on top of the box of fruit | sane. The bowl is replaced by a box of fruit, which sits on the table and is visible in ext and wrist. |
| VB-POSE | yes | none | Put the garlic in the bowl | sane. The garlic has moved left of the bottle. |
| VB-MOBJ | yes | none | Put the garlic in the bowl | sane |
| VSB-NOBJ | yes | none | Put the box of fruit in the bowl | sane. The garlic is replaced by a small box of fruit on the table, visible. |

### put_the_bowl_on_the_plate_b4fe86 (stack)
Default instruction: "Put the bowl on the plate".

| perturbation | ran | errors | instruction received | first frame |
|---|---|---|---|---|
| V-AUG | yes | none | Put the bowl on the plate | sane |
| V-VIEW | yes | none | Put the bowl on the plate | sane, but the ext camera looks down at the robot's base platform. **The bowl, plate and drill are only at the top edge of the ext view**; wrist is unchanged. |
| V-SC | yes | none | Put the bowl on the plate | sane. A bottle, a marker, a red ball, a pink bottle and a glass have been added. They are on the table and visible; the glass is cut off at the ext frame's bottom-left edge. |
| V-LIGHT | yes | none | Put the bowl on the plate | sane |
| S-PROP | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-LANG | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-MO | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-AFF | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-INT | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| B-HOBJ | yes | none | Put the bowl on the plate | sane |
| SB-NOUN | yes | none | Put the jar of dill seed on the plate | sane. The jar of dill seed is the existing dark jar. |
| SB-VRB | yes | none | **put the bowl into the bowl** | The scene is sane: the plate is replaced by a white octagonal bowl on the table. The instruction is ambiguous, because there are now two bowls. |
| VB-POSE | yes | none | Put the bowl on the plate | sane. The bowl has moved toward the table's near edge and is on the table. |
| VB-MOBJ | yes | none | Put the bowl on the plate | sane. The bowl is replaced by a different (oval) bowl model. |
| VSB-NOBJ | yes | none | Put the scoop of ice cream on the plate | sane. The bowl is replaced by a small dark sphere (the scoop of ice cream) on the table, visible in ext and wrist. |

### rotate_the_pot_20a53e (rotate)
Default instruction: "Rotate the pot".

In the Default ext view the pot is already cut off at the bottom edge (known from render_review). That carries into every perturbation except V-VIEW.

| perturbation | ran | errors | instruction received | first frame |
|---|---|---|---|---|
| V-AUG | yes | none | Rotate the pot | sane |
| V-VIEW | yes | none | Rotate the pot | sane. The new ext view shows the pot, pepper and bottles in full. |
| V-SC | yes | none | Rotate the pot | sane. A green bottle, jars, a dark bottle and a stick have been added on the table. One added bottle lies on its side (visible in wrist). |
| V-LIGHT | yes | none | Rotate the pot | sane |
| S-PROP | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-LANG | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-MO | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-AFF | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| S-INT | **no** | KeyError: 'cached_semantic_perturbations' | (none) | (none) |
| B-HOBJ | yes | none | Rotate the pot | sane |
| SB-NOUN | yes | none | Rotate the shaker | sane. The shaker is the existing small dark bottle. |
| SB-VRB | yes | none | **stack the pot on top of the crayon** | The scene is sane: a crayon was added as the receiver, lying on the table and visible in ext and wrist. Stacking a pot on a crayon is not a feasible task. |
| VB-POSE | yes | none | Rotate the pot | sane. The pot has moved and is on the table. |
| VB-MOBJ | yes | none | Rotate the pot | sane |
| VSB-NOBJ | yes | none | Rotate the ruler | sane. The pot is replaced by a ruler, lying on the table, visible in wrist; it is not identifiable in ext, which is cut off at the bottom. |

## By perturbation

| perturbation | ran (of 4) | notes |
|---|---|---|
| V-AUG, V-LIGHT, B-HOBJ, VB-MOBJ | 4 | clean |
| V-VIEW | 4 | The new viewpoint drops the target (bowl) or most of the scene from the ext camera in 2 of 4 tasks. |
| V-SC | 4 | Clutter is added and visible every time. |
| S-PROP, S-LANG, S-MO, S-AFF, S-INT | **0** | `KeyError: 'cached_semantic_perturbations'`: the DROID100 configs have no cached semantic instructions. |
| SB-NOUN | 4 | The swapped noun can be the container itself ("storage box from the box"). |
| SB-VRB | **3** | It crashes on a task with no target (receiver spawn). The receiver can be the same category as the object ("bowl into the bowl") or an infeasible receiver ("pot on top of the crayon"). |
| VB-POSE | 4 | For a pick-from-container task it moved the banana out of the box. |
| VSB-NOBJ | 4 | The replacement object is always visible and named in the instruction. |
