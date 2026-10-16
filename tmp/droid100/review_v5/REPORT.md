# render_review v5: REALM_DROID100 family run (104 tasks)

- **Code:** `feat/agentic-task-generation` at 1098632. It includes 9977373, which added the TIPPED, LEFT_CONTAINER and LEFT_SUPPORT checks.
- **Configs:** `realm/config/tasks/REALM_DROID100` was restored from `origin/pilot-configs`, giving 104 task folders.
- **Command:** `tmp/gpu_session/run_in_container.sh tooling/task_authoring/render_review.py --family REALM_DROID100 --out tmp/droid100/review_v5/droid100`, run with the default 60 settle steps. The log is `render.log`.
- **GPU:** I stopped the pi05 serve_policy server (13.4 GB) before the run, as you approved. The robometer server (9.8 GB) stayed up.
- **Run health:**
  - There were no tracebacks, out-of-memory or CUDA errors.
  - All 104 tasks wrote 6 renders each, with no render errors.
  - Each task logged the usual "Fatal Python error: Segmentation fault" at teardown. Exit codes were ignored.
- **How I judged:**
  - Verdicts come from the VERDICT lines and the JSONs.
  - The image judgement uses t060 (after the settle), with t000 for comparison and the wrist camera as a cross-check.

## Tally

| | PASS | FAIL |
|---|---|---|
| script verdict | 87 | 17 |
| **visual verdict (FAILs + the 8 named tasks checked)** | all 17 FAILs look fine | 0 |

**Every one of the 17 FAILs is a false positive.** One scene the script PASSes is visibly wrong: remove_the_marker_from_the_bowl_6facf8 (see the last section).

**Codes:** TIPPED ×9, LEFT_CONTAINER ×6, LEFT_SUPPORT ×2.

**Compared with the merged v4/v5 set** in `tmp/runs/render_review/droid100`:
- 6 of the 7 old-rule FAILs now PASS: the pruner ×3, mustard, mug and coconut oil. The chips task now fails on TIPPED instead.
- 15 tasks that passed before now FAIL under the new checks.

## Every FAIL

| task | finding(s) | image matches verdict? | what the image shows |
|---|---|---|---|
| put_the_ball_inside_the_bowl_c48915 | TIPPED ball 46°, DRIFT 0.027 | **no** | The tennis ball rolled a little and rests on the table beside the bowl. A sphere's rotation is not tipping. |
| remove_the_bag_of_chips_from_the_sink_427db2 | TIPPED bag_of_chips 96°, DRIFT 0.085 | **no** | The bag stands up against the faucet, inside the sink, in all three cameras. |
| remove_the_marker_from_the_cup_1fb5a0 | TIPPED marker 142°, DRIFT 0.022 | **no** | The marker leans inside the cup and still sticks out of it. |
| remove_the_marker_from_the_glass_1f96cb | TIPPED marker 42°, DRIFT 0.025 | **no** | The marker leans inside the glass. |
| remove_the_marker_from_the_mug_e57716 | TIPPED marker 37°, DRIFT 0.019 | **no** | The marker leans inside the mug. |
| remove_the_marker_from_the_pot_f89099 | TIPPED marker 68° | **no** | The marker lies inside the pot. |
| remove_the_pen_from_the_mug_e6c8fe | TIPPED pen 43°, DRIFT 0.029 | **no** | The pen leans inside the mug. |
| remove_the_pineapple_from_the_pot_3fb906 | TIPPED pineapple 37°, DRIFT 0.018 | **no** | The pineapple rests tilted inside the pot. |
| remove_the_spoon_from_the_basket_202542 | TIPPED spoon 45° (DRIFT bell pepper 0.012) | **no** | The spoon leans inside the basket. |
| remove_the_banana_from_the_box_859416 | LEFT_CONTAINER banana (gap 1.8 mm) | **no** | The banana lies on the box floor. |
| remove_the_bow_from_the_box_5d5118 | LEFT_CONTAINER bow (gap 1.9 mm) | **no** | The bow lies on the box floor. |
| remove_the_marker_from_the_box_3c06f6 | LEFT_CONTAINER marker (gap 1.8 mm) | **no** | The marker lies on the box floor. |
| take_the_blue_block_out_of_the_box_and_put_it_in_the_bowl_52e039 | LEFT_CONTAINER blue_block (gap 2.0 mm) | **no** | The block sits on the box floor. |
| take_the_soap_dispenser_out_of_the_container_5b8a87 | LEFT_CONTAINER soap_dispenser (gap 2.9 mm) | **no** | The dispenser stands upright in the container. |
| take_the_teddy_bear_out_of_the_box_d19562 | LEFT_CONTAINER teddy_bear (gap 1.4 mm) | **no** | The bear lies on the box floor. |
| remove_the_grey_block_from_the_tray_612012 | LEFT_SUPPORT grey_block | **no** | The block sits in the middle of the tray. |
| remove_the_yellow_object_from_the_plate_851e78 | LEFT_SUPPORT yellow_block | **no** | The block sits on the plate. |

The "gap" is the object's lower face minus the container's lower face.

### Why each check misfires

- **TIPPED** (> 35° from the start orientation) catches markers, pens and spoons that start standing upright in a cup and then lean against its rim. Leaning is their natural resting pose. It also catches the ball, whose rotation means nothing, and the chips bag, which flipped up but is still in the sink. None of the nine fell out of anything.
- **LEFT_CONTAINER, the through-floor branch.** An object is flagged when its lower face is less than `THROUGH_FLOOR_M` = 3 mm above the container's lower face.
  - All six objects are 1.4–2.9 mm above it: they rest on floors thinner than 3 mm. Every one is plainly inside its container in the images.
  - The v4 pen that really sank through its box ended 0.4 mm *below* the container's lower face. A threshold of ≤ 0 would separate the two cases.
- **LEFT_SUPPORT.** An object is flagged when its lower face is below the support's half-height.
  - That rule fails for rimmed supports, where the rim sets the box height but the object rests on the much lower floor:
    - Tray: 0.745 to 0.771, half-height 0.758. The block rests at 0.756.
    - Plate: 0.920 to 0.939, half-height 0.929. The block rests at 0.925.
  - Both blocks are visibly on their tray and plate.

These are observations only. I did not change any code.

## The 8 named tasks

| task | verdict | image matches? | main object visible (sensor0 / sensor1) |
|---|---|---|---|
| put_the_garlic_in_the_bowl_7b721a | PASS | yes | yes / yes |
| put_the_grapes_in_the_bowl_a731f9 | PASS | yes | partly, behind the bowl rim / yes |
| put_the_tape_in_the_bowl_be2b7f | PASS | yes | partly, behind the bowl / yes |
| put_the_cloth_in_the_basket_2ae4bf | PASS | yes | partly, behind the bottle / yes |
| put_the_plum_in_the_bowl_ab75a2 | PASS | yes | yes / yes |
| put_the_pen_in_the_bucket_9f1b9a | PASS | yes | yes / no, hidden behind the bucket and bottle |
| put_the_green_block_in_the_tray_d39aaa | PASS | yes | yes / yes |
| remove_the_banana_from_the_box_859416 | FAIL | **no**: false positive (above) | yes / yes |

In all of these tasks everything is at rest, and nothing floats or has fallen.

## Main object visible in neither external camera

**None** among the 24 tasks checked (the 17 FAILs plus the 8 named tasks; the banana is in both lists). Three tasks are visible in only one camera:
- put_the_pen_in_the_bucket: sensor0 only.
- remove_the_spoon_from_the_basket: sensor1 only, hidden by the basket wall in sensor0.
- remove_the_grey_block_from_the_tray: the block is small in sensor0 but visible.

## A real problem the script PASSes

**remove_the_marker_from_the_bowl_6facf8:** the coconut-oil bottle (a distractor) falls over during the settle and lies on its side, the same as in v4. 9977373 exempts distractors from TIPPED, so the scene now PASSes. I checked this task because it was the one real visual FAIL in the merged set. I did not look at the other 86 PASS images in this run.

## Bottom line

- The script says 87 PASS / 17 FAIL. By the images, all 17 FAILs are fine scenes. They come from three threshold issues: TIPPED on objects leaning in cups, the 3 mm through-floor margin, and the half-height support rule for rimmed trays and plates.
- The one known bad start state (the coconut-oil bottle falling over) passes because distractors are not checked for tipping.
