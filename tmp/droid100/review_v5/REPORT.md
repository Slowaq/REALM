# render_review v5: 8 REALM_DROID100 tasks

- Code: `feat/agentic-task-generation` at 1098632. It includes 9977373, which adds the TIPPED, LEFT_CONTAINER and LEFT_SUPPORT checks and stops failing on distractor or in-container drift.
- Configs: `realm/config/tasks/REALM_DROID100` restored from `origin/pilot-configs`, 104 task folders.
- Each task was run on its own:

  ```
  tmp/gpu_session/run_in_container.sh tooling/task_authoring/render_review.py --task_cfg_path REALM_DROID100/<folder>/default.yaml --out tmp/droid100/review_v5/droid100
  ```

  Default 60 steps. Log: `tmp/gpu_session/render_v5.log`.
- GPU: a pi05 server (openpi serve_policy, 13.4 GB) was running. Per your answer, the renders ran alongside it. There were no out-of-memory or CUDA errors, and every task wrote 6 renders with 0 render errors. The containers exited 139 (a segfault at teardown), which is expected; exit codes were ignored.
- Verdicts come from the VERDICT lines and JSONs. The image judgement uses t060 (after the settle), with t000 for comparison.

## Summary

| task | verdict | findings | image matches verdict? | main object visible (sensor0 / sensor1) |
|---|---|---|---|---|
| put_the_garlic_in_the_bowl_7b721a | PASS | none | yes | yes / yes |
| put_the_grapes_in_the_bowl_a731f9 | PASS | none | yes | partly, behind the bowl rim / yes |
| put_the_tape_in_the_bowl_be2b7f | PASS | none | yes | yes / yes |
| put_the_cloth_in_the_basket_2ae4bf | PASS | none | yes | partly, behind the catsup bottle / yes |
| put_the_plum_in_the_bowl_ab75a2 | PASS | none | yes | yes / yes |
| put_the_pen_in_the_bucket_9f1b9a | PASS | none | yes | yes / no, hidden behind the bucket |
| put_the_green_block_in_the_tray_d39aaa | PASS | none | yes | yes / yes |
| remove_the_banana_from_the_box_859416 | **FAIL** | LEFT_CONTAINER banana | **no: false positive** | yes / yes |

The main object is visible in at least one external camera in all 8 tasks.

## Per task

- **put_the_garlic_in_the_bowl:** garlic, bowl and distractors (salsa bottle, zucchini, bath-salt jar) all stand where they started. Largest drift_xy 0.0012 m; drift_z −0.006 to −0.027 m. The garlic is clearly visible next to the bowl in both cameras.
- **put_the_grapes_in_the_bowl:** grapes, red bowl, stapler, eraser and chilli jar are all at rest, drift_xy ≤ 1e-5. The grapes are fully visible in sensor1 and partly visible behind the bowl rim in sensor0.
- **put_the_tape_in_the_bowl:** the tape roll lies flat beside the bowl. The eraser and chopstick are at rest. The tape is visible in both cameras.
- **put_the_cloth_in_the_basket:** cloth, wire basket, catsup bottle, coffee jar and banana are all at rest; the largest drift_xy is the catsup bottle's 0.004 m. The cloth is fully visible in sensor1 and partly visible behind the catsup bottle in sensor0.
- **put_the_plum_in_the_bowl:** plum, bowl, protein-powder jar, mug and chopstick are all at rest. The plum is visible in both cameras.
- **put_the_pen_in_the_bucket:** the pen lies on the table; the bucket, cranberry-juice bottle, flask and stapler are upright. The pen is visible in sensor0. In sensor1 it is hidden behind the bucket, and sensor1 is framed mostly on the bucket and bottle.
- **put_the_green_block_in_the_tray:** the block sits beside the tray. The supplements bottle, marker and tongs are at rest. The block is visible in both cameras.
- **remove_the_banana_from_the_box:** see below.

## remove_the_banana_from_the_box: why the FAIL is a false positive

Finding: `LEFT_CONTAINER banana: declared inside 'box' but its lower face 0.7395 is at or below the container's own lower face 0.7377: it went through the floor`

**The images show the banana resting on the box floor, inside the box, in both cameras at t060.** It casts a shadow on the floor and has not gone through it.

The numbers agree with the images:
- The banana's lower face (0.73945) is 1.8 mm above the box's lower face (0.73768).
- The box's lower face equals the measured table (0.73766). So the box floor is about 1.8 mm thick, and the banana is lying on top of it.

The check fails any object within `THROUGH_FLOOR_M` = 3 mm of the container's lower face, on the assumption that "floors are never this thin". This box's floor is thinner than that.

For comparison, the v4 pen that really went through its box floor ended 0.4 mm *below* the box's lower face. A threshold at or below the container's lower face (≤ 0) would separate the two cases. This is an observation only; no code was changed.

The banana is also **yellow** now. In v4 it rendered black; 1098632 says it stops offering broken-rendering asset models.

## Bottom line

7 of 8 tasks PASS, and the images agree. The single FAIL (remove_the_banana_from_the_box) is a false positive from the 3 mm through-floor threshold. That start state is fine.
