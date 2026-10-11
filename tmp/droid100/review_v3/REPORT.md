# render_review v3 — REALM_DROID10 + REALM_DROID100

Run 2026-10-11/12 on feat/agentic-task-generation @ cb68868, REALM_DROID100 configs from
origin/pilot-configs @ c882d51 (103 task folders). Launcher: tmp/gpu_session/run_in_container.sh.
GPU: no pi05 server; only robometer_server (9.8 GB) + desktop daemon. 0 out-of-memory errors.
Verdicts taken from the VERDICT lines and per-task JSONs, never from exit codes.

**Image naming.** cb68868 renders `_t000` (after reset) and `_t060` (after the 60-step settle);
there are no `_t030` images. Every image judgement below uses `_t060`.

## Smoke test (REALM_DROID10/put_banana_into_box → smoke/)

| object | drift_xy | drift_z |
|---|---|---|
| banana | 0.0396 | **-0.1375** |
| box | 0.0001 | -0.0000 |
| orange | 0.0012 | -0.0000 |
| lemon | 0.0049 | -0.0084 |
| toy_dice1 | 0.0001 | 0.0000 |
| toy_dice2 | 0.0001 | -0.0000 |

At t000 the banana hangs in the air near the wrist camera; at t060 everything rests on the
round table. Objects authored already resting read ~0, which is expected.

## Tallies

| family | tasks | PASS | FAIL | finding codes | median drift_z (authored) | drift_xy > 0.03 |
|---|---|---|---|---|---|---|
| REALM_DROID10 | 11 | 5 | 6 | FLOATING 5, PHYSICS_FROZEN 4, UNSTABLE 2 | -0.0474 m (50 obj) | 2 |
| REALM_DROID100 | 103 | 81 | 22 | UNSTABLE 31, DRIFT 21, PENETRATION 2 | -0.1396 m (491 obj) | 31 |

## PHYSICS_FROZEN

DROID10: close_drawer, open_drawer, pour_from_bottle, push_switch. DROID100: none.

All four have every object at exactly 0.0 drift. But the arm visibly moves between t000 and
t060 (gripper opens) in all four, so the simulator did step. Their objects were authored already
at rest: pour_from_bottle's items sit on the table at t000, push_switch's switch is on the wall,
and the drawer tasks have only the cabinet and table. These look like false positives of the
"everything is authored above its rest height" assumption, not frozen physics.

## Every FAIL

### REALM_DROID10
| task | findings |
|---|---|
| close_drawer | FLOATING[light_over_table]; PHYSICS_FROZEN |
| open_drawer | FLOATING[light_over_table]; PHYSICS_FROZEN |
| pour_from_bottle | FLOATING[coffee_cup]; PHYSICS_FROZEN |
| push_switch | PHYSICS_FROZEN |
| put_banana_into_box | UNSTABLE[banana] 0.0396; FLOATING[banana] |
| rotate_mug | UNSTABLE[tea_bag] 0.0880 |

### REALM_DROID100 (number = drift_xy in m)
| task | findings |
|---|---|
| move_the_bowl_and_cup_then_after_put_the_cup_in_the_bowl_53015f | UNSTABLE[cup] 0.068 |
| pick_the_marker_from_the_pot_and_put_it_on_the_table_f89099 | UNSTABLE[marker] 0.104; UNSTABLE[pot] 0.193 |
| pick_up_the_black_lid_from_the_jar_and_put_it_on_the_counter_6f67e1 | UNSTABLE[lid] 0.311; UNSTABLE[jar] 0.091 |
| pick_up_the_lid_from_the_pot_and_put_it_in_the_bowl_a4f28c | UNSTABLE[bowl] 0.071; UNSTABLE[soy_sauce] 0.203 |
| pick_up_the_marker_from_the_bowl_and_put_it_on_the_table_6facf8 | UNSTABLE[coconut_oil] 0.130 |
| pick_up_the_object_from_the_sink_and_put_it_on_the_counter_427db2 | UNSTABLE[bag_of_chips] 0.085 |
| pick_up_the_towel_1bdaf2 | DRIFT[oatmeal] 0.012; UNSTABLE[honey] 0.072; DRIFT[rum] 0.027 |
| pick_up_the_toy_pan_and_put_on_the_black_cup_1f9f7c | UNSTABLE[almond_oil] 0.405; PENETRATION[almond_oil] |
| pick_up_the_wooden_spatula_from_the_utensil_holder_and_put_it_on_the_tab_010e2c | UNSTABLE[spatula] 0.118; UNSTABLE[holder] 0.064 |
| put_the_book_in_the_black_basket_5d00b8 | UNSTABLE[pruner] 0.149 |
| put_the_can_inside_the_sink_05972a | UNSTABLE[can] 0.109 |
| put_the_marker_in_the_pot_f6f5d5 | UNSTABLE[pruner] 0.150 |
| put_the_towel_in_the_bowl_22f218 | DRIFT[bowl] 0.027; UNSTABLE[powder] 0.222; UNSTABLE[pruner] 0.143 |
| remove_a_tissue_from_the_box_d19562 | UNSTABLE[teddy_bear] 0.758; PENETRATION[teddy_bear]; UNSTABLE[box] 0.127 |
| remove_the_cup_from_the_sink_ef7a36 | UNSTABLE[mug] 0.067 |
| remove_the_knife_from_the_cup_aec737 | UNSTABLE[knife] 0.104; UNSTABLE[cup] 0.072 |
| remove_the_lid_from_the_bowl_8f6fd3 | UNSTABLE[lid] 0.078; DRIFT[bowl] 0.025 |
| remove_the_marker_from_the_cup_1fb5a0 | DRIFT[marker] 0.016; UNSTABLE[sugar] 0.161 |
| remove_the_pink_plate_from_the_blue_bowl_82f5f5 | UNSTABLE[plate] 0.180; DRIFT[dill_seed] 0.014; UNSTABLE[bowl] 0.098 |
| remove_the_shaker_from_the_bowl_and_put_it_on_the_table_b7e492 | UNSTABLE[shaker] 0.138; UNSTABLE[bowl] 0.125 |
| remove_the_yellow_object_from_the_orange_plate_851e78 | UNSTABLE[bay_leaves] 0.074 |
| take_the_salt_cellar_out_of_the_white_pot_fbfd2e | UNSTABLE[salt_shaker] 0.032 |

The two sink PENETRATION false positives from v2 are gone; both sink tasks now fail (or pass)
on drift only.

## From the t060 images

No object hangs in the air in any task in either family. DROID10: all objects rest and every
main object is visible in both external cameras.

### Fallen off / partly off the table
- **remove_a_tissue_from_the_box_d19562**: teddy bear on the floor (low_z 0.0004, moved 0.76 m);
  the box slid 0.13 m and is empty in both cameras.
- **pick_up_the_toy_pan_and_put_on_the_black_cup_1f9f7c**: almond-oil bottle rolled 0.41 m to the
  table edge, low_z 0.850 vs table 0.899 (hanging over the edge).

### Tipped over / fell out of container (rest on a surface, but the start state is broken)
- pick_up_the_wooden_spatula_…_010e2c: spatula fell out of the holder onto the counter.
- put_the_can_inside_the_sink_05972a: can tipped and rolled 0.11 m, lies at the table edge.
- remove_the_knife_from_the_cup_aec737: cup tipped onto its side; knife fell out.
- remove_the_pink_plate_from_the_blue_bowl_82f5f5: plate slid off the bowl onto the table.
- remove_the_shaker_from_the_bowl_…_b7e492: bowl tipped against the back wall; shaker fell out.
- take_the_salt_cellar_out_of_the_white_pot_fbfd2e: salt cellar lies on its side inside the pot.
- Distractors tipped/rolled: pick_up_the_marker_from_the_bowl (coconut oil), pick_up_the_towel
  (rum bottle across the honey jar), put_the_towel_in_the_bowl (powder 0.22 m to the edge),
  remove_the_marker_from_the_cup (sugar jar 0.16 m).

### Main object not visible in at least one external camera
| task | sensor0 | sensor1 |
|---|---|---|
| pick_up_the_bowl_and_put_it_in_the_sink_06bee8 | yes | no (behind sink) |
| pick_up_the_lid_from_the_pot_and_put_it_in_the_bowl_a4f28c | no | yes |
| pick_up_the_toy_pan_and_put_on_the_black_cup_1f9f7c | no | yes |
| pick_up_the_wooden_spatula_…_010e2c | yes | no |
| place_the_snack_bar_inside_the_sink_cfa4fa | yes | no (behind sink) |
| place_the_yellow_and_white_book_inside_the_black_box_e34edc | sliver at frame edge | yes |
| put_a_piece_of_candy_in_the_cup_d5506e | no | yes |
| put_the_glass_cup_in_the_clear_jug_f6c064 | no | no (sensor1 framed on a corner) |
| put_the_glass_cup_into_the_jar_780251 | yes | no (sensor1 mostly chairs) |
| put_the_marker_in_the_pot_f6f5d5 | yes | no (behind pot) |
| remove_a_sachet_from_the_basket_68e23f | no (basket wall) | yes |
| remove_the_black_spoon_from_the_basket_202542 | no | no (inside basket) |
| remove_the_blue_spoon_from_the_silver_object_…_fffafd | yes | no |
| remove_the_purple_marker_from_the_box_…_3c06f6 | no | no |
| rotate_the_black_pot_20a53e | cut at frame edge | yes |

### Start state doesn't match the instruction (not physics, noted in passing)
- pick_the_spoon_in_the_bowl_…_c0d032: no bowl; spoon on the table.
- pick_up_the_lid_from_the_pot_…_a4f28c: config has no pot.
- pick_up_the_silver_lid_on_the_paper_towel_…_f35fef: lid beside the towel, not on it; no clear container.
- remove_the_purple_marker_from_the_box_…_3c06f6: marker on the table, box empty.
- remove_the_white_bowl_from_the_plate_…_b4fe86: bowl beside the plate (plate ghosted in sensor1).
