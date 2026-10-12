# render_review v4 — REALM_DROID100

- Code: `feat/agentic-task-generation` at ff067fc (two commits past the expected dd49e39; those two only stop tracking `asset_catalog.json` and change no render code).
- Configs: `realm/config/tasks/REALM_DROID100` restored from `origin/pilot-configs` (aa9f2a5, "1.5 cm start above the measured table"), 103 task folders.
- GPU: no pi05 server was running. robometer_server (9.8 GB, someone else's) shared the GPU; there were no out-of-memory errors.
- Command: `tmp/gpu_session/run_in_container.sh tooling/task_authoring/render_review.py --family REALM_DROID100 --out tmp/droid100/review_v4/droid100` (default 60 steps). Log: `tmp/gpu_session/render_v4_droid100.log`.
- Verdicts are taken from the VERDICT lines and JSONs only. Every task has a JSON: no NO_JSON and no ERROR.
- Images: the code writes t000 and t060. All image judgements below use **t060** (after the settle).

## Tally

| PASS | FAIL | finding codes |
|---|---|---|
| 91 | 12 | UNSTABLE 16, DRIFT 14 |

- **HIGH_DROP: 0.** The largest start gap above the measured surface was 0.052 m (limit 0.06); the median was 0.018 m.
- **PHYSICS_FROZEN: 0.** It is no longer a hard finding in this code.
- **PENETRATION: 0.**
- **Floor-only penetration check:** 1 task (remove_the_bag_of_chips_from_the_sink, where the sink sits on a raised counter).
- v3 for comparison: 81 PASS / 22 FAIL, UNSTABLE 31, DRIFT 21, PENETRATION 2.

## drift_z of authored objects (491 objects, lights excluded)

| statistic | drift_z |
|---|---|
| median | **−0.0188 m** |
| least negative (max) | −0.0061 m |
| most negative (min) | −0.1697 m (mug in remove_the_mug_from_the_sink) |

The deciles are −0.048, −0.030, −0.024, −0.021, −0.019, −0.017, −0.016, −0.015 and −0.014, so most objects fall 1.4–3 cm. v3's median was −0.140 m.

**The large drops are almost all objects that start inside a container.** HIGH_DROP does not check these, because it only looks at objects on the task surface. They still fall 10–17 cm:

| task | object | drift_z |
|---|---|---|
| remove_the_mug_from_the_sink | mug | −0.170 |
| remove_the_knife_from_the_cup | knife | −0.149 |
| remove_the_bag_of_chips_from_the_sink | bag_of_chips | −0.137 |
| remove_the_tea_bag_from_the_basket | tea_bag | −0.135 |
| remove_the_toy_from_the_basket | toy | −0.127 |
| remove_the_spatula_from_the_holder | spatula | −0.119 |
| put_the_marker_in_the_pot | distractor_pruner | −0.119 |
| remove_the_pineapple_from_the_pot | pineapple | −0.118 |
| take_the_salt_shaker_out_of_the_pot | salt_shaker | −0.118 |
| remove_the_marker_from_the_pot | marker | −0.115 |
| take_the_soap_dispenser_out_of_the_container | soap_dispenser | −0.113 |
| remove_the_spoon_from_the_basket | spoon | −0.108 |

## Every FAIL (all UNSTABLE: drift_xy > 0.03 m)

| task | findings |
|---|---|
| put_the_book_in_the_basket_5d00b8 | UNSTABLE distractor_pruner 0.146 |
| put_the_marker_in_the_pot_f6f5d5 | UNSTABLE distractor_pruner 0.144 |
| put_the_towel_in_the_bowl_22f218 | UNSTABLE distractor_pruner 0.146 |
| put_the_towel_inside_the_paper_bag_e2867b | UNSTABLE distractor_bottle_of_mustard 0.033 |
| remove_the_bag_of_chips_from_the_sink_427db2 | UNSTABLE bag_of_chips 0.085 |
| remove_the_banana_from_the_box_859416 | UNSTABLE banana 0.039 |
| remove_the_knife_from_the_cup_aec737 | UNSTABLE knife 0.099; UNSTABLE cup 0.070 |
| remove_the_lid_from_the_bowl_8f6fd3 | UNSTABLE lid 0.126; UNSTABLE bowl 0.069 |
| remove_the_marker_from_the_bowl_6facf8 | UNSTABLE distractor_bottle_of_coconut_oil 0.129 |
| remove_the_mug_from_the_sink_ef7a36 | UNSTABLE mug 0.066 |
| remove_the_spatula_from_the_holder_010e2c | UNSTABLE spatula 0.119; UNSTABLE holder 0.064 |
| take_the_sunglasses_out_of_the_box_a46534 | UNSTABLE sunglasses 0.031; UNSTABLE box 0.035 |

**distractor_pruner fails the same way in all three tasks that use it.** It moves about 0.145 m and its tilt changes about 90°. It is authored standing on edge and falls flat.

DRIFT (warning only, all PASS):
- place_the_mug_in_the_sink: sink
- put_the_ball_inside_the_bowl: ball
- put_the_can_inside_the_sink: sink
- remove_the_chopstick_from_the_cup: chopstick
- remove_the_fork_from_the_mug: fork
- remove_the_marker_from_the_cup: marker
- remove_the_marker_from_the_glass: marker
- remove_the_marker_from_the_mug: marker
- remove_the_pen_from_the_mug: pen
- remove_the_pineapple_from_the_pot: pineapple
- remove_the_spoon_from_the_basket: distractor_bell_pepper
- remove_the_spoon_from_the_mug: spoon
- rotate_the_pot: distractor_bell_pepper
- take_the_pen_out_of_the_bowl: pen

## From the t060 images

### Floating
None. Every object rests on a surface in both external cameras.

### Fallen off the table
None. No object's lower face is more than 2 cm below the measured surface. v3's teddy bear on the floor and almond-oil bottle over the edge are both fixed.

### Tipped over
| task | object |
|---|---|
| remove_the_knife_from_the_cup | **cup** lies on its side |
| remove_the_lid_from_the_bowl | **bowl** tipped about 48°, its opening facing sensor0 |
| remove_the_spatula_from_the_holder | **holder** (white cube) lies on its side |
| remove_the_marker_from_the_bowl | distractor coconut-oil bottle lies on its side (moved 0.13 m) |
| put_the_book_in_the_basket / put_the_marker_in_the_pot / put_the_towel_in_the_bowl | distractor pruner fell from on-edge to flat (moved 0.14 m) |
| remove_the_mug_from_the_sink | **mug** starts about 17 cm above the sink and ends leaning against the sink wall inside it |

### Left the container it starts in
| task | what happened |
|---|---|
| remove_the_knife_from_the_cup | cup tipped and the knife slid out; the knife is visible in sensor1 sticking out of the fallen cup |
| remove_the_lid_from_the_bowl | lid slid off the tipped bowl onto the table (lower face at table height) |
| remove_the_spatula_from_the_holder | spatula fell out of the holder onto the counter (moved 0.12 m) |
| remove_the_pen_from_the_box | **pen sank through the box floor.** At t000 it is inside the box; at t060 its lower face (0.8046) is below the box's lower face (0.8050), at table height, and it is not visible in either camera. render_review does not flag this. |

Borderline: in take_the_sunglasses_out_of_the_box the sunglasses' lower face is also 0.3 mm below the box's lower face. However, they are clearly visible inside the box in both cameras.

All other remove/take tasks keep their main object inside the starting container. I checked this numerically (settled XY within the container footprint) as well as in the images.

### Main object visible in neither external camera
| task | why |
|---|---|
| remove_the_pen_from_the_box_c9c5dd | pen is under the box floor (see above) |
| put_the_glass_in_the_jug_f6c064 | sensor1 is framed on the table's near corner (wine bottle and jug fill the frame); in sensor0 the glass cannot be told apart from the jug behind it |

The v3 "neither camera" cases are now visible:
- remove_the_black_spoon_from_the_basket: the spoon shows in both cameras.
- remove_the_purple_marker_from_the_box: the marker is in the box in both cameras.
- put_the_glass_cup_into_the_jar: the shot glass is visible in both cameras.

These main objects are visible in only one camera: the tea bag (hidden by the basket wall in sensor0), the candy (sensor1 only) and the toy pan (sensor1 only).
