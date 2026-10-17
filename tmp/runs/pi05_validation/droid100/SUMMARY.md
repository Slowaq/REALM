# pi05 scene validation: REALM_DROID100, all 104 scenes

The run checks the scenes, not the model. Each scene got one rollout: the Default perturbation, 500 steps max, DROID_mounted, `--no-render_on_demand`. The policy was the openpi server `pi05_full_droid_finetune` with checkpoint `pi05_droid_jointpos` on port 8000. Code: feat/agentic-task-generation 39dfbdf. Configs: origin/pilot-configs ccfc173 (104). The sweep ran from 2026-10-16 23:39 to 2026-10-17 05:31.

## Overall

**Success rate: 46/104 = 44.2%**

| task type | scenes | successes | success rate |
|---|---|---|---|
| pick | 43 | 23 | 53.5% |
| put | 48 | 18 | 37.5% |
| rotate | 3 | 0 | 0.0% |
| stack | 10 | 5 | 50.0% |

Rotate is 0/3. All three runs reached GRASP and stopped at ROTATED: pi05 grasped the object but did not turn it far enough for the checker.

## Crashes, load failures, logged errors

**None.** All 104 scenes loaded and ran to the end, and every one wrote a report and a video. No log has a traceback, an exception, a segfault, GPU out-of-memory, or an `[Error]` line other than the gymnasium "Casting input x to numpy array" warning that every clean run prints.

The first launch of this sweep did hit GPU out-of-memory. The pi05 server had been restarted and was preallocating 18 GB, and Isaac could not fit next to it. Those crashed runs were thrown away. The server was restarted with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and all 104 scenes were rerun from scratch, so none of the results above come from that launch.

## Scenes with TP 0.00

7 scenes reached no rubric stage. I reviewed each one's contact sheet (12 frames, external + wrist camera, `pi05/contact_sheets/<task>.jpg`).

| task | verdict | what the frames show |
|---|---|---|
| place_the_marker_in_the_bowl_767b1c | Policy failure (wrong object) | pi05 grasped the milk-bottle distractor over and over and never went for the marker. The marker is visible: small and dark-bodied, but its pink cap shows on the black table. |
| put_the_brush_in_the_cup_bcc31e | Policy failure | pi05 hovered over the pepper jar, pesto bottle and cup and never grasped anything. The brush (a thin wooden toothbrush) is visible on the table in the wrist view. |
| put_the_cap_in_the_bowl_f98bfa | Policy failure (wrong object) | pi05 picked up the sushi distractor and put it in the bowl. The cap is small (3 cm) and brown on a black table, but it is visible in both views. |
| put_the_cup_in_the_bowl_0b80d1 | Possibly unfair (a wrong object looks like the target) | pi05 grasped the blue bottle_of_paint distractor and lifted it again and again. That asset looks like a blue travel tumbler, so it reads as a cup. The actual coffee cup is small and visible next to the bowl. Swapping that distractor would remove the ambiguity. |
| put_the_marker_in_the_pot_f6f5d5 | Policy failure (wrong object) | pi05 grasped the red pruner distractor, which stands next to the marker, and pushed it against the pot. The marker is a thin dark stick on a brown table, visible from the external camera. |
| remove_the_lid_from_the_jar_6f67e1 | Policy failure | pi05 moved around the jar and knocked the rum bottle distractor over, but never grasped the lid. The lid (white, on top of the dark jar) is clearly visible and within reach. |
| remove_the_lid_from_the_mug_3b0f4c | Policy failure | pi05 hovered between the turmeric jar and the mug. It touched near the lid but never closed on it. The white lid on the mug is clearly visible and within reach. |

In summary, 6 of the 7 are policy failures. In none is the target hidden, out of reach, or missing. The one possible scene issue is put_the_cup_in_the_bowl, where the blue paint-bottle distractor looks like a cup.

## Per-task results

TP = final task progression. "Last stage reached" = the last rubric stage completed.

| task | type | TP | last stage reached | success |
|---|---|---|---|---|
| pick_up_the_cup_eef25d | pick | 1.00 | LIFT_LARGE | yes |
| pick_up_the_toothbrush_dcbbfd | pick | 0.67 | GRASP | no |
| pick_up_the_towel_1bdaf2 | pick | 0.33 | REACH | no |
| place_the_book_inside_the_box_e34edc | put | 1.00 | MOVE_CLOSE | yes |
| place_the_chocolate_bar_inside_the_sink_cfa4fa | put | 0.20 | REACH | no |
| place_the_lid_on_the_coffee_cup_456a7e | stack | 0.60 | LIFT_SLIGHT | no |
| place_the_marker_in_the_bowl_767b1c | put | 0.00 | (none) | no |
| place_the_mug_in_the_sink_7598ed | put | 1.00 | PLACE_INTO | yes |
| place_the_towel_inside_the_box_2f4888 | put | 0.20 | REACH | no |
| put_the_bag_of_chips_in_the_basket_e68620 | put | 0.40 | GRASP | no |
| put_the_ball_inside_the_bowl_c48915 | put | 1.00 | PLACE_INTO | yes |
| put_the_book_in_the_basket_5d00b8 | put | 0.60 | LIFT_SLIGHT | no |
| put_the_bottle_in_the_bowl_662239 | put | 0.80 | LIFT_SLIGHT | no |
| put_the_bottle_in_the_box_0e517f | put | 1.00 | MOVE_CLOSE | yes |
| put_the_bottle_in_the_sink_fe1ee8 | put | 1.00 | PLACE_INTO | yes |
| put_the_bottle_on_the_tray_ed1051 | stack | 1.00 | PLACE_ONTO | yes |
| put_the_bowl_in_the_sink_06bee8 | put | 1.00 | PLACE_INTO | yes |
| put_the_bowl_on_the_plate_b4fe86 | stack | 1.00 | PLACE_ONTO | yes |
| put_the_brush_in_the_cup_bcc31e | put | 0.00 | (none) | no |
| put_the_can_inside_the_sink_05972a | put | 0.60 | LIFT_SLIGHT | no |
| put_the_candy_in_the_cup_d5506e | put | 0.80 | MOVE_CLOSE | no |
| put_the_cap_in_the_bowl_f98bfa | put | 0.00 | (none) | no |
| put_the_carrot_in_the_pot_5cdf42 | put | 1.00 | PLACE_INTO | yes |
| put_the_cloth_in_the_basket_2ae4bf | put | 0.20 | REACH | no |
| put_the_cup_in_the_bowl_0b80d1 | put | 0.00 | (none) | no |
| put_the_cup_in_the_bowl_53015f | put | 1.00 | PLACE_INTO | yes |
| put_the_cup_in_the_pot_0b379b | put | 1.00 | MOVE_CLOSE | yes |
| put_the_egg_inside_the_cup_de0577 | put | 0.60 | LIFT_SLIGHT | no |
| put_the_garlic_in_the_bowl_7b721a | put | 1.00 | PLACE_INTO | yes |
| put_the_garlic_into_the_casserole_629761 | put | 1.00 | PLACE_INTO | yes |
| put_the_glass_into_the_jar_780251 | put | 0.20 | REACH | no |
| put_the_grapes_in_the_bowl_a731f9 | put | 0.40 | GRASP | no |
| put_the_green_block_in_the_tray_d39aaa | put | 1.00 | PLACE_INTO | yes |
| put_the_lid_in_the_bowl_a4f28c | put | 0.40 | GRASP | no |
| put_the_lid_on_the_bowl_fdb025 | stack | 0.60 | LIFT_SLIGHT | no |
| put_the_lid_on_the_pot_6f9c1b | stack | 0.20 | REACH | no |
| put_the_lid_on_the_towel_f35fef | stack | 0.20 | REACH | no |
| put_the_marker_in_the_cup_d31114 | put | 0.60 | LIFT_SLIGHT | no |
| put_the_marker_in_the_pot_f6f5d5 | put | 0.00 | (none) | no |
| put_the_orange_object_in_the_paper_bag_74a1ec | put | 1.00 | MOVE_CLOSE | yes |
| put_the_pen_in_the_bowl_f92409 | put | 1.00 | PLACE_INTO | yes |
| put_the_pen_in_the_bucket_9f1b9a | put | 0.60 | LIFT_SLIGHT | no |
| put_the_pen_in_the_cup_f2a5bc | put | 0.80 | MOVE_CLOSE | no |
| put_the_pen_on_the_bowl_1b0919 | stack | 1.00 | PLACE_ONTO | yes |
| put_the_plate_in_the_bowl_e7b2e9 | put | 0.60 | LIFT_SLIGHT | no |
| put_the_plate_on_top_of_the_bowl_f04584 | stack | 1.00 | PLACE_ONTO | yes |
| put_the_plum_in_the_bowl_ab75a2 | put | 1.00 | PLACE_INTO | yes |
| put_the_scissors_in_the_container_96bc4c | put | 0.40 | GRASP | no |
| put_the_scoop_inside_the_bowl_dc3a3f | put | 0.60 | LIFT_SLIGHT | no |
| put_the_spoon_in_the_cup_c0d032 | put | 0.80 | MOVE_CLOSE | no |
| put_the_spoon_in_the_holder_77da61 | put | 0.60 | LIFT_SLIGHT | no |
| put_the_tape_in_the_bowl_be2b7f | put | 1.00 | PLACE_INTO | yes |
| put_the_towel_in_the_bin_fae721 | put | 0.60 | LIFT_SLIGHT | no |
| put_the_towel_in_the_bowl_22f218 | put | 0.60 | LIFT_SLIGHT | no |
| put_the_towel_inside_the_basket_59d5e3 | put | 0.40 | GRASP | no |
| put_the_towel_inside_the_paper_bag_e2867b | put | 0.40 | GRASP | no |
| put_the_toy_in_the_bowl_1299a8 | put | 1.00 | PLACE_INTO | yes |
| put_the_toy_pan_on_the_cup_1f9f7c | stack | 1.00 | PLACE_ONTO | yes |
| put_the_wooden_spoon_in_the_basket_fffafd | put | 0.40 | GRASP | no |
| remove_the_bag_of_chips_from_the_sink_427db2 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_banana_from_the_box_859416 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_bottle_from_the_bowl_fa72b1 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_bottle_from_the_box_2fff8e | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_bow_from_the_box_5d5118 | pick | 0.67 | GRASP | no |
| remove_the_box_from_the_bowl_a06176 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_can_from_the_bowl_5612b5 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_chopstick_from_the_cup_ae0509 | pick | 0.67 | GRASP | no |
| remove_the_fork_from_the_mug_079e47 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_grey_block_from_the_tray_612012 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_lid_from_the_jar_6f67e1 | pick | 0.00 | (none) | no |
| remove_the_lid_from_the_mug_3b0f4c | pick | 0.00 | (none) | no |
| remove_the_lid_from_the_pot_8b4a39 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_lid_from_the_pot_and_put_it_on_the_towel_9f8054 | stack | 0.40 | GRASP | no |
| remove_the_marker_from_the_bowl_6facf8 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_marker_from_the_box_3c06f6 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_marker_from_the_cup_1fb5a0 | pick | 0.67 | GRASP | no |
| remove_the_marker_from_the_glass_1f96cb | pick | 0.33 | REACH | no |
| remove_the_marker_from_the_mug_e57716 | pick | 0.33 | REACH | no |
| remove_the_marker_from_the_pot_f89099 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_measuring_cup_from_the_bowl_32dd4a | pick | 0.33 | REACH | no |
| remove_the_mug_from_the_sink_ef7a36 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_orange_block_from_the_bowl_0c3ed4 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_pen_from_the_mug_e6c8fe | pick | 0.67 | GRASP | no |
| remove_the_pineapple_from_the_pot_3fb906 | pick | 0.67 | GRASP | no |
| remove_the_plate_from_the_bowl_82f5f5 | pick | 0.33 | REACH | no |
| remove_the_pot_from_the_bowl_956c1e | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_shaker_from_the_bowl_b7e492 | pick | 0.33 | REACH | no |
| remove_the_spice_bottle_from_the_bowl_bfa457 | pick | 0.67 | GRASP | no |
| remove_the_spoon_from_the_basket_202542 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_spoon_from_the_mug_d3b3a7 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_tea_bag_from_the_basket_68e23f | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_toy_from_the_basket_d94b4e | pick | 0.67 | GRASP | no |
| remove_the_toy_from_the_bowl_e7084f | pick | 0.67 | GRASP | no |
| remove_the_yellow_block_from_the_pan_404849 | pick | 1.00 | LIFT_LARGE | yes |
| remove_the_yellow_block_from_the_pot_203104 | pick | 0.67 | GRASP | no |
| remove_the_yellow_object_from_the_plate_851e78 | pick | 0.67 | GRASP | no |
| rotate_the_carton_d24666 | rotate | 0.67 | GRASP | no |
| rotate_the_pot_20a53e | rotate | 0.67 | GRASP | no |
| take_the_blue_block_out_of_the_box_and_put_it_in_the_bowl_52e039 | put | 1.00 | PLACE_INTO | yes |
| take_the_pen_out_of_the_bowl_a55d56 | pick | 1.00 | LIFT_LARGE | yes |
| take_the_salt_shaker_out_of_the_pot_fbfd2e | pick | 1.00 | GRASP | yes |
| take_the_soap_dispenser_out_of_the_container_5b8a87 | pick | 1.00 | LIFT_LARGE | yes |
| take_the_teddy_bear_out_of_the_box_d19562 | pick | 0.67 | GRASP | no |
| turn_the_bottle_5069c2 | rotate | 0.67 | GRASP | no |

## Label fix in scripts/videos_parquet_to_mp4.py (not committed)

The script matched each video to its result by repeat number alone. In a folder holding many tasks, every video got the label of whichever report was read last (all `TP1.00_SUCCESS` in the first 10-scene run). Results are now keyed by (task, perturbation, repeat). Perturbation is included because one task's several perturbation reports would otherwise collide in the same way. Checked by re-extracting the old 10-scene run: every label matches its report.

```diff
diff --git a/scripts/videos_parquet_to_mp4.py b/scripts/videos_parquet_to_mp4.py
index 5458640..cb6da46 100644
--- a/scripts/videos_parquet_to_mp4.py
+++ b/scripts/videos_parquet_to_mp4.py
@@ -36,7 +36,8 @@ def load_report(run_dir):
         with open(os.path.join(rep_dir, fn)) as f:
             for row in csv.DictReader(f):
                 try:
-                    out[int(row["run_id"])] = (float(row["task_progression"]), row["stage"])
+                    key = (row["task"], row["perturbation"], int(row["run_id"]))
+                    out[key] = (float(row["task_progression"]), row["stage"])
                 except (KeyError, ValueError):
                     pass
     return out
@@ -69,7 +70,7 @@ def main():
                 print(f"  {pq}: not an MP4 container (magic {bytes(blob[:8])!r}) -- skipped")
                 continue
             idx = int(row.get("repeat", written))
-            tp, stage = report.get(idx, (None, None))
+            tp, stage = report.get((row["task"], row["perturbation"], idx), (None, None))
             tag = f"_TP{tp:.2f}_{stage}" if tp is not None else ""
             name = f"{args.prefix}{row['task']}_{row['perturbation']}_run{idx:03d}{tag}.mp4"
             with open(os.path.join(outdir, name), "wb") as f:
```

## Files

- `results/<task>.json`: one per scene. Holds the rubric stages with their step numbers, TP, success, the report row (without qpos, actions or video) and any error lines.
- `summary.json`: all rows.
- `pi05/`: the eval artifacts: reports, parquets, `videos_mp4/`, and `contact_sheets/` (12-frame sheet and last frame per scene). Not pushed.
- `logs/`: `run_all.sh` (the sweep driver), `summarize.py`, `tasks/<task>.log` (container logs) and `serve_policy.log`.
