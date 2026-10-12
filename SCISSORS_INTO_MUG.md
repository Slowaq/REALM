# Scissors into mug: how to run it

The "put the scissors into the mug" task in the IMPACT scene, with the DROID robot.

## 1. Start the container

On the host (your normal prompt, e.g. `elmo@elmo-B860-DS3H:~$`):

```bash
cd /home/elmo/balucwil/REALM-scissors-into-mug
source ./scripts/run_docker.sh        # answer "y" to the EULA
```

When you see `(behavior) root@...:/app#`, you are inside the container. `/app` is this folder.

> Always start the container from this folder. Docker mounts the folder you start it from as `/app`.
> The datasets come from `REALM_DATA_PATH` (set in `~/.bashrc`), so nothing else needs configuring.

## 2. Look at the scene (no robot control)

Inside the container:

```bash
python /app/view_task.py
```

An Isaac Sim window opens showing the table, robot, scissors and mug. The first start can take a few minutes. Press Ctrl+C to quit.
To view another task, pass its config: `python /app/view_task.py IMPACT/put_banana_into_box/default.yaml`.

## 3. Run the debug evaluation

Inside the container:

```bash
python examples/02_evaluate.py --model_type debug --model_name debug --experiment_name debug --port 8000 --task_id 0 --perturbation_id 0 --repeats 1 --max_steps 1000000 --task_cfg_path IMPACT/scissors_into_mug/default.yaml    
```

- `--model_type debug` needs **no policy server**. The robot gets zero actions, so it stays still. Use it to check that the task loads and records.
- `--port` is required by the CLI but ignored for `debug`.
- Add `OMNIGIBSON_HEADLESS=1` before `python` to run without a window.
- Results and videos go to `logs/` in this folder.

For a real policy, start a model server first (see the main `README.md`) and swap in e.g.
`--model_type openpi --model_name pi05 --port 8000`.

## 4. Watch the results (on the host, not in the container)

```bash
cd /home/elmo/balucwil/REALM-scissors-into-mug
export REALM_LOGS=$PWD/logs
uv sync --locked
uv run streamlit run tooling/realm_viewer/dashboard.py
```

Open your experiment, scroll down and click "unpack video parquet".

## What's where

| File | What it is |
|---|---|
| `realm/config/tasks/IMPACT/scissors_into_mug/default.yaml` | The task: objects, positions, instruction, cameras (`default` + `CP3`) |
| `custom_assets/ycb/scissors.usd`, `mug.usd` | Self-contained YCB models with physics (mug rotated to stand upright) |
| `custom_assets/ycb/build_ycb.py` | Rebuilds these from NVIDIA YCB files (5th argument = X rotation, e.g. `-90`) |
| `realm/config/scenes/scenes.yaml` | IMPACT robot pose. z = −0.133891 puts the arm base at table height (0.73) |
| `realm/config/scenes/IMPACT/default/scene_definition.yaml` | IMPACT room. The 5 bookshelves are disabled |
| `view_task.py` | The scene viewer from step 2 |

Placing objects: x points away from the robot (robot base at x = −0.45), y is left/right, and z is height (tabletop ≈ 0.74).
`orientation` is a quaternion `[x, y, z, w]`, and `[0, 0, 0, 1]` means no rotation.

## Troubleshooting

- **`cd: No such file or directory` for `/home/elmo/...`**: you're inside the container. Type `exit` until the prompt is `elmo@...` again.
- **`$'\E[200~cd': command not found`**: a terminal paste glitch. Retype the command or paste it again.
- **`Accessed invalid null prim`**: a `usd_path` points at a wrapper file such as `custom_assets/*_ycb_phys.usd`, which only links to NVIDIA's server. Use the files in `custom_assets/ycb/`.
- **`Exactly one root link should have been found`**: a custom USD lacks the `root → base_link (Rigid Body) → mesh` structure. See `custom_assets/ycb/*.usd`.
- **The window freezes ("not responding")**: press Ctrl+C. If it stays stuck, close the terminal or use `docker ps` and `docker stop <name>`, then start the container again. Headless mode (`OMNIGIBSON_HEADLESS=1`) doesn't freeze.
- **Red `[Error] [gpu.foundation.plugin]` and the `omni.hydra` / `carb` warnings**: Isaac Sim log noise. You can ignore them.
