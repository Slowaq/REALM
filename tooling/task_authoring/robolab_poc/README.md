# RoboLab-style scene generation in REALM (PoC)

This is a throwaway proof of concept. The question it answers:

> Can RoboLab's predicate-based scene generation produce REALM task configs that load, settle
> and look right, when a model chooses the objects and relations and code chooses every number?

RoboLab's solver (`_vendor/robolab_scene_gen`, Apache-2.0, unmodified, see `NOTICE.md`) is driven
through a REALM adapter. Nothing in `realm/` changes. The only outputs are new YAML files under
`realm/config/tasks/ROBOLAB_POC/`, so no existing benchmark number can move.

## How RoboLab maps onto REALM

| RoboLab | This PoC |
|---|---|
| The LLM writes `place-on-base` with x/y | Not allowed. Anchors are placed at random by RoboLab's own code (seeded), and the spec carries no numbers |
| `left-of`/`right-of`/`front-of`/`back-of` in the table frame (+X = front) | `left-of`/`right-of`/`behind`/`in-front-of`, robot-relative. `frames.py` maps them onto each region through `env_config`'s x-mirroring |
| `place-in` (contents dropped 2 cm above the container mouth), `place-on` | `in`, `on`. The RoboLab geometry is kept; heights are shifted to REALM's 50 mm support clearance and the 10 mm on-top gap |
| `random-rot`, `facing-*`, arbitrary yaw candidates | Yaw is 0° or 90° only (AUTHORING_RULES.md) |
| `align-*` | Dropped: RoboLab parses these but its solver never applies them |
| A 312-object USD catalogue with dimensions | `catalog.json`: 50 assets in 36 categories, taken from REALM's own task configs. Run `cli catalog --dataset …` on the lab machine to get the full BEHAVIOR-1K set |
| A 0.5 × 0.8 m table | REALM spawn regions of about 0.4 × 0.5 m. These are the five axis-aligned regions the DROID100 generator also uses |
| Settle plus a screenshot in Isaac Sim | A short debug-policy rollout in the REALM container (below) |
| taskgen writes a success predicate | Not possible in REALM. `task_type` selects one of the fixed rubrics (put, pick, rotate, stack) |

## Files

| File | What it is |
|---|---|
| `GENERATE_SCENE.md` | The brief a coding agent follows. Written independently: RoboLab's skills are CC-BY-NC |
| `cli.py` | `regions`, `catalog`, `search`, `solve`, `plot`, `emit` |
| `scene.py` | Spec validation, the solver run, REALM heights and capacity, checks, the YAML document |
| `frames.py` | Authored `relative_bbox_position` ↔ the robot frame. Tested against `env_config`'s arithmetic |
| `catalog.py` | Builds and searches `catalog.json` |
| `specs/*.json` | Four example specs. Each has its emitted config under `realm/config/tasks/ROBOLAB_POC/` |
| `test_robolab_poc.py` | Host tests (unittest; no Isaac, no dataset) |

## 1. Host side (your Mac, no GPU)

From the repository root, on branch `feat/robolab-poc`:

```sh
uv sync --locked
uv run python -m unittest tooling.task_authoring.robolab_poc.test_robolab_poc -v
uv run python tests/test_task_type_literals.py
uv run python -m pytest -q tests/test_perturbation_task_types.py
```

The last two are REALM's own config tests. They scan every YAML under `realm/config/tasks/`, so
they also cover the new configs.

Try the loop by hand:

```sh
uv run python -m tooling.task_authoring.robolab_poc.cli regions
uv run python -m tooling.task_authoring.robolab_poc.cli search bowl
uv run python -m tooling.task_authoring.robolab_poc.cli solve tooling/task_authoring/robolab_poc/specs/pick_lemon_from_bowl.json
uv run python -m tooling.task_authoring.robolab_poc.cli plot  tooling/task_authoring/robolab_poc/specs/pick_lemon_from_bowl.json
open tmp/robolab_poc/pick_lemon_from_bowl.png
```

To have an agent write a new scene, give it a request and point it at `GENERATE_SCENE.md`. For
example: *"Follow tooling/task_authoring/robolab_poc/GENERATE_SCENE.md: put the orange on the plate,
with a mug behind the plate and a lemon to its left."* After it runs `cli emit`, re-run the tests
(they check the emitted configs) and commit the spec, `default.yaml` and `poc_provenance.json`.

## 2. Container side (lab GPU machine)

The container needs only the committed YAMLs; it does not run the PoC's Python. On the lab machine:

```sh
git fetch origin && git switch feat/robolab-poc       # after you push the branch from the Mac
export REALM_SIF=/path/to/realm.sif REALM_DATA_PATH=/path/to/realm/data
./scripts/run_apptainer.sh                            # or ./scripts/run_docker.sh --headless "$REALM_DATA_PATH"
```

Inside the container (repository at `/app`), run each task once with the debug policy. It needs no
server and records the scene settling:

```sh
cd /app
for t in put_soda_can_in_bowl_mug_left put_orange_in_bowl_with_lemon pick_lemon_from_bowl rotate_mug_chocolate_on_plate; do
  python -u examples/02_evaluate.py \
    --task_cfg_path ROBOLAB_POC/$t/default.yaml \
    --perturbation_id 0 --repeats 1 --max_steps 30 \
    --model_type debug --model_name debug --port 8000 \
    --experiment_name robolab_poc --run_id $t --log_dir /app/logs \
    --multi-view --no-render_on_demand
done
```

Then, on any machine with pandas:

```sh
for t in logs/robolab_poc/debug/*/; do uv run python scripts/videos_parquet_to_mp4.py "$t"; done
```

The debug policy drives the arm toward all-zero joints. Judge the scene from the first second or
so: anything the arm does after that isn't a scene problem. Don't trust exit codes (see CLAUDE.md);
read the log and the video.

Optional: run REALM's existing V-SC perturbation (`--perturbation_id 3`) on one task. It adds
sampled distractors around the authored layout, which checks that RoboLab's layout leaves room.

## 3. What to record per task

| Check | Pass when |
|---|---|
| Build | The env constructs, with no `Failed to place object` and no `KeyError` in the log |
| Settling | Every object comes to rest on the support within about 1 s. Nothing falls off, tips over, or is launched |
| `in` relations | The lemon ends up inside the bowl, not balanced on the rim |
| Robot-relative relations | In the external view, left/right/front/behind match the spec and the PNG from `cli plot` |
| Scale | Objects look plausibly sized. Watch the `BBOX_NOT_UPRIGHT` models (marker, pen) |
| Instruction | The instruction matches what's on the table |

## What the host run already shows

- **RoboLab's solver needs retries on REALM-sized regions.** In 2–17 seeded attempts per example,
  the first draft came back with overlapping footprints or a relation pushed off the support. The
  cause is in RoboLab's `SpatialSolver._optimize_placement`. Once no collisions remain, it clamps
  objects back inside the bounds but doesn't re-check collisions afterwards, so the clamp can push
  an object into its neighbour. The collision repair can also undo a relation that was already
  satisfied. Both are caught by `scene._check`, and `solve` moves to the next seed (50 by default).
  `poc_provenance.json` records every attempt.
- **Relations alone, without RoboLab's explicit x/y, are enough on these regions.** All four
  examples end up collision-free with every relation holding.
- **The catalogue is the limit.** 36 categories, and two of them (marker, pen) only have sizes
  taken from configs that tip them over. Build the full catalogue before judging coverage.
- **The DROID100 generator's region filter has a float bug.** It drops `Merom_1_int / Table` and
  `Pomaria_1_int / Light_Switch`, whose widths compute to 0.39999… against a `>= 0.4` test. This
  PoC keeps the same filter so the two tools stay comparable.

## Known limits

- Only regions whose robot yaw is a multiple of 90° are used. The two at 45° and 225° would make
  left/right diagonal.
- There is no articulated, drawer or faucet support. Task types are limited to put, pick, rotate
  and stack.
- Extents are axis-aligned boxes and the containment check is an ellipse. A mouth that fits on
  paper can still fail in simulation, which is what the container run is for.
- The catalogue built from configs uses *authored* extents, not the models' natural ones.
