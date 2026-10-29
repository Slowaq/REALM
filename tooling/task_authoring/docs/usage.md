# Usage

[← Back to overview](../README.md) · [How it works](how-it-works.md) · [Walkthrough](walkthrough.md) · **Usage** · [Results](results.md) · [Reference](reference.md) · [Limitations](limitations.md)

All commands run from the repository root. Host steps need Python 3.10+ and the repo's `uv` environment, but no GPU. Steps marked **GPU** run inside the REALM container (see the [REALM installation guide](../../../wiki/Installation.md)).

| I want to… | Go to |
|---|---|
| set up once on a new machine | [One-time setup](#one-time-setup) |
| rebuild the DROID-based family | [Generate the DROID family](#generate-the-droid-family) |
| turn my own sentences into tasks | [Generate tasks from your own instructions](#generate-tasks-from-your-own-instructions) |
| check tasks in the simulator | [Run the physics check](#run-the-physics-check-gpu) |
| evaluate a policy on generated tasks | [Evaluate a policy](#evaluate-a-policy-gpu) |
| author one task by hand | [The dashboard](dashboard.md) |
| refresh the charts in these docs | [Regenerate the figures](#regenerate-the-figures) |

## One-time setup

The generator searches a JSON snapshot of the BEHAVIOR-1K asset catalogue rather than the multi-gigabyte dataset itself. Build it once on any machine that has the dataset (no GPU needed), then copy the file to machines that don't:

```sh
python -m tooling.task_authoring.agent.catalog build \
    --dataset "$REALM_DATA_PATH/datasets/behavior-1k-assets"
python -m tooling.task_authoring.agent.catalog show     # expect: source=dataset categories=2430
```

The file `tooling/task_authoring/asset_catalog.json` is gitignored. Without it, every command accepts `--dataset <path>` instead. A wrong `--dataset` path is an error; it never silently indexes zero assets.

## Generate the DROID family

**1. Rank the instructions.** This needs the DROID instruction counts (`tmp/droid_instruction_counts.json`, from the DROID survey):

```sh
python3 -m tooling.task_authoring.select_droid100 \
    --counts tmp/droid_instruction_counts.json --limit 300 \
    --output tmp/droid100/DROID_clusters.json
```

**2. Generate, one instruction at a time.** Pick one driver:

| Driver | Command | Best for |
|---|---|---|
| Coding agent (how the current family was made) | Tell any shell-capable coding agent: *"Follow `tooling/task_authoring/agent/GENERATE_TASKS.md` for the next 20 instructions in `tmp/droid100/DROID_clusters.json`."* | careful, reviewable runs; no API key |
| API, unattended | `uv run python -m tooling.task_authoring.agent.run tmp/droid100/DROID_clusters.json --model <model-id> --report tmp/droid100/agent_report.json` | large batches; needs `uv add anthropic` and `ANTHROPIC_API_KEY` |
| Offline smoke test | `uv run python -m tooling.task_authoring.agent.run tmp/droid100/DROID_clusters.json --offline --limit 10` | checking the plumbing; its keyword grounding is not a real generator |

With the coding-agent driver, the loop the agent runs looks like this ([full example](walkthrough.md)):

```sh
python -m tooling.task_authoring.agent.cli next  tmp/droid100/DROID_clusters.json   # next unprocessed instruction
python -m tooling.task_authoring.agent.cli start "<instruction>" --ranking-id <id> --rank <n>
python -m tooling.task_authoring.agent.cli call <task_id> search_assets '{"query":"mug","role":"target"}'
python -m tooling.task_authoring.agent.cli call <task_id> propose_layout '{...roles...}'
python -m tooling.task_authoring.agent.cli call <task_id> validate_draft
python -m tooling.task_authoring.agent.cli call <task_id> submit_task '{"decisions":["..."]}'
```

**3. Check progress, then export.**

```sh
python -m tooling.task_authoring.agent.cli status tmp/droid100/DROID_clusters.json
# {"submitted": 114, "declined": 113, "open": 0, "stale": 0, "remaining_in_list": 0}

python -m tooling.task_authoring.agent.cli export        # -> realm/config/tasks/REALM_DROID100/
python -m tooling.task_authoring.validation realm/config/tasks/REALM_DROID100 --profile authored
```

`stale` counts tasks made against an older catalogue; `next` offers them again. To redo one task, run `start … --restart`.

## Generate tasks from your own instructions

DROID is the default source, not a requirement. Any sentence works, as long as it describes one main object and a goal REALM can score (put, pick, stack, rotate, push a switch, open or close a drawer).

1. Write one instruction per line:

   ```text
   Put the banana in the bowl
   Take the spoon out of the mug
   Stack the plate on the tray
   ```

2. Run any driver on that file, with your own ranking id and output folder so the DROID family is untouched:

   ```sh
   uv run python -m tooling.task_authoring.agent.run my_tasks.txt --model <model-id> \
       --ranking-id my-tasks-v1 --output realm/config/tasks/MY_TASKS
   ```

   Or ask a coding agent to follow `GENERATE_TASKS.md` for `my_tasks.txt`, then run `agent.cli export --output realm/config/tasks/MY_TASKS`.

3. Validate and physics-check the folder exactly as for the DROID family.

What to expect: each task is placed on an existing REALM tabletop with real catalogue assets and up to three distractors. It is not a newly generated room. Instructions that need an asset the catalogue doesn't have, several objects at once, or a motion REALM cannot score are declined with a reason. See [Limitations](limitations.md).

## Run the physics check (GPU)

Try three configs first and confirm you get a real PASS/FAIL split before sweeping the family:

```sh
./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py \
    --task_cfg_path REALM_DROID100/<task>/default.yaml --out tmp/droid100/review

./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py \
    --family REALM_DROID100 --out tmp/droid100/review
```

Each task writes `tmp/droid100/review/<task>.json` (verdict, findings, measured drift) plus camera images before and after settling. `--family` writes `_summary.json` with the tally.

Three things that trip people up:

- **Read the verdict in the JSON, never the exit code.** Isaac Sim exits 0 after unhandled exceptions and can segfault at teardown on a passing run.
- **There is no `uv` inside the container.** Call `python -u` directly.
- **Never run `pytest tests/`.** Test collection boots Isaac Sim.

## Evaluate a policy (GPU)

A generated task is a normal REALM task config, so the standard entry point takes it via `--task_cfg_path`:

```sh
OMNIGIBSON_HEADLESS=1 python /app/examples/02_evaluate.py \
    --task_cfg_path REALM_DROID100/put_the_marker_in_the_cup_d31114/default.yaml \
    --model_type <type> --model_name <name> --port <port>
```

See [Running evaluations](../../../wiki/Running-Evaluations.md) for every flag, including perturbations and repeats.

## Run the tests

```sh
uv run ruff check realm examples tests scripts tooling
uv run python -m pytest -q tooling/task_authoring/test_validation.py \
    tooling/task_authoring/test_authoring.py tooling/task_authoring/agent/test_agent.py \
    tooling/task_authoring/agent/test_catalog_cli.py
```

These run on the host and build their own synthetic asset tree, so they don't need the dataset.

## Regenerate the figures

Every chart in these docs is computed from the pipeline's own output files:

```sh
python3 tooling/task_authoring/docs/make_figures.py
python3 tooling/task_authoring/docs/make_figures.py --render-dir tmp/droid100/review   # adds the physics-check chart
```

This writes a light and a dark SVG of each figure into `docs/assets/`, and prints the numbers it used.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| every instruction is declined | catalogue missing or built from the wrong path | `agent.catalog show`; rebuild with the right `--dataset` |
| `no session <id>; run start first` | the session lives in a different `--sessions` folder | pass the same `--sessions` you used for `start` |
| `stale` tasks in `status` | the catalogue changed since they were made | run `next`; they are offered again |
| render JSON says `NO_JSON` or `ERROR` | the child process crashed before writing | rerun that one task with `--task_cfg_path` and read its log |
| git commands fail with `File exists` | an earlier git process left a `.lock` file | make sure no git process is running, then remove the lock |
