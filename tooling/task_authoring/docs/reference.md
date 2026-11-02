# Reference

[← Back to overview](../README.md) · [How it works](how-it-works.md) · [Walkthrough](walkthrough.md) · [Usage](usage.md) · [Results](results.md) · **Reference** · [Limitations](limitations.md)

## Files

| Path | Stage | Role |
|---|---|---|
| [`select_droid100.py`](../select_droid100.py) | 1 | DROID instructions → ranked task clusters, with rejection reasons |
| [`agent/catalog.py`](../agent/catalog.py) | 2 | build / show the asset catalogue; word-boundary noun matching; phantom-noun check |
| [`agent/tools.py`](../agent/tools.py) | 2 | the six tool schemas and the per-instruction session |
| [`agent/prompts.py`](../agent/prompts.py) | 2 | generator and review prompts, `PROMPT_VERSION` |
| [`agent/cli.py`](../agent/cli.py) | 2, 5 | the tools as shell commands; `next`, `start`, `call`, `status`, `export` |
| [`agent/run.py`](../agent/run.py) | 2, 5 | offline and API drivers |
| [`agent/layout.py`](../agent/layout.py) | 3 | roles → solved layout; the boundary between meaning and geometry |
| [`generate_realm_droid100.py`](../generate_realm_droid100.py) | 3 | placement solver shared with the original batch generator |
| [`validation.py`](../validation.py) | 4 | executable authoring rules; also a CLI over a folder of configs |
| [`render_review.py`](../render_review.py) | 6 | physics check in the simulator (GPU) |
| [`agent/review.py`](../agent/review.py), [`agent/corrections.py`](../agent/corrections.py) | 6–7 | vision-review loop and correction store (library; no CLI yet) |
| [`dashboard.py`](../dashboard.py) | — | manual authoring UI, see [dashboard.md](dashboard.md) |
| [`docs/make_figures.py`](make_figures.py) | — | regenerates the charts in these docs |

Deeper design documents: [`AGENTIC_PIPELINE.md`](../AGENTIC_PIPELINE.md) (original design and YAML schema), [`AUTHORING_RULES.md`](../AUTHORING_RULES.md) (the geometry rules in prose), [`agent/README.md`](../agent/README.md) (operator notes), [`agent/GENERATE_TASKS.md`](../agent/GENERATE_TASKS.md) (the brief a coding agent follows).

## CLI commands

`python -m tooling.task_authoring.agent.cli [--catalog FILE | --dataset DIR] [--sessions DIR] <command>`

| Command | Purpose |
|---|---|
| `prompt` | print the generator brief (the contract the model works under) |
| `next <instructions>` | show the next instruction without a finished session, and the exact `start` command |
| `start "<instruction>" --ranking-id ID --rank N [--seed 100] [--restart]` | open a session |
| `call <task_id> <tool> ['<json>']` | call one of the six tools; read `ok` in the printed JSON |
| `status [<instructions>]` | counts of submitted / declined / open / stale |
| `export [--output DIR] [--keep-duplicates] [--dry-run]` | write YAML configs and the run report |

`python -m tooling.task_authoring.agent.run <instructions> [--offline | --model ID] [--limit N] [--output DIR] [--seed 100] [--ranking-id ID] [--report FILE] [--no-cache] [--no-write]`

## Task types

REALM scores each task with a fixed stage rubric chosen by `task_type`. A generated task picks one and cannot invent a new success condition.

| Type | Rubric stages | Needs |
|---|---|---|
| `put` | reach, grasp, lift slightly, move close, place **into** | a receiver (`target`) |
| `stack` | reach, grasp, lift slightly, move close, place **onto** | a support (`target`) |
| `pick` | reach, grasp, lift high | optionally a `source` it starts in or on |
| `rotate` | reach, grasp, rotated | — |
| `push` | reach, touch, toggled on | a switch or button |
| `open_drawer` / `close_drawer` | drawer joint stages | an articulated cabinet |

## Validation codes

Errors block a task; warnings are reported.

| Code | Checks that… |
|---|---|
| `SCHEMA_MISSING_KEY`, `UNKNOWN_KEY`, `BBOX_INVALID`, `ORIENTATION_INVALID`, `QUAT_NOT_NORMALIZED`, `OBJECT_TYPE_UNKNOWN`, `NAME_DUPLICATE` | the config is well formed and has no keys OmniGibson would silently ignore |
| `TASK_TYPE_UNKNOWN`, `TARGET_STATE_RUBRIC`, `ROLE_CARDINALITY` | the task type exists, matches its rubric, and has one main object |
| `SCENE_MISSING`, `SCENE_REGION_UNKNOWN`, `SCENE_REGION_UNSAFE` | the table exists and is a reliable support |
| `SUPPORT_PENETRATION`, `SUPPORT_CLEARANCE`, `FLOATING_OBJECT`, `SUPPORT_CONTAINMENT`, `TIGHT_CLEARANCE` | objects start just above the table, inside its edges |
| `XY_OVERLAP`, `UNDECLARED_STACK` | objects don't start inside each other unless a relation says so |
| `RECEIVER_CAPACITY`, `LID_SMALLER_THAN_OPENING`, `RIM_OVERHANG` | containers and supports can actually hold what is put in or on them |
| `RELATION_GEOMETRY`, `RELATION_UNGROUNDED`, `RELATION_UNGRASPABLE`, `CONTAINMENT_ORIENTATION` | a declared starting relation holds geometrically and leaves the object graspable |
| `GOAL_SATISFIED_AT_START` | the task doesn't start already solved |
| `UPRIGHT_DISTRACTOR`, `NON_UPRIGHT_ORIENTATION`, `FIXED_BASE_ON_MOVABLE` | clutter stands upright; nothing the robot must move is fixed |
| `INSTRUCTION_CLOSURE`, `INSTRUCTION_GROUNDING`, `INSTRUCTION_PHANTOM_OBJECT` | the sentence names exactly the objects in the scene |

## Physics-check codes

| Code | Fails the task? | Meaning |
|---|---|---|
| `UNSTABLE` | yes | a task object moved more than 3 cm while settling |
| `PENETRATION` | yes | sunk more than 1.5 cm into the table, or below the floor |
| `TIPPED` | yes | a receiver or fixture tilted more than 35° |
| `LEFT_CONTAINER`, `LEFT_SUPPORT` | yes | left what it was declared to start in or on |
| `MISSING`, `DROPPED` | yes | object absent, or placement gave up |
| `DRIFT` | no | 1–3 cm settle, or clutter moving |
| `HIGH_DROP` | no | starts more than 6 cm above the surface |
| `FLOATING` | no | still above the surface after settling |
| `PHYSICS_FROZEN` | no | nothing moved at all (normal for hand-authored configs) |

The verdict is `PASS`, `FAIL` or `ERROR`. When the per-task JSON is missing or unreadable, `--family` reports `NO_JSON` or `BAD_JSON`.

## Glossary

| Term | Meaning |
|---|---|
| **Cluster** | all DROID rewordings of one task, merged in stage 1 |
| **Session** | the state of one instruction's generation, in `tmp/droid100/sessions/<task_id>.json` |
| **Main object** | the object the robot must manipulate (exactly one) |
| **Target** | the receiver (`put`) or support (`stack`) |
| **Source / immutable** | an object the main object starts in or on, e.g. the pot under a lid |
| **Distractor** | clutter; REALM's visual perturbations may move or swap it |
| **Region** | a rectangular tabletop in a REALM scene (`realm/config/scenes/scenes.yaml`) |
| **Decline** | `report_ungroundable`: the honest outcome when a task can't be represented |
| **Replay** | re-solving every submitted session after a rule changes, keeping the model's choices |
| **Prompt version** | a hand-bumped number identifying the generator's instructions; part of every task's provenance |
