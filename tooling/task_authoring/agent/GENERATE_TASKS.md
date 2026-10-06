# Generating REALM_DROID100 tasks with a coding agent

Hand this file to any shell-capable coding agent from the repo root:

> Follow `tooling/task_authoring/agent/GENERATE_TASKS.md` for the next 20 instructions in
> `data/DROID100_tabletop.json`.

The agent becomes the generator. It does the semantics (which task, which objects, which relation).
The deterministic solver does all the geometry. There's no API key or SDK. Every action is one
shell command, so the whole run is reviewable afterwards.

## 0. Once per session

```sh
python -m tooling.task_authoring.agent.cli prompt        # read this. It is the contract.
python -m tooling.task_authoring.agent.catalog show      # which asset catalogue you are grounding against
```

If `show` reports `source=seed-from-configs`, you only have about 35 categories. Expect honest declines
for pots, boxes, towels and tape. Don't substitute your way around a missing category. A decline
is the correct answer, and building the full catalogue fixes it (see the README).

## 1. Loop, one instruction at a time

```sh
python -m tooling.task_authoring.agent.cli next data/DROID100_tabletop.json
# -> prints the instruction, its task_id and the exact `start` command. Run it.
python -m tooling.task_authoring.agent.cli start "<instruction>" --ranking-id <id> --rank <n>
```

Then call tools until the task ends:

```sh
T=<task_id>
python -m tooling.task_authoring.agent.cli call $T search_assets '{"query":"marker","role":"main"}'
python -m tooling.task_authoring.agent.cli call $T list_scene_regions
python -m tooling.task_authoring.agent.cli call $T propose_layout '{
  "task_type":"put","instruction":"Put the marker in the mug",
  "main":{"name":"marker","category":"marker","model":"<from search>"},
  "target":{"name":"mug","category":"mug","model":"<from search>"},
  "decisions":["cup grounded as mug; instruction rewritten to name it"]}'
python -m tooling.task_authoring.agent.cli call $T validate_draft
python -m tooling.task_authoring.agent.cli call $T submit_task '{"decisions":["..."]}'
#   or
python -m tooling.task_authoring.agent.cli call $T report_ungroundable '{"reason":"...","blocking_terms":["pot"]}'
```

Use these roles:

- `pick` with "from X" means X is the `source`, plus `initial_state: [{predicate: inside|on_top_of, subject: main, object: X}]`.
- `put` needs a `target` receiver. `stack` needs a `target` support.
- "... and put it on the table" is a removal (`pick`). The table is the support, not an object.

- A coloured block or cube is not a catalogue asset. Author it as a primitive:
  `{"name":"yellow_block","primitive":"block","rgba":[0.9,0.8,0.1,1]}`. Its colour is guaranteed, so
  "Put the yellow block in the bowl" is groundable. Don't decline it for a missing block category.
- Omit `region_index` unless the instruction needs a specific surface. The solver then picks the
  scene from this task's seed, which spreads the family across scenes.
- Leave `distractors` empty unless the instruction implies specific clutter. The solver then samples
  plausible distractors from the catalogue with this task's own seed, so clutter varies across the
  family instead of repeating the same three objects.
- Substituting a different object (a frying pan for a pot) changes the task. Only do it when the
  stand-in plays the same role, rewrite the instruction to name it, and record why in `decisions`.
  When the full catalogue arrives, tasks made against the seed catalogue show as `stale` and `next`
  offers them again.

### Decline, don't bend

Decline when:

- a noun the instruction depends on has no honest asset,
- the instruction manipulates several objects ("two cups", "them", "the pile"),
- it asks for a motion REALM can't score ("move it to the left", "put it backwards"),
- three validation rounds fail.

`INSTRUCTION_PHANTOM_OBJECT` means the instruction names something the scene doesn't contain.
Either add that object as a role or rewrite the instruction. Don't ignore it.

### Never

- Never write or edit a YAML or session file by hand.
- Never put a number in metres, a quaternion or a scale into any argument. The tools reject coordinates on purpose.
- Never swap in an unrelated asset just to get a "grounded" result.

## 2. When the batch is done

```sh
python -m tooling.task_authoring.agent.cli status data/DROID100_tabletop.json
python -m tooling.task_authoring.agent.cli export        # writes realm/config/tasks/REALM_DROID100/, deduplicated
python -m tooling.task_authoring.validation realm/config/tasks/REALM_DROID100
```

`export` collapses instructions that solve to the same task, for example "marker in the cup" and
"marker in the mug". The report goes to `tmp/droid100/interactive_report.json`.

Generated configs are not yet benchmark tasks. They still need the container render and probe pass
(`render_review.py`) before anyone measures a policy on them.
