# Agentic task generation for REALM_DROID100

Execution plan for scaling task authoring from the ten hand-built REALM_DROID10 tasks to ~100
generated ones, with a model in the loop for language grounding and scene review and deterministic
code in the loop for everything measurable.

Companion documents: [AUTHORING_RULES.md](AUTHORING_RULES.md) is the normative rule text;
`validation.py` is that text in executable form; this file is the system around them.

---

## 0. What already exists, and what is structurally missing

| Stage | Today | Verdict |
|---|---|---|
| Instruction selection | `select_droid100.py` ranks DROID phrasings, drops anything `CONCEPT_PATTERN` cannot parse | Works, but the regex is the throughput ceiling: every unparsed phrasing is silently discarded, and the discard rate is not reported per reason |
| Grounding | `CATEGORY_BY_CONCEPT` — 20 hand-written concept→category entries | Does not scale to 100 instructions; new nouns require a code edit |
| Layout | `place()`, `ensure_receiver_capacity()`, `place_initial_relation()` in `generate_realm_droid100.py` | Sound and deterministic. Keep as-is — this is the part a model must not do |
| Validation | Inline `raise ValueError` inside the generator | Now extracted to `validation.py`, so the agent, the vision loop, and CI check identical rules |
| Render review | Done once, by hand, results frozen into `REVIEWED_CAMERA_SOURCES` / `REVIEWED_MODEL_OVERRIDES` / `REVIEWED_POSITION_OVERRIDES` / `REVIEWED_TASK_OVERRIDES`, keyed by **rank** | The single biggest structural problem — see below |
| Physics authoring | No `mass`/`friction` anywhere in any task config | Unrepresentable today, and failures are silent — see §3.3 |

**The rank-keying problem.** The `REVIEWED_*` tables are the crystallised output of exactly the
vision-correction loop this plan formalises, but they are addressed by position in one frequency
ranking, guarded by `REVIEWED_RANKING_ID = "droid100-v1"`. The module comment already admits the
consequence: *"The corrected instructions do not record the originals they replaced, so these
cannot be re-keyed by instruction text after the fact."* Every re-run of `select_droid100.py`
against a different DROID sample throws away all accumulated review. A correction store keyed by
content, not rank, is a prerequisite for iterating (§4.5).

---

## 1. Architecture

### 1.1 The split rule

> **The model proposes semantics. Code disposes geometry.**
> A model may name an object, choose between candidate assets, assert a spatial *relation*, and
> read a rendered image. It may never emit a coordinate, an extent, a quaternion, a scale factor,
> or a random draw.

This is not stylistic. REALM's governing rule is that a change which moves a number is a bug; a
pipeline where an LLM emits coordinates cannot be regenerated reproducibly, cannot be diffed, and
cannot be audited against `AUTHORING_RULES.md`. Every number in a generated config must come from
`generate_realm_droid100.py`'s solver under a recorded seed.

### 1.2 Stages

| # | Stage | Owner | In | Out | Gate |
|---|---|---|---|---|---|
| S0 | Instruction selection | code | DROID parquet | `DROID100_tabletop.json` + rejection reasons | ranking_id recorded |
| S1 | Intent extraction | **model** | one instruction | `TaskIntent` JSON (verb, roles, predicates, clutter hints) | schema-valid, task_type in the closed set |
| S2 | Asset resolution | model *picks*, code *supplies* | `TaskIntent` | category + model id per role | every id exists in the indexed catalogue |
| S3 | Layout solve | code | resolved roles + scene region | positions, orientations, uniform scales | deterministic under seed |
| S4 | Emit + validate | code | layout | `default.yaml` | `validation.py` errors == 0 |
| S5 | Headless render | code (GPU) | YAML | 4 views × 2 timesteps + pose probe | sim builds, no crash |
| S6 | Vision review | **model** | images + the config it is looking at | findings with object names | schema-valid |
| S7 | Patch + re-validate | code | findings | patched YAML + correction record | validation still clean |
| S8 | Freeze | code | all of the above | `generation_manifest.json` | manifest hash pinned |

S5→S7 loops at most twice (the measured number of iterations that took the earlier run from 40/100
to 99/100). A task still failing after two iterations is escalated to a human or to the SimFoundry
path (§5), never silently shipped.

### 1.3 Artifacts on disk

```
data/droid/DROID100_tabletop.json               S0  selection + ranking_id
tooling/task_authoring/corrections/<task_id>.json   S7  the correction store (§4.5)
realm/config/tasks/REALM_DROID100/
├── <rank>_<slug>/default.yaml                  S4/S7
└── generation_manifest.json                    S8  seed, overrides, audit
tmp/droid100/renders/<task_id>/{cam1,cam2,top,front}_{t000,t060}.png   S5 (not committed)
tmp/droid100/review/<task_id>.json              S6 findings (not committed)
```

`task_id` is `sha1(instruction + "|" + ranking_id)[:12]` — stable across re-ranking, which is the
whole point.

### 1.4 Determinism contract

- One RNG per concern, seeded from `--seed`, exactly as the generator does today
  (`scene_rng`, `camera_rng`, `distractor_rng`). Adding a fourth consumer means adding a fourth
  stream, never reusing one, or every downstream draw shifts.
- The model's outputs are cached keyed by `(task_id, prompt_version, model_id)`. A re-run with an
  unchanged cache produces byte-identical YAML and performs no API calls.
- `prompt_version` is bumped by hand whenever a prompt changes, and recorded in the manifest. A
  family regenerated under a different prompt version is a different family.

---

## 2. The generator agent

### 2.1 Tool surface

Six tools. Note what is absent: there is no `set_position`, no `set_scale`, no `write_yaml`.

| Tool | Purpose | Returns |
|---|---|---|
| `search_assets` | category/keyword → candidate assets with natural bboxes | up to 12 candidates |
| `list_scene_regions` | available scene/support rectangles | region dicts |
| `propose_layout` | run the deterministic solver on a role assignment | positions + audit, or a failure reason |
| `validate_draft` | run `validation.py` | findings |
| `submit_task` | final structured answer | accepted / rejected with findings |
| `report_ungroundable` | give up honestly | recorded, instruction dropped |

`report_ungroundable` matters more than it looks. Without an explicit give-up path a model
substitutes a plausible-sounding asset rather than admit an asset does not exist — which is how
"orange-handled tool" became a silent mismatch that a human had to catch and hard-code as
`REVIEWED_TASK_OVERRIDES[71]`.

```python
# tooling/task_authoring/agent/tools.py
from tooling.task_authoring.authoring import discover_assets, load_scene_regions

SEARCH_ASSETS = {
    "name": "search_assets",
    "description": (
        "Search the indexed OmniGibson catalogue. Returns real assets only: a category absent "
        "from the result does not exist and must not be invented. Bounding boxes are the asset's "
        "natural extents in metres, XYZ, before any authoring scale."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "category name or noun, e.g. 'saucepot'"},
            "role": {"type": "string", "enum": ["main", "target", "source", "distractor"]},
            "max_footprint_xy": {
                "type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                "description": "optional XY ceiling in metres; candidates larger are excluded",
            },
        },
        "required": ["query", "role"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROPOSE_LAYOUT = {
    "name": "propose_layout",
    "description": (
        "Place the assigned roles with REALM's deterministic solver: support clearance, "
        "collision-free packing, receiver-capacity fit and any declared initial relation. "
        "You do not choose coordinates; you choose WHICH objects and WHAT relation holds "
        "between them. Returns the authored layout or the reason placement is impossible."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": ["put", "pick", "rotate", "push", "stack", "open_drawer", "close_drawer"],
            },
            "main": {"$ref": "#/$defs/asset_ref"},
            "target": {"$ref": "#/$defs/asset_ref"},
            "source": {"$ref": "#/$defs/asset_ref"},
            "distractors": {"type": "array", "items": {"$ref": "#/$defs/asset_ref"}, "maxItems": 4},
            "initial_state": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "predicate": {"type": "string", "enum": ["inside", "on_top_of"]},
                        "subject": {"type": "string"},
                        "object": {"type": "string"},
                    },
                    "required": ["predicate", "subject", "object"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["task_type", "main"],
        "additionalProperties": False,
        "$defs": {
            "asset_ref": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "model": {"type": "string"},
                },
                "required": ["name", "category", "model"],
                "additionalProperties": False,
            },
        },
    },
    "strict": True,
}
```

`propose_layout` dispatches straight onto the existing functions — `place()`,
`ensure_receiver_capacity()`, `place_initial_relation()`, `bbox_fits_support()` — so the agent path
and the batch path cannot drift apart. Wire `validate_draft` to
`tooling.task_authoring.validation.validate` and return `[asdict(f) for f in report.findings]`
verbatim: the `fix` field gives the model the exact correction, which keeps it out of arithmetic.

### 2.2 System prompt

Store as `tooling/task_authoring/agent/prompts.py::GENERATOR_SYSTEM` with a `PROMPT_VERSION`
constant beside it.

```text
You ground one natural-language robot-manipulation instruction into a REALM simulation task
config. REALM is a benchmark: the configs you produce are measured, so a config that is subtly
wrong is worse than one you refuse to produce.

## What you decide, and what you must not

You decide: the task type, which real assets play which role, what spatial relation holds at the
start, which clutter belongs on the table, and how the instruction should read.

You never decide: positions, orientations, bounding boxes, scale factors, or camera poses. Call
propose_layout and use what it returns. If you find yourself about to write a number in metres,
you have taken over a tool's job.

## Task types

Exactly one of: put, pick, rotate, push, stack, open_drawer, close_drawer. Each scores a fixed
stage rubric, and the instruction may promise only what its rubric checks:

  put     REACH GRASP LIFT_SLIGHT MOVE_CLOSE PLACE_INTO   — needs a receiver
  stack   REACH GRASP LIFT_SLIGHT MOVE_CLOSE PLACE_ONTO   — needs a support
  pick    REACH GRASP LIFT_LARGE                          — lifting/removal only
  rotate  REACH GRASP ROTATED
  push    REACH TOUCH TOGGLED_ON

"Pick up the sponge and put it on the counter" is not a pick task. Either shorten it to the
removal clause, or author it as put with the counter object grounded. Never let an instruction
promise a second stage the rubric cannot see.

## Grounding rules

1. Every noun the instruction depends on must exist as an object in the scene. "Take the lid off
   the pot" needs both a lid and a pot, even though only the lid is manipulated: the pot is an
   immutable with an on_top_of relation.
2. Only assets search_assets returns exist. If the right asset is absent, pick the closest honest
   substitute AND rewrite the instruction to name the substitute. A bowl standing in for a sink is
   a bowl in the instruction. If nothing honest is close, call report_ungroundable.
3. Never split a compound noun into two roles. "The orange object tool" is one tool.
4. One main object. An instruction about "all the cups" cannot be represented; report it.
5. The instruction must name properties the config guarantees. Colour words are only allowed for
   PrimitiveObject assets, whose rgba you can see, or where the category name itself carries the
   colour. Do not describe an asset's texture.

## Physical rules

6. Distractors stand upright as they would on a real table. A bottle is never authored on its
   side.
7. A distractor is plausible tabletop clutter, is not the main/target/source category, and adds
   visual variety: at most one member of a product family (bottle_of_*, jar_of_*, can_of_*).
8. A receiver must be able to hold the main object. propose_layout enforces this and will yaw or
   uniformly shrink the main object; if it reports failure, choose a larger receiver or a smaller
   main object rather than arguing with it.
9. An elongated object (a marker, a pen, a utensil) declared `inside` a container goes in
   long-axis-vertical. State the relation; the solver applies the orientation.
10. Anything the policy must move is free-floating. Only a fixture whose JOINT is the task —
    a drawer, a switch — is fixed_base, and only for open_drawer/close_drawer/push.

## Procedure

Resolve the task type, then the roles, then search for each asset, then call propose_layout, then
validate_draft. If validate_draft returns errors, read the `fix` field: it contains the exact
correction. Apply it through the tool that owns it and validate again. Submit only a draft that
validates clean. Three failed validation rounds means the instruction is not groundable — say so
rather than submitting something that merely passes.

When you substitute an asset, shorten an instruction, or decline, record the reason in the
`decisions` field of submit_task. That text is stored with the task and is what a later
regeneration reads to avoid re-making a rejected choice.
```

### 2.3 Driver

```python
# tooling/task_authoring/agent/run.py
import anthropic
from anthropic import beta_tool

from tooling.task_authoring.agent.prompts import GENERATOR_SYSTEM, PROMPT_VERSION

client = anthropic.Anthropic()
MODEL = "claude-opus-5"


def generate_task(instruction: str, context: dict) -> dict:
    """One instruction -> one validated draft. Cached on (task_id, PROMPT_VERSION, MODEL)."""
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[{"type": "text", "text": GENERATOR_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[search_assets, list_scene_regions, propose_layout, validate_draft,
               submit_task, report_ungroundable],
        messages=[{"role": "user", "content": render_task_brief(instruction, context)}],
    )
    return collect_submission(runner.until_done())
```

Two deliberate choices. The system prompt carries a `cache_control` breakpoint and the volatile
per-task brief goes in the user turn, so 100 runs share one cached prefix. And the catalogue is
**not** pasted into the prompt — it is behind `search_assets`, because a 100-task run against a
pasted catalogue pays for it 100 times and still cannot guarantee the model only names real
models.

For the full 100-task sweep use the Batch API (50% cost, results keyed by `custom_id`) once the
prompt is stable; keep the interactive tool-runner path for development.

---

## 3. YAML schema

### 3.1 Annotated template

```yaml
# ---- REALM task config. Consumed by realm/environments/env_config.py ----
task:                              # unused by REALM scoring; OmniGibson requires the block
  type: "DummyTask"
  termination_config: {}
  reward_config: {}

task_type: "put"                   # closed set; must have a rubric in task_progressions.yaml
instruction: "put the banana in the box"
instruction_obj_to_replace: "banana"   # substring of `instruction`; SB-NOUN/S-LANG rewrite it
instruction_target_to_replace: "box"   # required for put/stack
instruction_verb_to_replace: "put"     # SB-VRB rewrites this

supported_scenes:                  # scene -> [support region]; both must exist in scenes.yaml
  Pomaria_1_int:
    - "Kitchen_Counter"

camera_extrinsics:                 # named pair, or inline pose dicts from the DROID extrinsics
  cam1: "droid_realm_ep_060817_cam1"
  cam2: "droid_realm_ep_060817_cam2"

reset_joint_pos: [...]             # optional; falls back to the scene's, then the default

# ---- Roles. Exactly one main object; at most one target. ----
main_objects:
  - type: "DatasetObject"
    name: "banana"                 # unique across ALL roles (OmniGibson names are per-scene)
    category: "banana"
    model: "vvyyyv"
    bounding_box: [0.15, 0.09, 0.03]    # EXTENT, not a half-extent, not a scale
    orientation: [0.0, 0.0, 0.0, 1.0]   # XYZW quaternion; yaw-only unless reviewed
    relative_bbox_position: [0.20, 0.15, 0.065]
    # ^ SCENE-frame offset from the spawn region's min corner; z is bbox-centre height above
    #   the support. Convention: bbox_height/2 + 0.050. World-frame values only look right in
    #   scene 0 of a vector build.

target_objects:                    # the receiver (put) or the support (stack)
  - type: "DatasetObject"
    name: "box"
    category: "tray"
    model: "xzcnjq"
    bounding_box: [0.23, 0.35, 0.05]
    orientation: [0.0, 0.0, 0.0, 1.0]
    relative_bbox_position: [0.40, 0.30, 0.075]

distractors:                       # clutter; re-placed by VB-POSE, replaced by VSB-NOBJ
  - type: "DatasetObject"
    name: "distractor_orange"
    category: "orange"
    model: "ucstpm"
    bounding_box: [0.08, 0.09, 0.08]
    orientation: [0.0, 0.0, 0.0, 1.0]
    relative_bbox_position: [0.15, 0.45, 0.090]

immutables:                        # authored fixtures: a source vessel, a support, a light.
  - type: "DatasetObject"          # V-SC leaves these alone; they are NOT swapped or re-placed
    name: "pot"
    category: "saucepot"
    model: "abcdef"
    bounding_box: [0.22, 0.22, 0.14]
    orientation: [0.0, 0.0, 0.0, 1.0]
    relative_bbox_position: [0.30, 0.30, 0.120]

# ---- NEW: authoring-tool blocks. Ignored by env_config.py; read by validation.py ----
initial_state:                     # declares a relation the geometry must satisfy
  - predicate: "on_top_of"         # inside | on_top_of
    subject: "lid"
    object: "pot"

target_state:                      # DOCUMENTATION of the success condition. REALM scores from
  rubric: "put"                    # task_progressions.yaml, NOT from this block -- see 3.4
  predicate: "inside"
  subject: "banana"
  object: "box"

provenance:                        # what a regeneration must not silently undo
  task_id: "a3f19c2b77e1"
  ranking_id: "droid100-v1"
  original_instruction: "put the banana into the box on the table"
  decisions:
    - "Shortened: REALM pick/put rubrics do not score a second placement stage."
  prompt_version: 3
  generated: "2026-09-14"

cached_semantic_perturbations:     # 10 rewrites each for S-LANG/S-AFF/S-INT/S-PROP/S-MO
  S-LANG: ["place the banana inside the box", ...]
```

### 3.2 Who consumes what

| Key | Consumed by | Note |
|---|---|---|
| `main_objects`/`target_objects`/`distractors`/`immutables` | `env_config._apply_object_cfg` | Concatenated into one OmniGibson `cfg["objects"]` list |
| `relative_bbox_position` | same | Mirrored in x when the robot yaw is 90–270°, then offset by `spawn_bbox` |
| per-object keys | OmniGibson constructors | **Anything unrecognised is swallowed by `**kwargs`** |
| `instruction_*_to_replace` | the S-/SB- perturbations | A token absent from `instruction` makes the perturbation a silent no-op |
| `initial_state`, `target_state`, `provenance` | authoring tools only | Safe to add: `env_config` indexes keys by name |

### 3.3 Physical constraints: what actually reaches the simulator

Verified against OmniGibson 3.9.1 (`omnigibson/objects/dataset_object.py`,
`prims/entity_prim.py`):

| Property | How to author it | Status |
|---|---|---|
| Fixed base | `fixed_base: true` | Works. Only for fixtures whose joint is the task |
| Kinematic (moved, never pushed) | `kinematic_only: true` | Works |
| Friction / restitution | `link_physics_materials: {base_link: {static_friction: 0.9, dynamic_friction: 0.8}}` | Works — kwargs go to Isaac's `PhysicsMaterial` |
| Self-collision | `self_collisions: true` | Works |
| **Mass / density** | — | **No config path.** `DatasetObject(load_config=...)` is the *entity's* load config; `EntityPrim` builds each link's config from a fixed key list (`remesh`, `xform_props_pre_loaded`, `scale`, `visual_only`) and drops everything else, so the `mass`/`density` keys `RigidPrim` reads are never populated |

A `mass:` key in a task YAML therefore does nothing at all, and OmniGibson raises no error —
`**kwargs` absorbs it. `validation.py` rejects it with `UNKNOWN_KEY` for exactly this reason.

To make mass authorable, apply it post-load in `SceneSetupMixin`:

```python
# realm/environments/scene_setup.py
def _apply_authored_masses(self, task_cfg):
    """Task configs may set `load_config: {mass: <kg>}`; OmniGibson only honours it post-load."""
    for role in ("main_objects", "target_objects", "distractors", "immutables"):
        for cfg in task_cfg.get(role) or []:
            mass = (cfg.get("load_config") or {}).get("mass")
            if mass is None:
                continue
            obj = self.scene.object_registry("name", cfg["name"])
            obj.root_link.mass = float(mass)
```

**This moves numbers.** Masses currently come from the asset metadata, so authoring one changes
the dynamics of that task. Land it as a `KNOWN ISSUE`-style gated change behind a `VERSION` bump,
and do not apply it to REALM_DROID10 configs without recomputing their results.

### 3.4 Success criteria: `target_state` is documentation

REALM does not read a per-task reward spec. `task: {type: DummyTask, reward_config: {}}` is inert;
scoring comes from `TaskProgressionMixin` walking the stage list that
`realm/config/tasks/task_progressions.yaml` keys by `task_type`, with the predicates implemented in
`environments/task_progression.py` against `main_objects[0]` and `target_objects[0]`.

So: **a generated task cannot invent a success condition.** It selects one of the existing rubrics.
`target_state` exists in the schema so the agent states what it believes success means and a
reviewer can compare that against the rubric the task_type actually runs — a check `validation.py`
performs (`target_state.rubric` must equal `task_type`). A genuinely new success condition is a
code change in `task_progression.py` plus a new rubric entry, and is out of scope for generation.

---

## 4. The vision review loop

### 4.1 Compute first, look second

Rendered review is the expensive, non-deterministic, hallucination-prone stage. Everything
computable must be computed before an image is shown to a model:

| Failure | Detector | Where |
|---|---|---|
| Object inside the support | bbox arithmetic | `validation.py` SUPPORT_PENETRATION |
| Floating object | bbox arithmetic | `validation.py` FLOATING_OBJECT |
| Objects spawned inside each other | AABB overlap | `validation.py` XY_OVERLAP |
| Distractor on its side | quaternion has roll/pitch | `validation.py` UPRIGHT_DISTRACTOR |
| Cover too small for the pot | capacity proxy | `validation.py` RECEIVER_CAPACITY |
| **Physical instability** | pose delta over 60 settling steps | S5 probe, below |
| **Mesh interpenetration** the outer bbox hides | pose delta + contact report | S5 probe |
| **Object hidden from both cameras** | pixel coverage of the object's segmentation id | S5 probe |
| Semantic mismatch (asset does not look like the noun) | — | vision only |
| Implausible arrangement a human would notice | — | vision only |

Sending 100 scenes to a vision model to catch a `z=0` that arithmetic catches for free is how a
review loop becomes expensive and flaky. Ask the model only the last two rows.

### 4.2 Render + probe harness (container, GPU)

```python
# tooling/task_authoring/render_review.py   -- runs inside the container
"""Build each generated task, let it settle, and emit review images plus a stability probe.

    ./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py \
        --family REALM_DROID100 --out tmp/droid100/renders

Physics instability is measured, not eyeballed: an object that moves more than SETTLE_TOL over
60 steps of an unactuated sim was authored in penetration or on an unstable base.
"""
SETTLE_STEPS = 60
SETTLE_TOL = 0.01          # metres of centre-of-mass travel with no policy acting

def review_one(task_cfg_path, out_dir, robot="DROID_mounted"):
    set_sim_config(robot=robot)
    vec = RealmVectorEnvironment(1, task_cfg_path=task_cfg_path, perturbations=["Default"],
                                 robot=robot, rendering_mode="rt")
    env = vec.envs[0]
    tracked = env.main_objects + env.target_objects + env.distractors
    before = {o.name: o.get_position_orientation()[0].clone() for o in tracked}
    save_views(env, out_dir, stamp="t000")          # cam1, cam2, + two diagnostic views
    for _ in range(SETTLE_STEPS):
        og.sim.step()
    after = {o.name: o.get_position_orientation()[0] for o in tracked}
    save_views(env, out_dir, stamp="t060")
    drift = {name: float((after[name] - before[name]).norm()) for name in before}
    return {"unstable": {n: d for n, d in drift.items() if d > SETTLE_TOL}, "drift": drift}
```

Two extra diagnostic cameras beyond the two DROID views — one top-down, one front elevation — are
what make "is this floating / is this inside that" answerable from an image at all. The DROID
camera pair alone is deliberately oblique.

Run it on the cluster as a Slurm array over the 100 configs, one container build per array task,
reusing the launch pattern in `scripts/cluster_evals/`.

### 4.3 The vision call

One task per request. Include the config the model is looking at, the computed probe result, and
the four images; ask only for what images can settle.

```python
REVIEW_SYSTEM = """You review a rendered simulation scene against the task config that produced it.

You are the last check before a benchmark task is frozen, and you are looking at four views of the
same scene at two timesteps: t000 is the authored pose, t060 is after 60 steps of physics with no
robot acting. Arithmetic checks have already passed, and a stability probe has already measured
which objects moved; that result is given to you. Do not re-report it.

Report only what the images show and the numbers cannot:

1. SEMANTIC_MISMATCH  - an asset does not depict the noun the instruction uses.
2. IMPLAUSIBLE_POSE   - an object stands in a way a person would not leave it, including a
                        distractor lying down, leaning, or balanced on an edge.
3. INTERPENETRATION   - meshes visibly pass through one another (bounding boxes can clear while
                        meshes do not).
4. RELATION_VIOLATED  - the instruction's required starting relation does not hold in the image:
                        the lid is beside the pot rather than on it, the marker is not in the cup.
5. OCCLUDED_TARGET    - the main or target object is not discernible in EITHER DROID camera view.
6. UNREACHABLE        - the main object sits where the arm's base plainly cannot reach.

For each finding name the exact object `name` from the config. If a fix is obvious and is one of
{make upright, move apart, swap asset, reduce size}, say which. Never give coordinates.

Report nothing when the scene is sound. A clean scene is the expected outcome, and an invented
finding costs a real correction round."""

FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "enum": [
                        "SEMANTIC_MISMATCH", "IMPLAUSIBLE_POSE", "INTERPENETRATION",
                        "RELATION_VIOLATED", "OCCLUDED_TARGET", "UNREACHABLE"]},
                    "object": {"type": "string"},
                    "evidence": {"type": "string", "description": "which view, what is visible"},
                    "suggested_action": {"type": "string", "enum": [
                        "make_upright", "move_apart", "swap_asset", "reduce_size",
                        "change_camera", "escalate"]},
                },
                "required": ["code", "object", "evidence", "suggested_action"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "findings"],
    "additionalProperties": False,
}

response = client.messages.parse(
    model="claude-opus-5",
    max_tokens=8000,
    thinking={"type": "adaptive"},
    system=[{"type": "text", "text": REVIEW_SYSTEM, "cache_control": {"type": "ephemeral"}}],
    output_config={"format": {"type": "json_schema", "schema": FINDING_SCHEMA}},
    messages=[{"role": "user", "content": [
        *image_blocks(render_dir),                       # 4 views x 2 timesteps
        {"type": "text", "text": review_brief(task_cfg, probe_result)},
    ]}],
)
```

### 4.4 Applying corrections

A suggested action is a *request to a deterministic transform*, never an edit the model writes:

| Action | Transform | Re-solve |
|---|---|---|
| `make_upright` | orientation → `[0,0,0,1]` | no |
| `move_apart` | drop the object, re-run `place()` with the remaining candidate slots | yes |
| `swap_asset` | next candidate from `search_assets` for that category, re-fit bbox | yes |
| `reduce_size` | one uniform scale step (0.85), never per-axis | yes |
| `change_camera` | re-draw from the camera pool using `camera_rng` | no |
| `escalate` | stop; human or SimFoundry | — |

After every applied patch, re-run `validation.py`. A correction that silences a vision finding
while breaking a geometric rule is the loop's characteristic failure mode, and it is exactly what
a cheap host-side re-validation catches. Then re-render; at most two iterations.

### 4.5 The correction store

This replaces the `REVIEWED_*` dictionaries. One JSON file per task, keyed by content:

```json
{
  "task_id": "a3f19c2b77e1",
  "instruction": "Remove the can from the sink",
  "ranking_id": "droid100-v1",
  "corrections": [
    {
      "iteration": 1,
      "source": "vision",
      "code": "SEMANTIC_MISMATCH",
      "object": "sink",
      "action": "swap_asset",
      "from": {"category": "sink", "model": null},
      "to": {"category": "bowl", "model": "jgethp"},
      "instruction_after": "Remove the can from the bowl",
      "reason": "No movable sink asset exists; the instruction must name the grounded asset.",
      "applied": "2026-09-14"
    }
  ]
}
```

Three properties the current tables lack: it is keyed by `task_id` (content), so a re-ranked
selection keeps its review; it records `from` as well as `to`, so a regeneration can detect that
the generator no longer makes the rejected choice and retire the correction; and `source`
distinguishes a measured correction from a model's judgement call. The generator loads the store
and applies corrections after layout, before validation, and lists every applied correction in
`generation_manifest.json`.

Migration: the four existing `REVIEWED_*` tables are transcribed into stores once, under
`ranking_id: droid100-v1`. The instruction text for each rank is recoverable from the current
manifest, so the `task_id` can be computed. Keep the tables in place, unused, for one release so
the transcription can be diffed.

---

## 5. Fallback: ingesting reconstructed scenes

Agentic generation is bounded by the asset catalogue. When an instruction needs an object or a
layout BEHAVIOR-1K does not carry — a specific appliance, a cluttered real desk — no amount of
prompting fixes it, and `report_ungroundable` fires. Video-to-scene reconstruction (SimFoundry or
equivalent) is the escape hatch for those.

**Trigger rule.** A task goes to the reconstruction path when it is (a) reported ungroundable, or
(b) still failing after two correction iterations, and (c) appears frequently enough in the DROID
ranking to be worth the cost. Everything else is dropped from the family and recorded as dropped —
a 92-task family with an honest manifest beats a 100-task family with eight bad configs.

**Adapter contract.** Reconstruction output varies, so isolate it behind one interface rather than
spreading assumptions:

```python
# tooling/task_authoring/ingest/reconstruction.py
@dataclass
class ReconstructedScene:
    meshes: list[Path]                  # per-object, metres, Z-up, origin at bbox centre
    poses: dict[str, tuple[list, list]] # name -> (position, XYZW quaternion), world frame
    support_plane: tuple[list, float]   # plane normal+offset, or an AABB for the table top
    cameras: dict[str, dict]            # name -> {position, orientation}, DROID frame
    labels: dict[str, str]              # name -> best-guess category
    provenance: dict                    # episode id, method, confidence
```

Bringing one into REALM, in order, with the parts that need verification against the real tool
marked:

1. **Units and up-axis.** Convert to metres, Z-up. `tests/test_scene_object_placement.py`
   documents what a mismatched `upAxis` costs: Kit appends a `unitsResolve` xform op that no
   OmniGibson pose setter strips, and it silently post-multiplies every pose — asymmetrically
   across vector members. Normalise at ingest; never at load. *(Verify the reconstruction's native
   convention.)*
2. **Mesh → USD**, one asset per object, with a convex-decomposition collision mesh; visual-only
   for anything the robot cannot touch. Reconstructed meshes are usually non-watertight, so the
   collision approximation, not the visual mesh, is what must be reviewed.
3. **Category assignment.** Map each label onto an existing REALM category where one fits, so the
   semantic perturbations (S-AFF, S-INT, SB-NOUN) keep working. A reconstructed object with no
   category is usable as an immutable but not as a main object.
4. **Registration into a spawn region.** Fit the reconstructed support plane to a
   `scenes.yaml`-style rectangle, translate every pose into `relative_bbox_position`, and emit the
   same YAML as §3.1. From here the object is indistinguishable from a catalogue object.
5. **Cameras.** Reuse `convert_droid_extrinsics.py` — the reconstruction's camera poses are in the
   DROID frame, which is the frame that script already targets. *(Verify the handedness.)*
6. **Same gates.** `validation.py`, then the render/probe, then vision review. A reconstructed
   scene earns no exemption; it is likelier to need step 2 reworked than a catalogue scene is.

Cost check before building this: steps 1–4 are roughly a week of work and are only worth it if
more than a handful of high-frequency instructions are ungroundable. Run stage S1 over the full
DROID ranking first — the count of `report_ungroundable` results, grouped by missing category, is
the number that decides whether this path is built at all.

---

## 6. Execution plan

| Phase | Work | Gate | Cost |
|---|---|---|---|
| **P0** ✅ | `validation.py` + `test_validation.py`; rules executable and tested | 14 tests green, lint clean | done |
| **P1** | Wire `validation.validate()` into `generate_realm_droid100.generate()` before each write; add `--strict` | Regenerating the current family produces byte-identical YAML | 0.5 d |
| **P2** | Correction store; transcribe the four `REVIEWED_*` tables; manifest records applied corrections | Regeneration with the store matches the tables' output exactly | 1 d |
| **P3** | Render + probe harness (`render_review.py`) + Slurm array | 100 configs render; instability list produced | 2 d, GPU |
| **P4** | Agent tools + prompts + driver; run S1–S4 over the DROID ranking | Groundable count and the ungroundable histogram | 2–3 d |
| **P5** | Vision review + patch application, two bounded iterations | ≥95/100 configs clean under `--profile generated` | 2 d, GPU |
| **P6** | Freeze: manifest, `ranking_id`, prompt_version; CI job runs validation over the family | Tier-1 CI green | 0.5 d |
| **P7** *(conditional)* | Reconstruction ingest, only if P4's histogram justifies it | — | ~1 w |

Verification, per `CLAUDE.md`'s two tiers:

```sh
# tier 1, host, no GPU -- must stay green
uv run ruff check realm examples tests scripts tooling
uv run python -m pytest -q tooling/task_authoring/test_validation.py \
    tooling/task_authoring/test_authoring.py
uv run python -m tooling.task_authoring.validation \
    realm/config/tasks/REALM_DROID100 --quiet --json tmp/suite/droid100.json

# tier 2, container, against a running allocation
./scripts/run_apptainer.sh python -u tooling/task_authoring/render_review.py --family REALM_DROID100
./scripts/run_apptainer.sh python -u tests/test_scene_object_placement.py \
    --task_cfg_path REALM_DROID100/001_.../default.yaml
```

Evaluating 100 tasks needs no change to `eval.py`: `resolve_task()` accepts a `--task_cfg_path`
outside `SUPPORTED_TASKS`, and `SUPPORTED_TASKS` must stay a ten-element literal — `eval.py`'s own
docstring and `tests/test_vector_integrity.py`'s `ast.parse` depend on it.

---

## 7. Risks

- **The benchmark rule.** REALM_DROID100 is a new family, so generating it moves no existing
  number. Two things would: the mass hook in §3.3, and any edit to shared placement code. Both are
  `VERSION`-gated changes, not generation work.
- **Validation is a proxy, not a proof.** Outer bboxes cannot see mesh interiors. `validation.py`
  says so in its own docstring; the settling probe is what closes most of that gap, and it must not
  be skipped because the arithmetic passed.
- **Vision review is not free of hallucination.** Hence: compute everything computable, give the
  model a `pass` verdict as the expected outcome, constrain findings to a closed code set, apply
  corrections through deterministic transforms, and re-validate after each patch. The measured
  40→99 improvement came from a loop with these properties; a loop where the model edits YAML
  directly does not have them.
- **Eval cost.** 100 tasks × 16 perturbations × repeats is ~10× the current sweep. Decide whether
  the DROID100 family runs the full perturbation matrix or a subset *before* committing cluster
  time; that decision belongs with the benchmark design, not with generation.
- **Prompt drift.** A prompt edit changes the family. `prompt_version` in the manifest, bumped by
  hand, is the only thing that makes an unreproducible family visible instead of silent.
