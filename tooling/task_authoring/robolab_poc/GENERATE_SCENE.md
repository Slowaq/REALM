# Generating a REALM scene with the RoboLab-style PoC

Instructions for a coding agent (or a person) turning a natural-language request into a REALM task
config. You write a small JSON spec. Code places every object and writes the YAML.

All commands run from the repository root:

```sh
uv run python -m tooling.task_authoring.robolab_poc.cli <command> ...
```

## What you decide, and what you must not decide

You decide:
- the instruction text,
- the task type,
- which object categories appear and the role of each,
- the spatial relations between objects.

You never write a coordinate, distance, size, scale or rotation. The spec validator rejects them.
The solver works out every number.

## Steps

1. **Get the vocabulary.** Run `cli regions`. It lists the support surfaces you can choose from and
   the relation words. The relations are robot-relative:
   - `left-of`, `right-of`: the robot's left and right.
   - `in-front-of`: closer to the robot. `behind`: farther from the robot.
   - `in`: the object starts inside a `container`.
   - `on`: the object starts on top of a `support`.
2. **Ground every noun.** Run `cli search <noun>` for each one. A noun with no hit does not exist
   in the catalogue. Do not invent a category or quietly swap one in. Either:
   - drop the request and say it can't be grounded, or
   - pick the closest honest category and rewrite the instruction so it names that object
     (for example, "bowl", not "sink").
3. **Write the spec** to `tooling/task_authoring/robolab_poc/specs/<name>.json`, following the
   examples next to it. The rules:
   - `task_type` is one of `put`, `pick`, `rotate`, `stack`.
   - Exactly one object has role `main`.
   - `put` and `stack` need exactly one `target`. `pick` and `rotate` have none.
   - A `source` object is the thing the main object starts in or on, as in "take the lemon out of
     the bowl". It needs an `in` or `on` relation from `main`.
   - Other objects are `distractor`s. Three is a good number; regions are only about 0.4 × 0.5 m.
   - The main object's noun, `name` with `_` replaced by a space, must appear word for word in the
     instruction. So must the target's. If the instruction uses a different word, set `noun` on the
     object. Set `verb` when the instruction's verb isn't the task type, e.g. `"take"` for a pick.
   - An object that is `in` or `on` something can't also be the subject or reference of a
     left/right/front/behind relation.
4. **Dry run.** Run `cli solve specs/<name>.json`. If `ok` is true, continue. If not, read the
   `findings` and change the spec. Don't change the code. The usual fixes:
   - `RELATION_BROKEN` or `OVERLAP` after all attempts: the relations don't fit the region. Drop a
     distractor or a relation, or choose a larger region.
   - `CONTAINER_MOUTH`: the object is too big for the container's opening. Choose a smaller object
     or a bigger container.
   - `INSTRUCTION_*_TOKEN`: the noun or verb isn't in the instruction. Fix `noun`, `verb` or the
     instruction.
   - `BBOX_NOT_UPRIGHT` (warning): the only known size for that model comes from a config that
     tips it over. Prefer another model, or flag the object for render review.
   Stop after three revisions and report what failed.
5. **Look at it.** Run `cli plot specs/<name>.json`, then open `tmp/robolab_poc/<name>.png`. The
   plot is drawn from the robot's point of view. Check that left, right, front and behind read the
   way the instruction intends.
6. **Emit.** Run `cli emit specs/<name>.json`. This writes
   `realm/config/tasks/ROBOLAB_POC/<name>/default.yaml` and `poc_provenance.json` next to it.
7. **Report** the `task_cfg_path`, the seed that solved, the number of attempts it took, and any
   warnings.
