# Limitations and roadmap

[← Back to overview](../README.md) · [How it works](how-it-works.md) · [Walkthrough](walkthrough.md) · [Usage](usage.md) · [Results](results.md) · [Reference](reference.md) · **Limitations**

## What it can't do (yet)

| Limitation | Why | Consequence |
|---|---|---|
| **No new rooms** | tasks are placed on six existing REALM tabletops | the scene varies by table, objects and clutter, not by room |
| **One main object per task** | REALM's rubrics score `main_objects[0]` | "put the cups in the sink" is declined |
| **Fixed success rubrics** | scoring comes from `task_progressions.yaml` by `task_type` | no "move it to the left", "pour", "stir", "flip"; a new rubric is a code change |
| **Catalogue assets only** | assets come from BEHAVIOR-1K, so REALM's noun and affordance perturbations keep working | no ring-stacking toys, no deformable cloth, no crumpled paper |
| **No colour for mesh assets** | textures are fixed and not described in the catalogue | colour words are dropped, except on primitive blocks |
| **Validation is a proxy** | bounding boxes can't see mesh interiors | the GPU physics check is required before a task is trusted |
| **Mass and friction are not authorable** | OmniGibson ignores the keys silently | `validation.py` rejects them |
| **One DROID chunk** | 1,000 of DROID's episodes were surveyed | the instruction pool is small and has near-duplicates |
| **Vision review not wired** | `agent/review.py` exists as a tested library with no CLI | render failures are fixed through rules and replay, not an automated loop |

## What's next

1. **Measure the current family in the simulator.** Run a full physics check with the current harness and publish the chart (`make_figures.py --render-dir`).
2. **Freeze `REALM_DROID100` v1.** Write a manifest with seed, prompt version and catalogue fingerprint, and decide which perturbations the family runs.
3. **Evaluate policies.** Run π0, π0.5 and GR00T on the family, and check whether their ranking agrees with REALM_DROID10 and, if possible, with real-robot results.
4. **Baseline the design rule.** Compare against generation where the model writes coordinates (as in RoboLab): measure collisions, settle drift and reproducibility.
5. **Scale and audit.** Use more DROID chunks, run with named API models and repeated seeds, and have several people rate how faithful each task is to its instruction.
6. **Close the loop.** Add the review CLI (render JSON → findings → deterministic fixes, at most two rounds).

## Related work

| Work | Relation to this tool |
|---|---|
| [REALM](https://arxiv.org/abs/2512.19562) | the benchmark these tasks extend: real-to-sim validated, 15 perturbation types |
| [RoboLab](https://arxiv.org/abs/2604.09860) | LLM-generated scenes and task code for DROID-style evaluation; the model places objects directly |
| [GenManip](https://awesomepapers.io/robotics/papers/2506.10966) | LLM scene graphs generate large task sets, with human correction |
| [SceneSmith](https://arxiv.org/abs/2602.09153) | agents generate whole rooms with new assets; open-ended rather than reproducible |
| [SimFoundry](https://arxiv.org/abs/2606.28276) | builds simulation scenes from real video, a complementary source of tasks |
