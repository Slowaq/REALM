# Vendored: RoboLab predicate scene solver

These files are from NVIDIA's RoboLab (https://github.com/NVLabs/RoboLab),
`robolab/scene_gen/llm_scene_gen/`, at commit `ad45d4f` (2026-09-12). They are
licensed under Apache-2.0, and the license text is in `LICENSE` in this folder.

The files are **unmodified**. Any REALM-specific behaviour lives outside this folder,
in `../../scene.py`. That includes the coordinate frames, the heights above the support,
the yaw limits and the seeding. Keeping the files unmodified lets the PoC test RoboLab's
solver as it is published.

The RoboLab Claude Code skills (`skills/robolab-scenegen`, `skills/robolab-taskgen`)
are CC-BY-NC-4.0 and are not copied here. `../GENERATE_SCENE.md` was written
independently.
