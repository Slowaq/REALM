"""Host-runnable coverage for the authoring rule gate.

    uv run python -m pytest -q tooling/task_authoring/test_validation.py

Each test pins one rule AND, where the rule implies a correction, the `fix` value the loop
applies, because a wrong fix is worse than no fix: it silences the finding without moving
the object.
"""
import unittest

from tooling.task_authoring.validation import (
    load_regions,
    load_task_types,
    validate,
)

REGIONS = {("TestScene", "TestTable"): {"width": 0.6, "depth": 0.6, "z": 0.8}}
TASK_TYPES = {"put", "pick", "rotate", "push", "stack", "open_drawer", "close_drawer"}


def obj(name, bbox, position, **extra):
    config = {
        "type": "DatasetObject",
        "name": name,
        "category": name,
        "model": "aaaaaa",
        "bounding_box": list(bbox),
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "relative_bbox_position": list(position),
    }
    config.update(extra)
    return config


def document(**overrides):
    base = {
        "task_type": "put",
        "instruction": "put the apple in the bowl",
        "instruction_obj_to_replace": "apple",
        "instruction_target_to_replace": "bowl",
        "supported_scenes": {"TestScene": ["TestTable"]},
        "main_objects": [obj("apple", [0.08, 0.08, 0.08], [0.15, 0.15, 0.09])],
        "target_objects": [obj("bowl", [0.20, 0.20, 0.07], [0.40, 0.40, 0.085])],
        "distractors": [],
        "immutables": [],
    }
    base.update(overrides)
    return base


def check(**overrides):
    return validate(document(**overrides), regions=REGIONS, task_types=TASK_TYPES)


def codes(report):
    return {finding.code for finding in report.findings}


def finding(report, code):
    return next(item for item in report.findings if item.code == code)


class ValidationTest(unittest.TestCase):
    def test_clean_draft_passes(self):
        report = check()
        self.assertTrue(report.ok, [item.line() for item in report.findings])

    def test_support_penetration_reports_the_corrected_z(self):
        report = check(distractors=[obj("can", [0.06, 0.06, 0.12], [0.30, 0.15, 0.0])])
        self.assertIn("SUPPORT_PENETRATION", codes(report))
        self.assertEqual(finding(report, "SUPPORT_PENETRATION").fix["value"], [0.30, 0.15, 0.11])

    def test_floating_object_is_an_error_without_a_declared_relation(self):
        report = check(distractors=[obj("can", [0.06, 0.06, 0.10], [0.30, 0.15, 0.40])])
        self.assertIn("FLOATING_OBJECT", codes(report))

    def test_declared_relation_exempts_the_stacked_object(self):
        report = check(
            task_type="pick",
            instruction="take the lid off the pot",
            instruction_obj_to_replace="lid",
            instruction_target_to_replace="",
            main_objects=[obj("lid", [0.16, 0.16, 0.02], [0.30, 0.30, 0.205])],
            target_objects=[],
            immutables=[obj("pot", [0.22, 0.22, 0.14], [0.30, 0.30, 0.12])],
            initial_state=[{"predicate": "on_top_of", "subject": "lid", "object": "pot"}],
        )
        self.assertTrue(report.ok, [item.line() for item in report.findings])

    def test_relation_geometry_catches_a_lid_sunk_into_the_pot(self):
        report = check(
            task_type="pick",
            instruction="take the lid off the pot",
            instruction_obj_to_replace="lid",
            instruction_target_to_replace="",
            main_objects=[obj("lid", [0.16, 0.16, 0.02], [0.30, 0.30, 0.10])],
            target_objects=[],
            immutables=[obj("pot", [0.22, 0.22, 0.14], [0.30, 0.30, 0.12])],
            initial_state=[{"predicate": "on_top_of", "subject": "lid", "object": "pot"}],
        )
        self.assertIn("RELATION_GEOMETRY", codes(report))
        self.assertEqual(finding(report, "RELATION_GEOMETRY").fix["value"], [0.30, 0.30, 0.21])

    def test_receiver_capacity_prefers_a_yaw_over_shrinking(self):
        report = check(
            main_objects=[obj("apple", [0.19, 0.06, 0.06], [0.15, 0.15, 0.08])],
            target_objects=[obj("bowl", [0.08, 0.24, 0.07], [0.40, 0.40, 0.085])],
        )
        self.assertIn("RECEIVER_CAPACITY", codes(report))
        self.assertEqual(
            finding(report, "RECEIVER_CAPACITY").fix,
            {"path": "main_objects[0].orientation", "value": [0.0, 0.0, 0.7071068, 0.7071068]},
        )

    def test_receiver_capacity_falls_back_to_a_uniform_scale(self):
        report = check(
            main_objects=[obj("apple", [0.19, 0.19, 0.06], [0.15, 0.15, 0.08])],
            target_objects=[obj("bowl", [0.10, 0.10, 0.07], [0.40, 0.40, 0.085])],
        )
        fix = finding(report, "RECEIVER_CAPACITY").fix
        self.assertEqual(fix["path"], "main_objects[0].bounding_box")
        ratios = [round(new / old, 5) for new, old in zip(fix["value"], [0.19, 0.19, 0.06])]
        self.assertEqual(len(set(ratios)), 1, f"scale must be uniform, got {ratios}")

    def test_sideways_distractor_is_rejected_with_an_upright_fix(self):
        bottle = obj("bottle", [0.07, 0.07, 0.22], [0.30, 0.15, 0.16])
        bottle["orientation"] = [0.7071068, 0.0, 0.0, 0.7071068]
        report = check(distractors=[bottle])
        self.assertIn("UPRIGHT_DISTRACTOR", codes(report))
        self.assertEqual(
            finding(report, "UPRIGHT_DISTRACTOR").fix["value"], [0.0, 0.0, 0.0, 1.0],
        )

    def test_overlapping_distractors_are_an_error_but_a_near_miss_is_a_warning(self):
        overlapping = check(distractors=[
            obj("can", [0.08, 0.08, 0.10], [0.30, 0.15, 0.10]),
            obj("jar", [0.08, 0.08, 0.10], [0.33, 0.15, 0.10]),
        ])
        self.assertIn("XY_OVERLAP", codes(overlapping))
        near = check(distractors=[
            obj("can", [0.08, 0.08, 0.10], [0.30, 0.15, 0.10]),
            obj("jar", [0.08, 0.08, 0.10], [0.385, 0.15, 0.10]),
        ])
        self.assertNotIn("XY_OVERLAP", codes(near))
        self.assertIn("TIGHT_CLEARANCE", codes(near))
        self.assertTrue(near.ok)

    def test_unknown_key_is_typed_per_object_class(self):
        report = check(distractors=[obj("can", [0.06, 0.06, 0.10], [0.30, 0.15, 0.10], mass=0.4)])
        self.assertIn("UNKNOWN_KEY", codes(report))
        usd = obj("cabinet", [0.10, 0.10, 0.10], [0.15, 0.45, 0.10])
        usd.pop("model")  # a DatasetObject key; on a USDObject it is silently ignored at load
        usd.update({"type": "USDObject", "usd_path": "/assets/cabinet.usd"})
        self.assertNotIn("UNKNOWN_KEY", codes(check(immutables=[usd])))

    def test_fixed_base_is_an_error_only_for_a_free_body_task(self):
        pinned = obj("apple", [0.08, 0.08, 0.08], [0.15, 0.15, 0.09], fixed_base=True)
        self.assertIn("FIXED_BASE_ON_MOVABLE", codes(check(main_objects=[pinned])))
        drawer = obj("drawer", [0.40, 0.40, 0.50], [0.30, 0.30, 0.30], fixed_base=True)
        articulated = check(
            task_type="open_drawer", instruction="open the drawer",
            instruction_obj_to_replace="drawer", instruction_target_to_replace="",
            main_objects=[drawer], target_objects=[],
        )
        self.assertNotIn("FIXED_BASE_ON_MOVABLE", codes(articulated))

    def test_instruction_tokens_must_be_substitutable(self):
        report = check(instruction_obj_to_replace="banana")
        self.assertIn("INSTRUCTION_CLOSURE", codes(report))

    def test_support_findings_are_warnings_under_the_authored_profile(self):
        overrides = {"distractors": [obj("can", [0.06, 0.06, 0.12], [0.30, 0.15, 0.0])]}
        strict = validate(document(**overrides), regions=REGIONS, task_types=TASK_TYPES)
        lenient = validate(document(**overrides), regions=REGIONS, task_types=TASK_TYPES,
                           profile="authored")
        self.assertFalse(strict.ok)
        self.assertTrue(lenient.ok)
        self.assertIn("SUPPORT_PENETRATION", codes(lenient))

    def test_target_state_must_name_the_rubric_that_will_actually_run(self):
        good = check(target_state={"rubric": "put", "predicate": "inside",
                                   "subject": "apple", "object": "bowl"})
        self.assertTrue(good.ok, [item.line() for item in good.findings])
        bad = check(target_state={"rubric": "stack", "predicate": "on_top_of",
                                  "subject": "apple", "object": "bowl"})
        self.assertIn("TARGET_STATE_RUBRIC", codes(bad))
        self.assertEqual(finding(bad, "TARGET_STATE_RUBRIC").fix["value"], "put")

    def test_repository_scene_and_progression_tables_load(self):
        self.assertIn(("Pomaria_1_int", "Kitchen_Counter"), load_regions())
        self.assertTrue({"put", "pick", "stack"} <= load_task_types())


if __name__ == "__main__":
    unittest.main()
