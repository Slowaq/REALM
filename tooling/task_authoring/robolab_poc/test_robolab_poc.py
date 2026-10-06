"""Host tests for the RoboLab-style scene PoC (no Isaac, no dataset).

    uv run python -m unittest tooling.task_authoring.robolab_poc.test_robolab_poc -v
"""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import yaml

from tooling.task_authoring.robolab_poc import catalog as catalog_mod
from tooling.task_authoring.robolab_poc import cli
from tooling.task_authoring.robolab_poc.frames import find_region, usable_regions
from tooling.task_authoring.robolab_poc.scene import SpecError, solve, validate_spec

HERE = Path(__file__).parent
SPECS = sorted((HERE / "specs").glob("*.json"))
CATALOG = catalog_mod.load_catalog()


def spec(name: str) -> dict:
    return json.loads((HERE / "specs" / f"{name}.json").read_text())


def env_config_position(relative, spawn_bbox, robot_rot_deg_z):
    """Verbatim arithmetic of realm/environments/env_config._apply_object_cfg."""
    relative = list(relative)
    modifier = -1 if 90 <= robot_rot_deg_z <= 270 else 1
    relative[0] *= modifier
    if modifier != 1:
        if relative[0] < 0:
            relative[0] -= modifier * (spawn_bbox[1] - spawn_bbox[0])
        else:
            relative[0] += modifier * (spawn_bbox[1] - spawn_bbox[0])
    return [x + y for x, y in zip(relative, [spawn_bbox[0], spawn_bbox[2], spawn_bbox[4]])]


class FrameTests(unittest.TestCase):

    def test_world_matches_env_config(self):
        rng = random.Random(0)
        for frame in usable_regions():
            r = frame.region
            spawn = [r["x_min"], r["x_max"], r["y_min"], r["y_max"], r["z"]]
            for _ in range(20):
                rel = [rng.uniform(0.01, r["width"]), rng.uniform(0.01, r["depth"]), 0.1]
                expected = env_config_position(rel, spawn, r["robot_rot"][2])
                got = frame.world(rel[0], rel[1])
                self.assertAlmostEqual(got[0], expected[0], places=9, msg=frame.id)
                self.assertAlmostEqual(got[1], expected[1], places=9, msg=frame.id)

    def test_round_trip(self):
        for frame in usable_regions():
            a, b = frame.to_solver(0.11, 0.23)
            x, y = frame.from_solver(a, b)
            self.assertAlmostEqual(x, 0.11, places=9)
            self.assertAlmostEqual(y, 0.23, places=9)

    def test_every_region_is_in_front_of_its_robot(self):
        # Independent evidence the robot-frame convention is right: every support sits
        # 0.3-0.9 m ahead of the base and roughly centred laterally.
        for frame in usable_regions():
            a, b = frame.to_solver(frame.region["width"] / 2, frame.region["depth"] / 2)
            self.assertTrue(0.3 < a < 0.9, f"{frame.id}: forward {a:.3f}")
            self.assertLess(abs(b), 0.2, f"{frame.id}: lateral {b:.3f}")

    def test_non_axis_aligned_regions_are_excluded(self):
        self.assertTrue(all(frame.yaw % 90 == 0 for frame in usable_regions()))
        with self.assertRaises(ValueError):
            find_region("Wainscott_0_int / Dining_Table")


class CatalogTests(unittest.TestCase):

    def test_committed_catalog_is_current(self):
        self.assertEqual(catalog_mod.build_catalog()["fingerprint"], CATALOG["fingerprint"],
                         "catalog.json is stale: run `cli catalog`")

    def test_search_is_whole_word(self):
        categories = [hit["category"] for hit in catalog_mod.search(CATALOG, "can")]
        self.assertIn("can_of_soda", categories)
        self.assertNotIn("box_of_cane_sugar", categories)
        self.assertEqual(catalog_mod.search(CATALOG, "sink"), [])

    def test_placeholder_extents_are_skipped(self):
        self.assertTrue(all(max(asset["bbox"]) >= 0.02 for asset in CATALOG["assets"]))


class SpecValidationTests(unittest.TestCase):

    def assertRejected(self, mutate):
        candidate = spec("put_soda_can_in_bowl_mug_left")
        mutate(candidate)
        with self.assertRaises(SpecError):
            validate_spec(candidate)

    def test_rejects_numbers(self):
        self.assertRejected(lambda s: s["objects"][0].update(x=0.2))
        self.assertRejected(lambda s: s["relations"][0].update(distance=0.1))

    def test_rejects_role_contract_violations(self):
        self.assertRejected(lambda s: s["objects"].pop(1))                       # put without target
        self.assertRejected(lambda s: s.update(task_type="pick"))                # pick with a target
        self.assertRejected(lambda s: s["relations"].append(
            {"type": "in", "object": "can_of_soda", "container": "bowl"}))      # main in a target
        self.assertRejected(lambda s: s["relations"].append(
            {"type": "in", "object": "mug", "container": "bowl"}))              # in + spatial
        self.assertRejected(lambda s: s["relations"].append({"type": "near", "object": "mug", "reference": "bowl"}))

    def test_ungroundable_category(self):
        candidate = spec("put_soda_can_in_bowl_mug_left")
        candidate["objects"][1]["category"] = "sink"
        with self.assertRaisesRegex(SpecError, "ungroundable"):
            solve(candidate, CATALOG)


class SolveTests(unittest.TestCase):

    def test_examples_solve_without_errors(self):
        for path in SPECS:
            result = solve(json.loads(path.read_text()), CATALOG)
            self.assertTrue(result.ok, f"{path.name}: {result.error} {result.findings}")

    def test_deterministic(self):
        first = solve(spec("pick_lemon_from_bowl"), CATALOG)
        second = solve(spec("pick_lemon_from_bowl"), CATALOG)
        self.assertEqual(first.document, second.document)
        self.assertEqual(first.provenance["seed"], second.provenance["seed"])

    def test_global_random_state_is_untouched(self):
        random.seed(123)
        expected = random.random()
        random.seed(123)
        solve(spec("put_orange_in_bowl_with_lemon"), CATALOG)
        self.assertEqual(random.random(), expected)

    def test_heights_follow_authoring_rules(self):
        result = solve(spec("pick_lemon_from_bowl"), CATALOG)
        doc = result.document
        bowl, lemon = doc["immutables"][0], doc["main_objects"][0]
        self.assertAlmostEqual(bowl["relative_bbox_position"][2], bowl["bounding_box"][2] / 2 + 0.05, places=6)
        bowl_top = bowl["relative_bbox_position"][2] + bowl["bounding_box"][2] / 2
        lemon_bottom = lemon["relative_bbox_position"][2] - lemon["bounding_box"][2] / 2
        self.assertGreater(lemon_bottom, bowl_top)               # dropped in, never interpenetrating
        self.assertEqual(lemon["relative_bbox_position"][:2], bowl["relative_bbox_position"][:2])

    def test_relations_hold_in_robot_frame(self):
        result = solve(spec("put_soda_can_in_bowl_mug_left"), CATALOG)
        layout = result.layout
        self.assertGreater(layout["mug"]["robot_frame_forward_left"][1], layout["bowl"]["robot_frame_forward_left"][1])
        self.assertLess(layout["lemon"]["robot_frame_forward_left"][0], layout["bowl"]["robot_frame_forward_left"][0])

    def test_instruction_tokens(self):
        doc = solve(spec("put_soda_can_in_bowl_mug_left"), CATALOG).document
        self.assertIn(doc["instruction_obj_to_replace"], doc["instruction"])
        self.assertIn(doc["instruction_target_to_replace"], doc["instruction"])
        bad = spec("put_soda_can_in_bowl_mug_left")
        bad["instruction"] = "put the drink in the bowl"
        result = solve(bad, CATALOG, attempts=1)
        self.assertFalse(result.ok)
        self.assertIn("INSTRUCTION_OBJ_TOKEN", {f["code"] for f in result.findings})


class EmitTests(unittest.TestCase):

    def test_committed_configs_regenerate_identically(self):
        family = cli.DEFAULT_OUT
        committed = sorted(family.glob("*/default.yaml"))
        self.assertEqual(len(committed), len(SPECS), "every example spec should be emitted under ROBOLAB_POC/")
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(cli.main(["emit", *map(str, SPECS), "--out", out]), 0)
            for path in committed:
                fresh = Path(out) / path.parent.name / "default.yaml"
                self.assertEqual(yaml.safe_load(fresh.read_text()), yaml.safe_load(path.read_text()), path.parent.name)

    def test_emit_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as out:
            args = ["emit", str(SPECS[0]), "--out", out, "--cameras", "ep_024104_cam1,ep_024104_cam2"]
            self.assertEqual(cli.main(args), 0)
            self.assertEqual(cli.main(args), 1)
            self.assertEqual(cli.main([*args, "--force"]), 0)

    def test_emitted_documents_satisfy_repo_config_tests(self):
        # Mirrors tests/test_perturbation_task_types.py without importing pytest fixtures.
        for path in sorted(cli.DEFAULT_OUT.glob("*/default.yaml")):
            cfg = yaml.safe_load(path.read_text())
            self.assertIn(cfg["task_type"], {"put", "pick", "rotate", "stack"})
            self.assertIn(cfg["instruction_obj_to_replace"], cfg["instruction"])
            self.assertEqual(copy.deepcopy(cfg["supported_scenes"]), cfg["supported_scenes"])
            for role in ("main_objects", "target_objects", "distractors", "immutables"):
                for obj in cfg[role]:
                    self.assertEqual(set(obj), {"type", "name", "category", "model", "bounding_box",
                                                "orientation", "relative_bbox_position"}, path)


if __name__ == "__main__":
    unittest.main()
