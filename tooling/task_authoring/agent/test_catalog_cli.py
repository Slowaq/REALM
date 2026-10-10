"""Host-safe tests for the asset catalogue, the semantic checks, and the interactive CLI driver.

    uv run python -m pytest -q tooling/task_authoring/agent/test_catalog_cli.py
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tooling.task_authoring.agent import catalog, cli, run, tools
from tooling.task_authoring.agent.test_agent import build_fixture


class MatchCategoryTest(unittest.TestCase):
    NAMES = ("bowl", "teaspoon", "saucepot", "pot_plant", "can_of_soda", "box_of_cane_sugar",
             "coffee_cup", "mug", "lemon")

    def test_word_boundaries(self):
        self.assertEqual(catalog.match_category("bowl", self.NAMES), "bowl")
        self.assertEqual(catalog.match_category("spoon", self.NAMES), "teaspoon")
        self.assertEqual(catalog.match_category("pot", self.NAMES), "saucepot")
        self.assertEqual(catalog.match_category("can", self.NAMES), "can_of_soda")
        self.assertEqual(catalog.match_category("cup", self.NAMES), "coffee_cup")
        self.assertEqual(catalog.match_category("cups", self.NAMES), "coffee_cup")

    def test_bare_pot_is_a_cooking_pot(self):
        names = ("copper_pot", "crock_pot", "drip_pot", "instant_pot", "saucepot", "teapot")
        self.assertEqual(catalog.match_category("pot", names), "saucepot")

    def test_no_loose_substrings(self):
        self.assertIsNone(catalog.match_category("on", self.NAMES))        # not lem-on
        self.assertIsNone(catalog.match_category("cane", ("can_of_soda",)))
        self.assertIsNone(catalog.match_category("plant", ("saucepot",)))


class CatalogResolveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_explicit_empty_dataset_is_an_error_not_silent_declines(self):
        with self.assertRaises(catalog.CatalogError):
            catalog.resolve(dataset=self.root / "missing")

    def test_dataset_build_round_trips_through_a_snapshot(self):
        dataset = build_fixture(self.root / "ds", categories=("bowl", "mug"))
        out = catalog.save(catalog.build_from_dataset(dataset), self.root / "cat.json")
        assets, info = catalog.resolve(catalog=out)
        live, live_info = catalog.resolve(dataset=dataset)
        self.assertEqual(sorted(assets), ["bowl", "mug"])
        self.assertEqual(info["fingerprint"], live_info["fingerprint"])

    def test_seed_from_configs_lists_only_real_shipped_assets(self):
        seed = catalog.build_from_configs()
        self.assertEqual(seed["source"], "seed-from-configs")
        pairs = {(item["category"], item["model"]) for item in seed["assets"]}
        self.assertIn(("mug", "waqrdy"), pairs)
        self.assertFalse({category for category, _ in pairs} & catalog.NON_TABLETOP)

    def test_committed_catalogue_loads(self):
        assets, info = catalog.resolve()
        self.assertTrue(assets)
        self.assertEqual(info["path"], str(catalog.DEFAULT_CATALOG))


class SemanticCheckTest(unittest.TestCase):
    ASSETS = {"bowl": [{}], "marker": [{}], "saucepot": [{}], "lid": [{}], "orange": [{}]}

    def doc(self, instruction, *categories):
        return {"instruction": instruction,
                "main_objects": [{"name": categories[0], "category": categories[0]}],
                "target_objects": [{"name": c, "category": c} for c in categories[1:]]}

    def test_object_named_but_absent_is_a_phantom(self):
        self.assertEqual(
            catalog.phantom_nouns(self.doc("Put the lid on the pot", "lid"), self.ASSETS), ["pot"])

    def test_support_words_and_colours_are_not_phantoms(self):
        document = self.doc("Put the orange marker in the bowl on the table", "marker", "bowl")
        self.assertEqual(catalog.phantom_nouns(document, self.ASSETS), [])

    def test_verbs_and_function_words_are_not_phantoms(self):
        # The full dataset reaches real categories from these words through the compound rules.
        assets = dict(self.ASSETS, nightstand=[{}], post_it=[{}], toothpick=[{}], shiitake=[{}],
                      classroom_mock_up=[{}], desk_top=[{}])
        document = self.doc("Pick up the marker and take it out of the bowl", "marker", "bowl")
        self.assertEqual(catalog.phantom_nouns(document, assets), [])
        self.assertEqual(catalog.phantom_nouns(self.doc("Put the toothpick in the bowl", "bowl"), assets),
                         ["toothpick"])


class DriverRegressionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = build_fixture(self.root / "ds")

    def tearDown(self):
        self._tmp.cleanup()

    def test_multi_object_instruction_is_declined(self):
        record = run.generate_one("Stack the cups together", offline=True, dataset=self.dataset,
                                  cache_dir=self.root / "cache")
        self.assertFalse(record["grounded"])
        self.assertIn("several objects", record["ungroundable"]["reason"])

    def test_removal_to_the_table_is_a_pick(self):
        self.assertEqual(run.classify("Remove the marker from the mug and put it on the table")[0], "pick")
        self.assertEqual(run.classify("Pick the pen on the table and put it in the bowl")[0], "put")

    def test_cache_is_invalidated_when_the_catalogue_changes(self):
        cache = self.root / "cache"
        small = build_fixture(self.root / "small", categories=("bowl", "apple", "lemon", "sponge"))
        first = run.generate_one("Put the marker in the bowl", offline=True, dataset=small, cache_dir=cache)
        self.assertFalse(first["grounded"])                      # no marker in the small catalogue
        second = run.generate_one("Put the marker in the bowl", offline=True, dataset=self.dataset,
                                  cache_dir=cache)
        self.assertTrue(second["grounded"], second.get("ungroundable"))

    def test_task_id_is_the_original_instruction_even_after_a_rewrite(self):
        session = tools.Session("Pick up the spoon", dataset=self.dataset, ranking_id="R",
                                corrections_dir=self.root / "corr")
        roles, _ = run.build_roles("Pick up the spoon", "pick", session)
        session.propose_layout(**roles)
        self.assertEqual(session.document["instruction"], "Pick up the teaspoon")
        self.assertEqual(session.document["provenance"]["task_id"],
                         run.task_id("Pick up the spoon", "R"))


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = build_fixture(self.root / "ds")
        self.base = ["--dataset", str(self.dataset), "--sessions", str(self.root / "sessions")]

    def tearDown(self):
        self._tmp.cleanup()

    def cli(self, *argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main([*self.base, *argv])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        try:
            return json.loads(text)
        except ValueError:
            return text

    def test_full_interactive_round_trip_writes_a_validated_config(self):
        started = self.cli("start", "Put the apple in the bowl", "--ranking-id", "R")
        key = started["task_id"]
        apple = self.cli("call", key, "search_assets", '{"query": "apple", "role": "main"}')
        bowl = self.cli("call", key, "search_assets", '{"query": "bowl", "role": "target"}')
        roles = {
            "task_type": "put", "instruction": "Put the apple in the bowl",
            "main": {"name": "apple", "category": "apple", "model": apple["candidates"][0]["model"]},
            "target": {"name": "bowl", "category": "bowl", "model": bowl["candidates"][-1]["model"]},
        }
        self.assertTrue(self.cli("call", key, "propose_layout", json.dumps(roles))["ok"])
        # A fresh process replays the proposal deterministically; validation sees the same draft.
        self.assertTrue(self.cli("call", key, "validate_draft")["ok"])
        self.assertTrue(self.cli("call", key, "submit_task", '{"decisions": []}')["ok"])
        self.cli("export", "--output", str(self.root / "out"), "--report", str(self.root / "r.json"))
        written = list((self.root / "out").glob("*/default.yaml"))
        self.assertEqual(len(written), 1)

    def test_replay_keeps_choices_and_reopens_what_no_longer_validates(self):
        key = self.cli("start", "Put the apple in the bowl", "--ranking-id", "R")["task_id"]
        apple = self.cli("call", key, "search_assets", '{"query": "apple", "role": "main"}')
        bowl = self.cli("call", key, "search_assets", '{"query": "bowl", "role": "target"}')
        roles = {
            "task_type": "put", "instruction": "Put the apple in the bowl",
            "main": {"name": "apple", "category": "apple", "model": apple["candidates"][0]["model"]},
            "target": {"name": "bowl", "category": "bowl", "model": bowl["candidates"][-1]["model"]},
        }
        self.cli("call", key, "propose_layout", json.dumps(roles))
        self.cli("call", key, "submit_task", '{"decisions": ["kept"]}')
        out = self.cli("replay")
        self.assertEqual(out["reopened"], 0)
        state = json.loads((self.root / "sessions" / f"{key}.json").read_text())
        self.assertEqual(state["outcome"]["status"], "submitted")
        self.assertEqual(state["outcome"]["record"]["document"]["provenance"]["decisions"], ["kept"])

    def test_tool_surface_matches_the_api_path(self):
        self.assertEqual(set(cli.TOOL_NAMES), {schema["name"] for schema in tools.tool_schemas()})

    def test_bad_json_is_reported_not_raised(self):
        key = self.cli("start", "Put the apple in the bowl")["task_id"]
        self.assertFalse(self.cli("call", key, "propose_layout", "{not json")["ok"])


class FamilyVarietyTest(unittest.TestCase):
    """Regression: one shared seed gave every task the same cameras and the same sampled clutter."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = build_fixture(self.root / "ds")

    def tearDown(self):
        self._tmp.cleanup()

    def solve(self, instruction):
        session = tools.Session(instruction, dataset=self.dataset, ranking_id="R",
                                corrections_dir=self.root / "corr")
        roles, _ = run.build_roles(instruction, "put", session)
        roles["distractors"] = []
        self.assertTrue(session.propose_layout(**roles)["ok"])
        return session.document

    def test_tasks_get_distinct_but_reproducible_seeds(self):
        self.assertEqual(tools.task_seed(100, "a", "R"), tools.task_seed(100, "a", "R"))
        self.assertNotEqual(tools.task_seed(100, "a", "R"), tools.task_seed(100, "b", "R"))
        first = self.solve("Put the apple in the bowl")
        again = self.solve("Put the apple in the bowl")
        self.assertEqual(first["camera_extrinsics"], again["camera_extrinsics"])
        cameras = {json.dumps(self.solve(f"Put the {name} in the bowl")["camera_extrinsics"], sort_keys=True)
                   for name in ("apple", "lemon", "orange", "marker", "sponge", "teaspoon")}
        self.assertGreater(len(cameras), 1)

    def test_solver_spreads_tasks_across_regions_when_none_is_named(self):
        scenes = set()
        for name in ("apple", "lemon", "orange", "marker", "sponge", "teaspoon", "mug", "plate"):
            document = self.solve(f"Put the {name} in the bowl")
            scenes.add(json.dumps(document["supported_scenes"], sort_keys=True))
        self.assertGreater(len(scenes), 1)

    def test_distractors_never_confusable_with_a_role_object(self):
        from tooling.task_authoring.agent.test_agent import CATEGORIES

        dataset = build_fixture(self.root / "ds2", categories=CATEGORIES + ("pen", "wineglass"))
        session = tools.Session("Put the marker in the mug", dataset=dataset,
                                corrections_dir=self.root / "corr")
        roles = {
            "task_type": "put", "instruction": "Put the marker in the mug",
            "main": {"name": "marker", "category": "marker", "model": "m2"},
            "target": {"name": "mug", "category": "mug", "model": "m1"},
            "distractors": [{"name": "d_pen", "category": "pen", "model": "m2"}],
        }
        self.assertTrue(session.propose_layout(**roles)["ok"])
        categories = {item["category"] for item in session.document["distractors"]}
        self.assertNotIn("pen", categories)
        self.assertFalse(categories & {"marker", "mug", "coffee_cup", "wineglass"})

    def test_block_search_points_at_a_primitive(self):
        session = tools.Session("Put the yellow block in the bowl", dataset=self.dataset,
                                corrections_dir=self.root / "corr")
        result = session.search_assets("block", "main")
        self.assertFalse(result["found"])
        self.assertIn("primitive", result["note"])


class GoalNotSatisfiedAtStartTest(unittest.TestCase):
    """Regression: `stack` used to place the main object ON its target at reset."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = build_fixture(self.root / "ds")

    def tearDown(self):
        self._tmp.cleanup()

    def blocks(self, **extra):
        return {
            "task_type": "stack", "instruction": "Put the orange block on top of the green block",
            "main": {"name": "orange_block", "primitive": "block", "rgba": [0.95, 0.5, 0.1, 1]},
            "target": {"name": "green_block", "primitive": "block", "rgba": [0.15, 0.65, 0.2, 1]},
            **extra,
        }

    def test_stack_starts_with_the_objects_apart(self):
        session = tools.Session("Put the orange block on top of the green block",
                                dataset=self.dataset, corrections_dir=self.root / "corr")
        self.assertTrue(session.propose_layout(**self.blocks())["ok"])
        document = session.document
        self.assertNotIn("initial_state", document)
        self.assertEqual(document["target_state"]["predicate"], "on_top_of")
        main = document["main_objects"][0]["relative_bbox_position"]
        target = document["target_objects"][0]["relative_bbox_position"]
        self.assertGreater(max(abs(main[0] - target[0]), abs(main[1] - target[1])), 0.05)
        self.assertTrue(session.validate_draft()["ok"])

    def test_initial_state_on_the_target_is_refused(self):
        session = tools.Session("Put the orange block on top of the green block",
                                dataset=self.dataset, corrections_dir=self.root / "corr")
        result = session.propose_layout(**self.blocks(initial_state=[
            {"predicate": "on_top_of", "subject": "orange_block", "object": "green_block"}]))
        self.assertFalse(result["ok"])
        self.assertIn("already solved", result["reason"])

    def test_validator_flags_a_main_object_resting_on_its_target(self):
        from tooling.task_authoring.validation import validate

        session = tools.Session("Put the orange block on top of the green block",
                                dataset=self.dataset, corrections_dir=self.root / "corr")
        session.propose_layout(**self.blocks())
        document = json.loads(json.dumps(session.document))
        target = document["target_objects"][0]["relative_bbox_position"]
        document["main_objects"][0]["relative_bbox_position"] = [target[0], target[1], target[2] + 0.06]
        codes = {item.code for item in validate(document).findings}
        self.assertIn("GOAL_SATISFIED_AT_START", codes)


class LengthwiseInsertionTest(unittest.TestCase):
    def test_pen_fits_a_mug_standing_up_without_shrinking(self):
        from tooling.task_authoring.validation import fits_lengthwise

        self.assertTrue(fits_lengthwise([0.16, 0.012, 0.012], [0.08, 0.12, 0.14], 1.15))
        self.assertFalse(fits_lengthwise([0.16, 0.012, 0.012], [0.27, 0.27, 0.03], 1.15))  # plate: too shallow
        self.assertFalse(fits_lengthwise([0.08, 0.08, 0.10], [0.06, 0.06, 0.15], 1.15))   # not elongated

    def test_solver_keeps_the_pen_full_size(self):
        assets = {
            "pen": [{"category": "pen", "model": "p1", "bbox": [0.16, 0.012, 0.012]}],
            "mug": [{"category": "mug", "model": "m1", "bbox": [0.08, 0.12, 0.14]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            session = tools.Session("Put the pen in the mug", assets_by_category=assets,
                                    corrections_dir=Path(tmp))
            result = session.propose_layout(
                task_type="put", instruction="Put the pen in the mug",
                main={"name": "pen", "category": "pen", "model": "p1"},
                target={"name": "mug", "category": "mug", "model": "m1"})
            self.assertTrue(result["ok"], result)
            # Only the generic 14 cm main-object ceiling applies; no receiver-capacity shrink.
            self.assertEqual(session.document["main_objects"][0]["bounding_box"], [0.14, 0.0105, 0.0105])
            self.assertFalse([a for a in result["audit"]["resized_assets"] if a["reason"] == "receiver_capacity"])
            self.assertTrue(result["audit"]["receiver_capacity"]["lengthwise_insertion"])
            self.assertTrue(session.validate_draft()["ok"])

    def test_pen_starting_inside_a_mug_keeps_full_size(self):
        # A pick whose source holds the pen `inside` stands it upright, so the same cross-section
        # rule applies: no receiver-capacity shrink to a toy pen. The mug is shallower than the pen
        # so it sticks out above the rim (RELATION_UNGRASPABLE otherwise).
        assets = {
            "pen": [{"category": "pen", "model": "p1", "bbox": [0.16, 0.012, 0.012]}],
            "mug": [{"category": "mug", "model": "m1", "bbox": [0.08, 0.12, 0.10]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            session = tools.Session("Remove the pen from the mug", assets_by_category=assets,
                                    corrections_dir=Path(tmp))
            result = session.propose_layout(
                task_type="pick", instruction="Remove the pen from the mug",
                main={"name": "pen", "category": "pen", "model": "p1"},
                source={"name": "mug", "category": "mug", "model": "m1"},
                initial_state=[{"predicate": "inside", "subject": "main", "object": "mug"}])
            self.assertTrue(result["ok"], result)
            self.assertEqual(session.document["main_objects"][0]["bounding_box"], [0.14, 0.0105, 0.0105])
            self.assertFalse([a for a in result["audit"]["resized_assets"] if a["reason"] == "receiver_capacity"])
            self.assertTrue(result["audit"]["receiver_capacity"]["lengthwise_insertion"])
            self.assertTrue(session.validate_draft()["ok"])


class SourceRelationTest(unittest.TestCase):
    """Regressions from the full-catalogue pilot: lids shrunk into pots, markers tipping in pots."""

    def session(self, instruction, assets):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return tools.Session(instruction, assets_by_category=assets, corrections_dir=Path(self._tmp.name))

    def test_lid_on_a_pot_keeps_a_rim_resting_size(self):
        assets = {
            "lid": [{"category": "lid", "model": "l1", "bbox": [0.136, 0.136, 0.03]}],
            "saucepot": [{"category": "saucepot", "model": "p1", "bbox": [0.118, 0.118, 0.12]}],
        }
        session = self.session("Remove the lid from the pot", assets)
        result = session.propose_layout(
            task_type="pick", instruction="Remove the lid from the saucepot",
            main={"name": "lid", "category": "lid", "model": "l1"},
            source={"name": "saucepot", "category": "saucepot", "model": "p1"},
            initial_state=[{"predicate": "on_top_of", "subject": "lid", "object": "saucepot"}])
        self.assertTrue(result["ok"], result)
        self.assertEqual(session.document["main_objects"][0]["bounding_box"], [0.136, 0.136, 0.03])
        self.assertTrue(session.validate_draft()["ok"], session.validate_draft())

    def test_put_the_lid_on_the_pot_keeps_a_full_size_lid(self):
        assets = {
            "lid": [{"category": "lid", "model": "l1", "bbox": [0.232, 0.232, 0.03]}],
            "saucepot": [{"category": "saucepot", "model": "p1", "bbox": [0.208, 0.208, 0.12]}],
        }
        session = self.session("Put the lid on the pot", assets)
        result = session.propose_layout(
            task_type="stack", instruction="Put the lid on the saucepot",
            main={"name": "lid", "category": "lid", "model": "l1"},
            target={"name": "saucepot", "category": "saucepot", "model": "p1"})
        self.assertTrue(result["ok"], result)
        self.assertEqual(session.document["main_objects"][0]["bounding_box"][:2], [0.232, 0.232])
        self.assertTrue(session.validate_draft()["ok"], session.validate_draft())

    def test_marker_in_a_wide_pot_lies_flat(self):
        assets = {
            "marker": [{"category": "marker", "model": "k1", "bbox": [0.14, 0.02, 0.02]}],
            "saucepot": [{"category": "saucepot", "model": "p1", "bbox": [0.25, 0.25, 0.12]}],
        }
        session = self.session("Remove the marker from the pot", assets)
        result = session.propose_layout(
            task_type="pick", instruction="Remove the marker from the saucepot",
            main={"name": "marker", "category": "marker", "model": "k1"},
            source={"name": "saucepot", "category": "saucepot", "model": "p1"},
            initial_state=[{"predicate": "inside", "subject": "marker", "object": "saucepot"}])
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["audit"]["initial_relation"]["lying_flat"])
        self.assertEqual(session.document["main_objects"][0]["orientation"], [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(session.validate_draft()["ok"], session.validate_draft())

    def test_flat_object_is_dropped_in_from_the_rim_not_authored_at_the_bbox_floor(self):
        """The bbox bottom is not the floor: sunglasses authored there were ejected from a storage
        box's thick base, and a match box clipped through a bowl's curved wall."""
        assets = {
            "sunglasses": [{"category": "sunglasses", "model": "s1", "bbox": [0.107, 0.042, 0.034]}],
            "storage_box": [{"category": "storage_box", "model": "b1", "bbox": [0.235, 0.25, 0.057]}],
        }
        session = self.session("Take the sunglasses out of the box", assets)
        result = session.propose_layout(
            task_type="pick", instruction="Take the sunglasses out of the box",
            main={"name": "sunglasses", "category": "sunglasses", "model": "s1"},
            source={"name": "box", "category": "storage_box", "model": "b1"},
            initial_state=[{"predicate": "inside", "subject": "sunglasses", "object": "box"}])
        self.assertTrue(result["ok"], result)
        main = session.document["main_objects"][0]
        box = session.document["immutables"][0]
        rim = box["relative_bbox_position"][2] + box["bounding_box"][2] / 2
        low = main["relative_bbox_position"][2] - main["bounding_box"][2] / 2
        self.assertGreater(low, rim)
        self.assertLess(low, rim + 0.02)
        self.assertTrue(session.validate_draft()["ok"], session.validate_draft())

    def test_upright_object_in_a_wide_pot_stays_upright(self):
        """A salt shaker is long along Z: yawing it does not lay it down, and putting it 'flat' above
        the rim left it wholly outside the pot."""
        assets = {
            "salt_shaker": [{"category": "salt_shaker", "model": "s1", "bbox": [0.051, 0.051, 0.14]}],
            "saucepot": [{"category": "saucepot", "model": "p1", "bbox": [0.205, 0.28, 0.138]}],
        }
        session = self.session("Take the salt shaker out of the pot", assets)
        result = session.propose_layout(
            task_type="pick", instruction="Take the salt shaker out of the saucepot",
            main={"name": "salt_shaker", "category": "salt_shaker", "model": "s1"},
            source={"name": "pot", "category": "saucepot", "model": "p1"},
            initial_state=[{"predicate": "inside", "subject": "salt_shaker", "object": "pot"}])
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["audit"]["initial_relation"]["lying_flat"])
        self.assertTrue(session.validate_draft()["ok"], session.validate_draft())

    def test_plate_on_a_bowl_must_overhang_the_rim(self):
        from tooling.task_authoring.validation import validate

        assets = {
            "plate": [{"category": "plate", "model": "p1", "bbox": [0.18, 0.18, 0.0045]}],
            "bowl": [{"category": "bowl", "model": "b1", "bbox": [0.125, 0.125, 0.074]}],
        }
        session = self.session("Remove the plate from the bowl", assets)
        result = session.propose_layout(
            task_type="pick", instruction="Remove the plate from the bowl",
            main={"name": "plate", "category": "plate", "model": "p1"},
            source={"name": "bowl", "category": "bowl", "model": "b1"},
            initial_state=[{"predicate": "on_top_of", "subject": "plate", "object": "bowl"}])
        self.assertTrue(result["ok"], result)
        # Not capped at the 14 cm hand-object ceiling: it rests across the opening.
        self.assertGreater(session.document["main_objects"][0]["bounding_box"][0], 0.15)
        self.assertTrue(session.validate_draft()["ok"], session.validate_draft())
        # The rendered failure: a 15.2 cm plate on a 12.5 cm bowl, 7 mm a side.
        session.document["main_objects"][0]["bounding_box"][:2] = [0.152, 0.152]
        codes = [f.code for f in validate(session.document).findings]
        self.assertIn("RIM_OVERHANG", codes)

    def test_search_lists_alternative_categories(self):
        assets = {name: [{"category": name, "model": "m", "bbox": [0.08, 0.08, 0.1]}]
                  for name in ("soda_cup", "coffee_cup", "paper_cup", "teacup", "mug")}
        result = self.session("Put the marker in the cup", assets).search_assets("cup", "target")
        self.assertEqual(result["category"], "soda_cup")
        self.assertEqual(set(result["alternatives"]), {"coffee_cup", "paper_cup", "teacup"})


class RollingClutterTest(unittest.TestCase):
    def test_round_objects_are_not_clutter(self):
        from tooling.task_authoring.agent import layout

        for category in ("volleyball", "pear", "apple", "orange", "baseball", "tennis_ball",
                         "chestnut", "gooseberry", "strawberry", "papaya"):
            self.assertTrue(layout.rolls(category), category)
        for category in ("banana", "sponge", "half_apple", "toy_dice", "bottle_of_water",
                         "bottle_of_olive_oil", "jar_of_strawberry_jam", "bottle_of_strawberry_juice"):
            self.assertFalse(layout.rolls(category), category)


class ClusterSelectionTest(unittest.TestCase):
    def test_rewordings_merge_and_failed_demos_do_not_count(self):
        import sys
        import types

        sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
        sys.modules.setdefault("pyarrow.parquet", types.ModuleType("pyarrow.parquet"))
        from tooling.task_authoring import select_droid100

        rows = [
            {"instruction": "Put the marker in the mug", "successful_episodes": 3, "locations": ["A", "B"]},
            {"instruction": "Place the marker in the mug.", "successful_episodes": 1, "locations": ["C"]},
            {"instruction": "Put the pen in the bowl", "successful_episodes": 5, "locations": ["A"]},
            {"instruction": "Put the spoon in the bowl", "successful_episodes": 0, "locations": []},
            {"instruction": "Open the drawer", "successful_episodes": 9, "locations": ["A", "B", "C", "D"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counts.json"
            path.write_text(json.dumps({"chunks": 1, "instructions": rows}))
            result = select_droid100.select_clusters(path, limit=10)
        signatures = [tuple(task["signature"]) for task in result["tasks"]]
        self.assertEqual(signatures[0], ("put", "marker", "mug"))      # 3 locations beats 5 episodes
        self.assertEqual(result["tasks"][0]["episodes"], 4)
        self.assertEqual(result["tasks"][0]["instruction"], "Put the marker in the mug")
        self.assertNotIn(("put", "teaspoon", "bowl"), signatures)        # no successful episode
        self.assertFalse(any(sig[0] == "open_drawer" for sig in signatures))  # fixture verb


class SignatureTest(unittest.TestCase):
    def test_colour_alone_is_not_a_new_task(self):
        def doc(rgba):
            return {"task_type": "put",
                    "main_objects": [{"type": "PrimitiveObject", "primitive_type": "Cube", "rgba": rgba}],
                    "target_objects": [{"category": "bowl"}]}

        self.assertEqual(run.signature(doc([0.1, 0.2, 0.9, 1])), run.signature(doc([0.9, 0.8, 0.1, 1])))


class ClutterSourceTest(unittest.TestCase):
    def test_automatic_clutter_comes_from_droid_categories(self):
        from tooling.task_authoring.agent import layout

        droid = ["apple", "banana", "lemon", "sponge", "marker", "toy_dice", "teaspoon"]
        assets = {name: [{"category": name, "model": "m", "bbox": [0.05, 0.05, 0.05]}]
                  for name in droid + ["diced_watermelon", "cooked_squid"]}
        eligible = layout._eligible_distractors(assets)
        self.assertNotIn("diced_watermelon", eligible)
        self.assertNotIn("cooked_squid", eligible)
        self.assertIn("banana", eligible)
        self.assertNotIn("apple", eligible)  # round: rolls at settle


class StaleSessionTest(CliTest):
    def test_session_from_another_catalogue_is_offered_again(self):
        listing = self.root / "list.txt"
        listing.write_text("Put the apple in the bowl\n")
        key = self.cli("start", "Put the apple in the bowl")["task_id"]
        self.cli("call", key, "report_ungroundable", '{"reason": "test"}')
        self.assertTrue(self.cli("next", str(listing))["done"])
        other = build_fixture(self.root / "other", categories=("bowl", "apple", "lemon"))
        self.base = ["--dataset", str(other), "--sessions", str(self.root / "sessions")]
        self.assertEqual(self.cli("next", str(listing))["task_id"], key)


if __name__ == "__main__":
    unittest.main()
