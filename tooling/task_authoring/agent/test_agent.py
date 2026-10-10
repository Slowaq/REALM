"""Host-safe tests for the agentic generator: layout solver, tools, corrections, driver.

These run with no GPU, no container, and no API key. They build a synthetic asset tree in a temp
directory rather than reading the real OmniGibson catalogue, so the pipeline is testable on a
development machine that has no dataset staged -- which is the property that keeps the agent code
reviewable before it ever reaches the authoring host.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from tooling.task_authoring.agent import corrections, layout, review, run, tools
from tooling.task_authoring.validation import validate


CATEGORIES = (
    "apple", "bowl", "mug", "marker", "plate", "storage_box", "teaspoon", "tablefork",
    "orange", "lemon", "sponge", "chocolate_bar", "toy_dice", "saucepot", "lid",
)

PUT_ROLES = {
    "task_type": "put",
    "instruction": "Put the orange block in the bowl",
    "main": {"name": "orange_block", "primitive": "block", "rgba": [0.95, 0.4, 0.05, 1.0], "extent": 0.05},
    "target": {"name": "bowl", "category": "bowl", "model": "m1"},
    "distractors": [
        {"name": "distractor_apple", "category": "apple", "model": "m2"},
        {"name": "distractor_marker", "category": "marker", "model": "m2"},
    ],
}


def build_fixture(root: Path, categories=CATEGORIES) -> Path:
    """A minimal `objects/<category>/<model>/USD/*.usd` tree with metadata the indexer reads."""
    for category in categories:
        for model, bbox in (("m1", [0.08, 0.08, 0.10]), ("m2", [0.05, 0.05, 0.05])):
            usd = root / "objects" / category / model / "USD"
            usd.mkdir(parents=True, exist_ok=True)
            (usd / f"{model}.usd").write_text("#usda 1.0\n")
            misc = root / "objects" / category / model / "misc"
            misc.mkdir(exist_ok=True)
            (misc / "metadata.json").write_text(json.dumps({"bbox_size": bbox}))
    return root


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset = build_fixture(Path(self._tmp.name) / "ds")
        self.assets = layout.index_assets(self.dataset)

    def tearDown(self):
        self._tmp.cleanup()

    def test_layout_solves_and_validates_clean(self):
        result = layout.build_document(PUT_ROLES, dataset=self.dataset)
        document = result["document"]
        self.assertTrue(validate(document).ok, validate(document).as_dict())
        self.assertEqual(len(document["main_objects"]), 1)
        self.assertEqual(len(document["target_objects"]), 1)

    def test_layout_is_deterministic_under_seed(self):
        first = yaml.safe_dump(layout.build_document(PUT_ROLES, dataset=self.dataset)["document"])
        second = yaml.safe_dump(layout.build_document(PUT_ROLES, dataset=self.dataset)["document"])
        self.assertEqual(first, second)

    def test_main_object_is_a_literal_substring_of_the_instruction(self):
        document = layout.build_document(PUT_ROLES, dataset=self.dataset)["document"]
        self.assertIn(document["instruction_obj_to_replace"], document["instruction"])
        self.assertIn(document["instruction_target_to_replace"], document["instruction"])

    def test_unknown_model_is_refused_with_the_known_ones(self):
        roles = {**PUT_ROLES, "target": {"name": "bowl", "category": "bowl", "model": "nope"}}
        with self.assertRaises(layout.LayoutError) as caught:
            layout.build_document(roles, dataset=self.dataset)
        self.assertIn("does not exist", str(caught.exception))
        self.assertIn("m1", str(caught.exception))

    def test_unknown_category_is_refused_not_invented(self):
        roles = {**PUT_ROLES, "target": {"name": "x", "category": "hyperflux", "model": "m1"}}
        with self.assertRaises(layout.LayoutError) as caught:
            layout.build_document(roles, dataset=self.dataset)
        self.assertIn("not in the indexed catalogue", str(caught.exception))

    def test_put_without_a_receiver_is_refused(self):
        roles = {key: value for key, value in PUT_ROLES.items() if key != "target"}
        with self.assertRaises(layout.LayoutError) as caught:
            layout.build_document(roles, dataset=self.dataset)
        self.assertIn("target", str(caught.exception))

    def test_unknown_task_type_is_refused(self):
        with self.assertRaises(layout.LayoutError):
            layout.build_document({**PUT_ROLES, "task_type": "levitate"}, dataset=self.dataset)

    def test_region_index_out_of_range_is_reported(self):
        with self.assertRaises(layout.LayoutError) as caught:
            layout.build_document(PUT_ROLES, region_index=9999, dataset=self.dataset)
        self.assertIn("out of range", str(caught.exception))

    def test_source_role_is_authored_as_an_immutable(self):
        roles = {
            "task_type": "pick", "instruction": "Take the lid off the saucepot",
            # A lid as wide as the pot rests on its rim; a narrower one would drop inside.
            "main": {"name": "lid", "category": "lid", "model": "m1"},
            "source": {"name": "saucepot", "category": "saucepot", "model": "m1"},
            "initial_state": [{"predicate": "on_top_of", "subject": "lid", "object": "saucepot"}],
        }
        document = layout.build_document(roles, dataset=self.dataset)["document"]
        self.assertEqual([item["name"] for item in document["immutables"]], ["saucepot"])
        self.assertTrue(validate(document).ok, validate(document).as_dict())

    def test_receiver_capacity_shrinks_the_main_object_and_audits_it(self):
        # A main object larger than the receiver must be uniformly shrunk, and the audit recorded.
        roles = {
            "task_type": "put", "instruction": "Put the apple in the mug",
            "main": {"name": "apple", "category": "apple", "model": "m1"},
            "target": {"name": "mug", "category": "mug", "model": "m1"},
            "distractors": [],
        }
        result = layout.build_document(roles, dataset=self.dataset)
        capacity = result["audit"]["receiver_capacity"]
        self.assertIsNotNone(capacity)
        if capacity["uniform_scale"] < 1:
            self.assertTrue(any(item["reason"] == "receiver_capacity" for item in result["audit"]["resized_assets"]))


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset = build_fixture(Path(self._tmp.name) / "ds")
        self.corrections = Path(self._tmp.name) / "corrections"

    def tearDown(self):
        self._tmp.cleanup()

    def session(self, instruction="Put the orange block in the bowl"):
        return tools.Session(instruction, dataset=self.dataset, corrections_dir=self.corrections)

    def test_tool_schemas_expose_no_coordinate_setter(self):
        names = {schema["name"] for schema in tools.tool_schemas()}
        self.assertEqual(names, {
            "search_assets", "list_scene_regions", "propose_layout",
            "validate_draft", "submit_task", "report_ungroundable",
        })
        blob = json.dumps(tools.tool_schemas())
        for forbidden in ("set_position", "set_scale", "write_yaml", "relative_bbox_position"):
            self.assertNotIn(forbidden, blob)

    def test_search_assets_reports_a_missing_category_rather_than_guessing(self):
        result = self.session().search_assets("hyperflux", "main")
        self.assertFalse(result["found"])
        self.assertEqual(result["candidates"], [])
        self.assertIn("does not exist", result["note"])

    def test_search_assets_finds_a_substring_and_names_the_real_category(self):
        result = self.session().search_assets("spoon", "main")
        self.assertTrue(result["found"])
        self.assertEqual(result["category"], "teaspoon")

    def test_search_assets_honours_the_footprint_ceiling(self):
        result = self.session().search_assets("bowl", "target", max_footprint_xy=(0.01, 0.01))
        self.assertFalse(result["found"])

    def test_validate_before_propose_is_reported_not_raised(self):
        result = self.session().validate_draft()
        self.assertFalse(result["ok"])
        self.assertIn("propose_layout", result["reason"])

    def test_submit_before_propose_is_reported_not_raised(self):
        self.assertFalse(self.session().submit_task([])["ok"])

    def test_unknown_tool_name_returns_an_error_not_an_exception(self):
        result = tools.dispatch(self.session(), "delete_everything", {})
        self.assertIn("unknown tool", result["error"])

    def test_agent_cannot_reach_the_solver_with_a_bad_model(self):
        session = self.session()
        result = session.propose_layout(
            task_type="put", instruction="Put the apple in the bowl",
            main={"name": "apple", "category": "apple", "model": "invented"},
            target={"name": "bowl", "category": "bowl", "model": "m1"},
        )
        self.assertFalse(result["ok"])
        self.assertIn("does not exist", result["reason"])

    def test_submit_refuses_a_draft_that_does_not_validate(self):
        session = self.session()
        session.propose_layout(
            task_type="pick", instruction="Pick up the spoon",
            main={"name": "teaspoon", "category": "teaspoon", "model": "m2"},
        )
        result = session.submit_task([])
        self.assertFalse(result["ok"])
        self.assertEqual(result["findings"][0]["code"], "INSTRUCTION_CLOSURE")

    def test_full_loop_submits_a_validated_document(self):
        session = self.session()
        self.assertTrue(session.propose_layout(**PUT_ROLES)["ok"])
        self.assertTrue(session.validate_draft()["ok"])
        submitted = session.submit_task(["authored the block as a primitive"])
        self.assertTrue(submitted["ok"])
        self.assertTrue(validate(session.document).ok)


class CorrectionStoreTest(unittest.TestCase):
    def test_task_id_is_content_keyed_and_stable_across_reranking(self):
        first = corrections.task_id("Put the apple in the bowl", "droid100-v1")
        second = corrections.task_id("Put the apple in the bowl", "droid100-v1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, corrections.task_id("Put the apple in the bowl", "other"))
        self.assertNotEqual(first, corrections.task_id("Put the pear in the bowl", "droid100-v1"))

    def test_recorded_correction_merges_back_into_the_document(self):
        document = {"instruction": "Pick up the teaspoon", "main_objects": [
            {"name": "teaspoon", "category": "teaspoon", "model": "m2"},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            corrections.record_correction(
                document, code="SCALE", action="swap_asset", obj="teaspoon",
                to={"model": "m1"}, reason="too small", corrections_dir=store,
            )
            applied = corrections.merge_document(document, corrections_dir=store)
        self.assertEqual(document["main_objects"][0]["model"], "m1")
        self.assertEqual(applied[0]["applied"], {"model": "m1"})

    def test_a_correction_naming_an_absent_object_is_skipped_not_fatal(self):
        document = {"instruction": "Pick up the teaspoon", "main_objects": [
            {"name": "teaspoon", "category": "teaspoon", "model": "m2"},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            corrections.record_correction(
                document, code="SCALE", action="swap_asset", obj="teaspoon",
                to={"model": "m1"}, corrections_dir=store,
            )
            document["main_objects"][0]["name"] = "renamed"
            applied = corrections.merge_document(document, corrections_dir=store)
        self.assertIn("skipped", applied[0])

    def test_stored_correction_never_carries_a_position(self):
        document = {"instruction": "x", "main_objects": [{"name": "a", "model": "m1"}]}
        with tempfile.TemporaryDirectory() as directory:
            corrections.record_correction(
                document, code="SCALE", action="swap_asset", obj="a", to={"model": "m2"},
                corrections_dir=Path(directory),
            )
            stored = json.loads(next(Path(directory).glob("*.json")).read_text())
        self.assertNotIn("position", json.dumps(stored))
        self.assertNotIn("bounding_box", json.dumps(stored))

    def test_malformed_store_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            (store / "broken.json").write_text("{not json")
            self.assertEqual(corrections.load_store(store), {})


class DriverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = build_fixture(self.root / "ds")

    def tearDown(self):
        self._tmp.cleanup()

    def run_one(self, instruction, **kwargs):
        return run.generate_one(
            instruction, offline=True, dataset=self.dataset,
            output=self.root / "out", cache_dir=self.root / "cache", **kwargs,
        )

    def test_intent_classification_covers_the_closed_task_set(self):
        self.assertEqual(run.classify("Put the apple in the bowl")[0], "put")
        self.assertEqual(run.classify("Pick up the spoon")[0], "pick")
        self.assertEqual(run.classify("Stack the cup on the plate")[0], "stack")
        self.assertEqual(run.classify("Rotate the mug")[0], "rotate")
        self.assertEqual(run.classify("Push the switch")[0], "push")
        self.assertEqual(run.classify("Open the drawer")[0], "open_drawer")
        self.assertEqual(run.classify("Close the drawer")[0], "close_drawer")

    def test_ungroundable_verb_is_declined_with_a_reason(self):
        record = self.run_one("Please recombobulate the hyperflux capacitor")
        self.assertFalse(record["grounded"])
        self.assertIn("task type", record["ungroundable"]["reason"])

    def test_catalogue_words_are_nouns_even_when_the_static_vocabulary_lacks_them(self):
        # Regression: `nouns` once scanned a fixed vocabulary that had no "apple", so
        # "Put the apple in the bowl" saw only "bowl" and authored a bowl on a bowl.
        assets = layout.index_assets(self.dataset)
        self.assertEqual(run.nouns("Put the apple in the bowl", assets), ["apple", "bowl"])
        self.assertEqual(run.nouns("Put the apple in the bowl"), ["bowl"])  # no catalogue: no apple

    def test_two_distinct_catalogue_nouns_never_share_a_name(self):
        record = self.run_one("Put the apple in the bowl")
        self.assertTrue(record["grounded"], record.get("ungroundable"))
        document = record["document"]
        names = [str(config["name"]) for role in
                 ("main_objects", "target_objects", "distractors", "immutables")
                 for config in document.get(role) or []]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(document["instruction_obj_to_replace"], "apple")
        self.assertTrue(validate(document).ok, validate(document).as_dict())

    def test_a_declined_run_always_records_a_reason(self):
        # A refusal with no reason reads as a crash: the report is all an operator sees.
        for instruction in (
            "Please recombobulate the hyperflux capacitor",
            "Open the drawer",
            "Put the apple in the bowl",
            "Pick up the spoon",
        ):
            record = self.run_one(instruction)
            if not record["grounded"]:
                self.assertIsNotNone(record["ungroundable"], instruction)
                self.assertTrue(record["ungroundable"]["reason"].strip(), instruction)

    def test_stopwords_cannot_be_grounded_as_objects(self):
        # `ground_word` deliberately keeps the loose substring match that lets "spoon" reach
        # "teaspoon", so "on" does match "lemon" in isolation. The stopword filter in `nouns` is
        # what stops instruction grammar becoming an object, and that is the layer under test.
        assets = layout.index_assets(self.dataset)
        for word in ("the", "put", "into", "blue", "on"):
            self.assertIn(word, run.STOPWORDS, word)
        self.assertEqual(run.nouns("Put the apple on the lemon", assets), ["apple", "lemon"])

    def test_substituted_noun_is_rewritten_and_recorded(self):
        record = self.run_one("Pick up the spoon")
        self.assertTrue(record["grounded"], record.get("ungroundable"))
        document = record["document"]
        self.assertEqual(document["instruction"], "Pick up the teaspoon")
        self.assertIn("teaspoon", document["instruction"])
        self.assertTrue(validate(document).ok)

    def test_grounded_record_validates_and_writes_a_config(self):
        record = self.run_one("Put the orange block in the bowl")
        self.assertTrue(record["grounded"])
        self.assertEqual(record["validation"]["error_count"], 0)
        written = list((self.root / "out").glob("*/default.yaml"))
        self.assertEqual(len(written), 1)
        self.assertTrue(validate(yaml.safe_load(written[0].read_text())).ok)

    def test_offline_run_is_byte_identical_across_invocations(self):
        first = self.run_one("Put the orange block in the bowl", use_cache=False)
        second = self.run_one("Put the orange block in the bowl", use_cache=False)
        self.assertEqual(yaml.safe_dump(first["document"]), yaml.safe_dump(second["document"]))

    def test_cache_is_keyed_by_content_and_prompt_version(self):
        first = self.run_one("Put the orange block in the bowl")
        path = run.cache_path("Put the orange block in the bowl", None, self.root / "cache")
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text())["prompt_version"], first["prompt_version"])

    def test_a_withheld_asset_is_declined_not_substituted_silently(self):
        # The fixture has no cabinet, so a drawer instruction must decline rather than invent one.
        record = self.run_one("Open the drawer")
        self.assertFalse(record["grounded"])
        self.assertIn("cabinet", record["ungroundable"]["reason"])

    def test_every_grounded_config_names_only_indexed_categories(self):
        assets = layout.index_assets(self.dataset)
        for instruction in (
            "Put the orange block in the bowl",
            "Pick up the spoon",
            "Stack the cup on the plate",
            "Rotate the mug",
        ):
            record = self.run_one(instruction)
            self.assertTrue(record["grounded"], instruction)
            for role in ("main_objects", "target_objects", "distractors", "immutables"):
                for config in record["document"].get(role) or []:
                    if config.get("type") != "DatasetObject":
                        continue
                    self.assertIn(config["category"], assets, f"{instruction}: {config['category']}")
                    models = {item["model"] for item in assets[config["category"]]}
                    self.assertIn(config["model"], models, f"{instruction}: {config['model']}")


class _Reply:
    def __init__(self, text):
        self.content = [type("Block", (), {"type": "text", "text": text})()]


class _FakeClient:
    """A message-API stand-in. `messages.create` returns a scripted reply per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

        class Messages:
            def __init__(self, outer):
                self.outer = outer

            def create(self, **kwargs):
                self.outer.calls.append(kwargs)
                text = self.outer.replies.pop(0) if self.outer.replies else '{"verdict":"pass","findings":[]}'
                return _Reply(text)

        self.messages = Messages(self)


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.corrections = self.root / "corrections"

    def tearDown(self):
        self._tmp.cleanup()

    def document(self):
        return {
            "task_type": "rotate", "instruction": "Rotate the mug",
            "instruction_obj_to_replace": "mug", "instruction_verb_to_replace": "rotate",
            "main_objects": [{
                "type": "DatasetObject", "name": "mug", "category": "mug", "model": "m2",
                "bounding_box": [0.05, 0.05, 0.05], "orientation": [0.0, 0.5, 0.0, 0.866],
                "relative_bbox_position": [0.2, 0.2, 0.075],
            }],
            "target_objects": [], "distractors": [], "immutables": [],
        }

    def record(self):
        return {
            "task": "fixture", "support_z": 0.0, "renders": [],
            "objects": [{"name": "mug", "present": True, "settled_position": [0.2, 0.2, 0.075],
                         "authored_position": [0.2, 0.2, 0.075], "drift_xy": 0.0, "bbox": [0.05, 0.05, 0.05]}],
            "compute_findings": [],
        }

    def test_parse_accepts_clean_json_and_fenced_lowercase(self):
        self.assertEqual(review.parse_findings('{"verdict":"pass","findings":[]}')["verdict"], "pass")
        fenced = review.parse_findings(
            '```json\n{"verdict":"fail","findings":[{"code":"orientation","object":"mug"}]}\n```'
        )
        self.assertEqual(fenced["findings"][0]["code"], "ORIENTATION")

    def test_parse_defaults_to_pass_on_garbage(self):
        for text in ("not json", "", "{", "[]", '{"verdict":"fail","findings":"nope"}'):
            self.assertEqual(review.parse_findings(text)["verdict"], "pass", text)

    def test_parse_drops_codes_outside_the_closed_set(self):
        parsed = review.parse_findings(
            '{"verdict":"fail","findings":[{"code":"MADE_UP","object":"mug"},'
            '{"code":"SCALE","object":"mug","reason":"big"}]}'
        )
        self.assertEqual([item["code"] for item in parsed["findings"]], ["SCALE"])

    def test_parse_drops_findings_that_name_no_object(self):
        parsed = review.parse_findings('{"verdict":"fail","findings":[{"code":"SCALE"}]}')
        self.assertEqual(parsed["verdict"], "pass")

    def test_make_upright_applies_and_records_a_correction(self):
        document = self.document()
        self.assertNotEqual(document["main_objects"][0]["orientation"], [0.0, 0.0, 0.0, 1.0])
        client = _FakeClient(['{"verdict":"fail","findings":[{"code":"ORIENTATION","object":"mug","reason":"tipped"}]}'])
        result = review.review_once(
            self.record(), document, review.config_text(document), [],
            client=client, corrections_dir=self.corrections,
        )
        self.assertEqual(document["main_objects"][0]["orientation"], [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(result["applied"])
        self.assertEqual(len(list(self.corrections.glob("*.json"))), 1)

    def test_reduce_size_is_uniform_never_per_axis(self):
        document = self.document()
        document["main_objects"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]
        client = _FakeClient(['{"verdict":"fail","findings":[{"code":"SCALE","object":"mug","reason":"big"}]}'])
        review.review_once(
            self.record(), document, review.config_text(document), [],
            client=client, corrections_dir=self.corrections,
        )
        bbox = document["main_objects"][0]["bounding_box"]
        self.assertEqual(len(set(round(value / original, 6) for value, original in
                                          zip(bbox, (0.05, 0.05, 0.05)))), 1,
                         f"per-axis scaling detected in {bbox}")

    def test_a_finding_naming_an_absent_object_is_skipped_not_applied(self):
        document = self.document()
        client = _FakeClient(['{"verdict":"fail","findings":[{"code":"SCALE","object":"ghost","reason":"x"}]}'])
        result = review.review_once(
            self.record(), document, review.config_text(document), [],
            client=client, corrections_dir=self.corrections,
        )
        self.assertFalse(result["applied"])
        self.assertIn("no authored object", result["skipped"][0]["result"])
        self.assertEqual(list(self.corrections.glob("*.json")), [])

    def test_a_pass_verdict_stops_the_loop_after_one_iteration(self):
        client = _FakeClient(['{"verdict":"pass","findings":[]}'])
        outcome = review.review_and_patch(
            self.record(), self.document(), "yaml", [], client=client, corrections_dir=self.corrections,
        )
        self.assertEqual(outcome["iterations"], 1)
        self.assertEqual(len(client.calls), 1)

    def test_loop_never_exceeds_the_bounded_iteration_count(self):
        persistent = '{"verdict":"fail","findings":[{"code":"SCALE","object":"mug","reason":"big"}]}'
        client = _FakeClient([persistent] * 10)
        outcome = review.review_and_patch(
            self.record(), self.document(), "yaml", [], client=client, corrections_dir=self.corrections,
        )
        self.assertLessEqual(outcome["iterations"], review.MAX_ITERATIONS)
        self.assertLessEqual(len(client.calls), review.MAX_ITERATIONS)

    def test_missing_anthropic_package_is_reported_not_raised(self):
        import builtins

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guard
        try:
            result = review.review_once(self.record(), self.document(), "yaml", [], client=None)
        finally:
            builtins.__import__ = real_import
        self.assertIn("anthropic", result.get("skipped", ""))

    def test_review_prompt_forbids_re_reporting_computed_findings(self):
        record = self.record()
        record["compute_findings"] = [{"code": "UNSTABLE", "object": "mug", "reason": "moved"}]
        request = review.build_review_request(record, "yaml: {}", [])
        text = request["content"][-1]["text"]
        self.assertIn("UNSTABLE", text)
        self.assertIn("do NOT report these again", text)


class RenderHarnessTest(unittest.TestCase):
    """The harness runs in the container, but its pure functions must be host-testable.

    Every test here exists because a bug of its own class shipped in `render_review.py` and no
    existing test could have caught it: the harness had no coverage past `stability_findings`, so a
    `TypeError` on the very first line of `review_one()` -- a wrong constructor kwarg -- meant the
    GPU stage had literally never executed, while 48 agent tests stayed green.
    """

    def test_module_imports_without_omnigibson(self):
        from tooling.task_authoring import render_review

        self.assertTrue(callable(render_review.stability_findings))

    def test_unstable_drift_is_a_finding(self):
        from tooling.task_authoring import render_review

        rows = [{"name": "a", "present": True, "settled_position": [0, 0, 0.075],
                 "authored_position": [0, 0, 0.075], "drift_xy": 0.05, "drift_z": 0.0,
                 "bbox": [0.05, 0.05, 0.05]}]
        codes = [item["code"] for item in render_review.stability_findings(rows, 0.0)]
        self.assertIn("UNSTABLE", codes)

    def test_absent_object_is_missing(self):
        from tooling.task_authoring import render_review

        codes = [item["code"] for item in render_review.stability_findings([{"name": "b", "present": False}], 0.0)]
        self.assertEqual(codes, ["MISSING"])

    @staticmethod
    def _resting(name, low, **extra):
        return {"name": name, "present": True, "authored": True,
                "settled_position": [0, 0, low + 0.025], "authored_position": [0, 0, low + 0.025],
                "drift_xy": 0.0, "drift_z": 0.0, "bbox": [0.05, 0.05, 0.05], "aabb_low_z": low,
                **extra}

    def test_buried_object_is_penetration(self):
        from tooling.task_authoring import render_review

        rows = [self._resting(name, 0.85) for name in ("a", "b", "c")]
        rows.append(self._resting("sunk", 0.80))
        findings = render_review.stability_findings(rows, 1.05)
        self.assertEqual([(item["code"], item["object"]) for item in findings], [("PENETRATION", "sunk")])

    def test_spawn_height_above_the_table_is_not_penetration(self):
        """Regression: scenes.yaml `z` is the spawn height, ~0.2 m above the real table in some
        scenes. Treating it as the table top reported every object of all 11 REALM_DROID10 tasks as
        sunk 18-20 cm. Objects resting together on the real surface must be clean."""
        from tooling.task_authoring import render_review

        rows = [self._resting(name, 0.85 + 0.002 * index) for index, name in enumerate("abcd")]
        self.assertEqual(render_review.stability_findings(rows, 1.05), [])
        self.assertAlmostEqual(render_review.measured_support_z(rows), 0.852)

    def test_floor_standing_fixtures_do_not_define_the_table(self):
        """Drawer scenes author a table support on the floor and a lamp above the table; both pulled
        the measured surface to z=-0.001, so every table object looked sunk."""
        from tooling.task_authoring import render_review

        rows = [self._resting(name, 0.85) for name in ("a", "b", "c")]
        floor = self._resting("table_support", -0.001)
        floor["authored_position"] = [0, 0, 0.36]
        rows.append(floor)
        self.assertAlmostEqual(render_review.measured_support_z(rows, 0.9), 0.85)
        self.assertEqual(render_review.stability_findings(rows, 0.9), [])

    def test_declared_resting_object_is_not_floating(self):
        from tooling.task_authoring import render_review

        lid = {"name": "lid", "present": True, "authored": True,
               "settled_position": [0, 0, 1.0], "authored_position": [0, 0, 1.0],
               "drift_xy": 0.0, "drift_z": 0.0, "bbox": [0.2, 0.2, 0.06]}
        self.assertIn("FLOATING", [f["code"] for f in render_review.stability_findings([lid], 0.8)])
        self.assertEqual(render_review.stability_findings([lid], 0.8, resting={"lid"}), [])

    def test_object_authored_on_the_drop_plane_is_not_a_failed_pack(self):
        """pick_spoon authors the teaspoon at relative z 0.10 on a plate; stack_cubes authors cube4
        there too. Same height as placement's fallback, but the pose is the authored one."""
        from tooling.task_authoring import render_review

        rows = [{"name": "teaspoon", "present": True, "authored": True, "explicitly_placed": True,
                 "settled_position": [0.1, 0.2, 0.03], "authored_position": [0.1, 0.2, 0.1],
                 "drift_xy": 0.0, "drift_z": -0.07, "bbox": [0.19, 0.04, 0.01]}]
        self.assertNotIn("DROPPED", [item["code"] for item in render_review.stability_findings(rows, 0.0)])

    def test_explicit_placement_is_read_from_the_live_config(self):
        from tooling.task_authoring import render_review

        spawn = [-4.7, -4.3, -2.0, -1.5, 1.05]
        authored = {"relative_bbox_position": [0.15, 0.47, 0.1], "position": [-4.55, -1.53, 1.15]}
        replaced = {"relative_bbox_position": [0.15, 0.47, 0.1], "position": [-4.6, -1.8, 1.15]}
        self.assertTrue(render_review._explicitly_placed(authored, spawn))
        self.assertFalse(render_review._explicitly_placed(replaced, spawn))
        self.assertFalse(render_review._explicitly_placed({"position": [0, 0, 0]}, spawn))

    def test_resting_object_produces_no_finding(self):
        """The clean case must be clean. A harness that always fails is as useless as one that
        always passes, and 100 false FAILs would have been read as 100 bad layouts."""
        from tooling.task_authoring import render_review

        rows = [{"name": "a", "present": True, "authored": True,
                 "settled_position": [-4.62, -1.9, 1.075],
                 "authored_position": [-4.62, -1.9, 1.075], "drift_xy": 0.0, "drift_z": 0.0,
                 "bbox": [0.05, 0.05, 0.05]}]
        self.assertEqual(render_review.stability_findings(rows, 1.05), [])

    def test_scene_fixture_absence_is_not_reported(self):
        """`env.cfg["objects"]` carries the scene's own fixtures, and `scene_setup` removes some of
        them on purpose. Reporting those as MISSING failed every config in a scene with `to_remove`.
        """
        from tooling.task_authoring import render_review

        codes = [item["code"] for item in render_review.stability_findings(
            [{"name": "cabinet_1", "present": False, "authored": False}], 0.0)]
        self.assertEqual(codes, [])

    def test_a_object_the_solver_could_not_pack_is_dropped_not_unstable(self):
        """placement.py drops an unpackable object from support + DROP_HEIGHT instead of failing.
        The authored z is the signature (the object has fallen, so its drift reads negative), and
        `og.log.error` about it is invisible in the container, so this finding is the only record."""
        from tooling.task_authoring import render_review

        rows = [{"name": "a", "present": True, "authored": True,
                 "settled_position": [0.1, 0.2, 0.025], "authored_position": [0.1, 0.2, 0.1],
                 "drift_xy": 0.0, "drift_z": -0.075, "bbox": [0.05, 0.05, 0.05]}]
        codes = [item["code"] for item in render_review.stability_findings(rows, 0.0)]
        self.assertEqual(codes, ["DROPPED"])

    def test_object_authored_at_rest_height_is_never_dropped(self):
        """bbox_z = 0.10 makes the drop plane and the correct resting height identical. An object
        authored there that stays put is fine, and must not be reported."""
        from tooling.task_authoring import render_review

        rows = [{"name": "a", "present": True, "authored": True,
                 "settled_position": [0.1, 0.2, 0.1], "authored_position": [0.1, 0.2, 0.1],
                 "drift_xy": 0.0, "drift_z": 0.0, "bbox": [0.05, 0.05, 0.10]}]
        self.assertEqual(render_review.stability_findings(rows, 0.0), [])

    def test_authored_names_covers_every_role(self):
        """Immutables ride in the distractor list downstream but are declared separately, so a probe
        that only read main+target would judge scene objects and skip authored fixtures."""
        from tooling.task_authoring import render_review

        names = render_review.authored_names({
            "main_objects": [{"name": "mug"}], "target_objects": [{"name": "plate"}],
            "distractors": [{"name": "apple"}], "immutables": [{"name": "pot"}],
        })
        self.assertEqual(names, {"mug", "plate", "apple", "pot"})

    def test_collect_rgb_walks_the_nested_obs_tree(self):
        """Obs is nested -- `obs['external']['external_sensor0']['rgb']` -- so a flat scan for a
        top-level key starting with 'rgb' matched nothing and the harness wrote zero images."""
        from tooling.task_authoring import render_review

        obs = {
            "external": {"external_sensor0": {"rgb": "A"}, "external_sensor1": {"rgb": "B"}},
            "DROID_mounted": {"DROID_mounted:base_link:Camera:0": {"rgb": "C"},
                              "proprio": [0.0]},
        }
        found = dict(render_review.collect_rgb(obs))
        self.assertEqual(sorted(found), [
            "DROID_mounted/DROID_mounted:base_link:Camera:0",
            "external/external_sensor0", "external/external_sensor1",
        ])
        self.assertEqual(found["external/external_sensor0"], "A")


class _FakeObject:
    """A prim whose asset origin sits well below its geometry, as most DROID100 assets do."""

    category = "teddy_bear"

    def __init__(self, origin, low, high):
        self._origin, self.aabb = origin, (np.asarray(low, float), np.asarray(high, float))

    def get_position_orientation(self, frame="scene"):
        return np.asarray(self._origin, float), np.asarray([0, 0, 0, 1], float)


class _FakeScene:
    def __init__(self, objects):
        self.objects = objects

    def object_registry(self, key, name):
        return self.objects.get(name)


class _FakeProbeEnv:
    def __init__(self, cfg, objects):
        self.cfg, self.spawn_bbox = cfg, None
        self.omnigibson_env = self
        self.scene = _FakeScene(objects)


class SettleDriftTest(unittest.TestCase):
    """Drift is live-before vs live-after, never live origin vs authored bbox centre.

    The full DROID100 render failed 27 configs, 26 of them on UNSTABLE objects that were visibly
    where they belonged: the authored `position` is a bbox centre, the live pose an asset origin,
    and their difference is a property of the mesh, not of the layout.
    """

    def test_object_that_did_not_move_has_no_drift_whatever_its_origin(self):
        from tooling.task_authoring import render_review

        cfg = {"main_objects": [{"name": "bear"}],
               "objects": [{"name": "bear", "position": [1.0, 1.0, 1.0], "bounding_box": [0.2, 0.2, 0.3]}]}
        bear = _FakeObject(origin=[0.0, 0.0, 0.5], low=[-0.1, -0.1, 0.7], high=[0.1, 0.1, 1.0])
        env = _FakeProbeEnv(cfg, {"bear": bear})
        before = render_review.snapshot(env)
        rows = render_review.probe(env, before)
        self.assertEqual(rows[0]["drift_xy"], 0.0)
        self.assertEqual(rows[0]["drift_z"], 0.0)
        self.assertEqual(rows[0]["drift_reference"], "aabb_center")
        self.assertEqual(render_review.stability_findings(rows, 1.2, resting={"bear"}), [])

    def test_object_that_slid_is_unstable(self):
        from tooling.task_authoring import render_review

        before = {"aabb_center": [0.0, 0.0, 0.9], "origin": [0, 0, 0.5]}
        after = {"aabb_center": [0.03, 0.04, 0.88], "origin": [0.03, 0.04, 0.48]}
        drift = render_review.settle_drift(before, after)
        self.assertAlmostEqual(drift["drift_xy"], 0.05)
        self.assertAlmostEqual(drift["drift_z"], -0.02)

    def test_origin_is_the_fallback_and_no_snapshot_means_no_drift(self):
        from tooling.task_authoring import render_review

        self.assertEqual(render_review.settle_drift(None, {"origin": [0, 0, 0]}), {})
        drift = render_review.settle_drift({"origin": [0, 0, 0]}, {"origin": [0.0, 0.02, 0]})
        self.assertEqual(drift["drift_reference"], "origin")
        self.assertAlmostEqual(drift["drift_xy"], 0.02)

    def test_review_snapshots_before_the_settle(self):
        source = (Path(__file__).resolve().parents[1] / "render_review.py").read_text(encoding="utf-8")
        body = source[source.index("def review_one("):]
        self.assertLess(body.index("snapshot(env)"), body.index("settle(env"))
        self.assertIn("probe(env, before)", body)

    def test_floor_penetration_is_checked_when_no_surface_can_be_measured(self):
        """Drawer scenes author no objects on the spawn plane, so the measured surface is None and
        PENETRATION silently never ran. The floor is still a hard bound."""
        from tooling.task_authoring import render_review

        drawer = {"name": "cabinet", "present": True, "authored": True,
                  "authored_position": [0, 0, 0.4], "bbox": [0.5, 0.5, 0.8],
                  "drift_xy": 0.0, "drift_z": 0.0, "aabb_low_z": -0.05}
        self.assertIsNone(render_review.measured_support_z([drawer], 0.8))
        codes = [f["code"] for f in render_review.stability_findings([drawer], 0.8)]
        self.assertEqual(codes, ["PENETRATION"])
        drawer["aabb_low_z"] = -0.001
        self.assertEqual(render_review.stability_findings([drawer], 0.8), [])


class _FakeEnv:
    """Just enough of an env for `render()`: one method, no simulator."""

    def __init__(self, obs):
        self._obs = obs
        self.omnigibson_env = self

    def get_obs(self):
        return self._obs, {}


class RenderFailureSurfacingTest(unittest.TestCase):
    """A review stage that silently produces no images is worse than one that fails loudly: the
    vision model reviews nothing, reports a clean scene, and the family ships unrendered."""

    def test_render_returns_errors_and_never_a_bare_empty_success(self):
        from tooling.task_authoring import render_review

        written, errors = render_review.render(_FakeEnv({"proprio": [0.0]}), Path("."), "t", "t000")
        self.assertEqual(written, [])
        self.assertTrue(errors, "an obs with no rgb leaf must report why nothing was written")

    def test_render_reports_a_broken_obs_instead_of_swallowing_it(self):
        from tooling.task_authoring import render_review

        class Broken(_FakeEnv):
            def get_obs(self):
                raise RuntimeError("sim not playing")

        written, errors = render_review.render(Broken({}), Path("."), "t", "t000")
        self.assertEqual(written, [])
        # On a host with no omnigibson there is also a render error; get_obs' own failure must be
        # reported *in addition to* it, not in place of it.
        self.assertTrue(any("get_obs failed" in e for e in errors), errors)


class RenderHarnessStaticContractTest(unittest.TestCase):
    """Static guards on the container-only call sites.

    These are the parts of the harness that no host test can execute -- they need a built scene --
    but whose correctness is checkable from source. `perturbation=` vs `perturbations=` passed the
    whole review process and only blew up inside the container on the first config, which is the
    most expensive place to discover a typo.
    """

    HARNESS = Path(__file__).resolve().parents[1] / "render_review.py"
    ENV_DYNAMIC = (Path(__file__).resolve().parents[3] / "realm" / "environments"
                   / "env_dynamic.py")

    def _harness(self):
        return self.HARNESS.read_text(encoding="utf-8")

    def test_harness_kwargs_exist_on_the_constructor(self):
        import ast

        accepted = None
        for node in ast.walk(ast.parse(self.ENV_DYNAMIC.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.ClassDef) and node.name == "RealmEnvironmentDynamic"
                    and not accepted):
                for item in node.body:
                    if (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
                        accepted = {a.arg for a in item.args.args}
        self.assertIsNotNone(accepted, "could not find RealmEnvironmentDynamic.__init__")

        tree = ast.parse(self._harness())
        called = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and getattr(node.func, "id", None) == "RealmEnvironmentDynamic"]
        self.assertEqual(len(called), 1, "expected exactly one construction site")
        unknown = [kw.arg for kw in called[0].keywords if kw.arg not in accepted]
        self.assertEqual(unknown, [],
                         f"harness passes kwargs the constructor does not accept: {unknown}. "
                         f"Accepted: {sorted(accepted)}")

    def test_harness_passes_perturbations_as_a_list(self):
        """The constructor iterates `active_perturbations`; a bare string is the wrong shape even
        when the name is right."""
        import ast

        for node in ast.walk(ast.parse(self._harness())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RealmEnvironmentDynamic":
                perturbations = next(kw for kw in node.keywords if kw.arg == "perturbations")
                self.assertIsInstance(perturbations.value, ast.List)
                return
        self.fail("no RealmEnvironmentDynamic construction found")

    def test_harness_never_shuts_the_simulator_down(self):
        """`og.shutdown()` is terminal, so calling it inside `review_one` made config #2 of a family
        unwinnable -- and the JSON for config #1 was never written either, because the write sits
        after the `finally`. One process per config is the contract."""
        import ast

        calls = [node.func.attr for node in ast.walk(ast.parse(self._harness()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("shutdown", calls,
                         "og.shutdown() is process-terminal; --family must spawn a child per config")

    def test_family_mode_fans_out_to_children(self):
        source = self._harness()
        self.assertIn("subprocess", source, "--family must run each config in its own process")

    def test_child_is_invoked_with_flags_the_child_accepts(self):
        """A typo in the child's argv fails every config of a family identically, which reads as a
        broken family rather than a broken launcher."""
        import ast

        tree = ast.parse(self._harness())
        # argparse takes the flag as a POSITIONAL string, e.g. add_argument("--robot", ...).
        declared = {node.args[0].value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "add_argument"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)}
        asked = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_child_command":
                asked = {n.value for n in ast.walk(node)
                         if isinstance(n, ast.Constant) and isinstance(n.value, str)
                         and n.value.startswith("--")}
        self.assertTrue(asked, "could not read _child_command's flags")
        self.assertFalse(asked - declared, f"child asks for {sorted(asked - declared)}, which main rejects")

    def test_review_reads_the_frame_the_probe_writes(self):
        """`backfill_object_cfgs` / `set_scene_positions` / `vb_pose._place` all treat
        `cfg['objects'][i]['position']` as SCENE-frame. The probe must compare like with like: it
        once diffed a region-relative `relative_bbox_position` against a world pose, which reported
        ~5 m of drift on an object that never moved (Pomaria_1_int/Kitchen_Counter sits at
        -4.7, -2.0)."""
        import ast

        tree = ast.parse(self._harness())
        probe = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "probe")
        # Docstrings may name the wrong key to explain it; only a real read counts.
        read_keys = {node.args[0].value for node in ast.walk(probe)
                     if isinstance(node, ast.Call)
                     and getattr(node.func, "attr", None) == "get"
                     and node.args and isinstance(node.args[0], ast.Constant)
                     and isinstance(node.args[0].value, str)}
        self.assertIn("position", read_keys)
        self.assertNotIn("relative_bbox_position", read_keys,
                         "probe() must diff scene-frame `position`, not region-relative offsets")
        live = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "live_pose")
        frames = {kw.arg: kw.value.value for node in ast.walk(live)
                  if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get_position_orientation"
                  for kw in node.keywords}
        self.assertEqual(frames.get("frame"), "scene",
                         "live poses must be read in the frame the cfg positions are written in")


if __name__ == "__main__":
    unittest.main()
