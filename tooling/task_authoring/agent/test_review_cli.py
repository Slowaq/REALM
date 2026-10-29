"""The vision review loop: critic briefs, strict verdicts, and reopening sessions for the author."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tooling.task_authoring.agent import cli, layout, review_cli
from tooling.task_authoring.agent.review import MAX_ITERATIONS
from tooling.task_authoring.agent.test_agent import PUT_ROLES, build_fixture

REAL_SIZES = layout.real_sizes


def setUpModule():
    layout.real_sizes = lambda path=None: {}


def tearDownModule():
    layout.real_sizes = REAL_SIZES


def run_main(main, argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(argv)
    text = buffer.getvalue()
    try:
        return code, json.loads(text)
    except ValueError:
        return code, text


def ingest(verdicts, sessions):
    """Ingest prints one line per reopened task and a JSON summary last; return the summary."""
    _, out = run_main(review_cli.main, ["--verdicts", str(verdicts), "ingest", "--sessions", str(sessions)])
    return out if isinstance(out, dict) else json.loads(out.strip().splitlines()[-1])


RENDER_RECORD = {
    "task": "put_the_apple_in_the_bowl_abc123",
    "task_cfg_path": "REALM_DROID100/put_the_apple_in_the_bowl_abc123/default.yaml",
    "renders": ["out/x/x_cam1_t000.png", "out/x/x_cam1_t060.png", "out/x/x_wrist_t060.png"],
    "verdict": "PASS",
    "compute_findings": [{"code": "NOT_VISIBLE", "object": "apple", "severity": "soft"}],
    "visibility": {"apple": {"cam1": {"pixels": 0}}},
    "objects": [
        {"name": "apple", "role": "main", "category": "apple", "authored": True,
         "settled_aabb_low": [0, 0, 0], "settled_aabb_high": [0.08, 0.08, 0.075], "drift_xy": 0.001},
        {"name": "robot", "authored": False},
    ],
}


class CriticBriefTest(unittest.TestCase):
    def test_settled_images_are_the_last_timestep_only(self):
        self.assertEqual(review_cli.settled_images(RENDER_RECORD),
                         ["out/x/x_cam1_t060.png", "out/x/x_wrist_t060.png"])

    def test_brief_carries_measured_numbers_and_a_record_command(self):
        brief = review_cli.task_brief(RENDER_RECORD, "review_v7/droid100")
        measured = brief["measured"]
        self.assertEqual([o["name"] for o in measured["objects"]], ["apple"])  # the robot is not authored
        self.assertTrue(measured["objects"][0]["main"])
        self.assertEqual(measured["objects"][0]["settled_size_m"], [0.08, 0.08, 0.075])
        self.assertEqual(measured["findings_from_numbers"][0]["code"], "NOT_VISIBLE")
        self.assertIn("--render-id review_v7/droid100", brief["record_with"])

    def test_prompt_skips_tasks_already_judged_for_this_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            review_dir = Path(tmp) / "review_v7" / "droid100"
            review_dir.mkdir(parents=True)
            (review_dir / "a.json").write_text(json.dumps({**RENDER_RECORD, "task": "a"}))
            (review_dir / "b.json").write_text(json.dumps({**RENDER_RECORD, "task": "b"}))
            (review_dir / "_summary.json").write_text("{}")
            verdicts = Path(tmp) / "verdicts"
            verdicts.mkdir()
            (verdicts / "a.json").write_text(json.dumps({"render_id": "review_v7/droid100"}))
            (verdicts / "b.json").write_text(json.dumps({"render_id": "review_v6/droid100"}))
            _, text = run_main(review_cli.main, ["--verdicts", str(verdicts), "prompt",
                                                 "--review-dir", str(review_dir), "--only-unreviewed"])
            self.assertIn("1 task(s)", text)
            self.assertIn('"task": "b"', text)
            self.assertNotIn('"task": "a"', text)


class RecordTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.verdicts = self.root / "verdicts"
        self._tasks_root = review_cli.TASKS_ROOT
        review_cli.TASKS_ROOT = self.root / "tasks"
        folder = review_cli.TASKS_ROOT / "REALM_DROID100" / "t1"
        folder.mkdir(parents=True)
        (folder / "default.yaml").write_text(yaml.safe_dump({"provenance": {"task_id": "abc"}}))

    def tearDown(self):
        review_cli.TASKS_ROOT = self._tasks_root
        self._tmp.cleanup()

    def record(self, verdict, render="r1"):
        return run_main(review_cli.main, ["--verdicts", str(self.verdicts), "record", "t1",
                                          "--render-id", render, verdict])

    def stored(self):
        return json.loads((self.verdicts / "t1.json").read_text())

    def test_a_fail_is_stored_with_the_session_key_from_the_config(self):
        code, out = self.record('{"verdict": "fail", "findings": [{"code": "low_contrast", "object": "pen", '
                                '"reason": "black on dark wood", "suggestion": "a lighter pen"}]}')
        self.assertEqual(code, 0)
        self.assertTrue(out["ok"])
        stored = self.stored()
        self.assertEqual(stored["task_id"], "abc")
        self.assertEqual(stored["verdict"], "fail")
        self.assertEqual(stored["findings"][0]["code"], "LOW_CONTRAST")
        self.assertEqual(stored["findings"][0]["suggestion"], "a lighter pen")
        self.assertEqual(stored["iteration"], 1)

    def test_unknown_codes_and_unreadable_verdicts_are_refused_not_passed(self):
        for bad in ('{"verdict": "fail", "findings": [{"code": "UGLY", "object": "pen"}]}',
                    "looks fine to me",
                    '{"verdict": "fail", "findings": []}',
                    '{"verdict": "fail", "findings": [{"code": "SCALE"}]}'):
            code, out = self.record(bad)
            self.assertEqual(code, 2, bad)
            self.assertFalse(out["ok"])
        self.assertFalse((self.verdicts / "t1.json").exists())

    def test_rerecording_a_render_keeps_the_iteration_a_new_render_advances_it(self):
        self.record('{"verdict": "pass", "findings": []}')
        self.record('{"verdict": "fail", "findings": [{"code": "SCALE", "object": "pen"}]}')
        self.assertEqual(self.stored()["iteration"], 1)
        self.record('{"verdict": "pass", "findings": []}', render="r2")
        self.assertEqual(self.stored()["iteration"], 2)


class IngestLoopTest(unittest.TestCase):
    """Fail -> reopened with findings -> author fixes and resubmits -> second fail -> escalated."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = build_fixture(self.root / "ds")
        self.sessions = self.root / "sessions"
        self.verdicts = self.root / "verdicts"
        self.verdicts.mkdir()
        self.base = ["--dataset", str(self.dataset), "--sessions", str(self.sessions)]

    def tearDown(self):
        self._tmp.cleanup()

    def cli(self, *argv):
        code, out = run_main(cli.main, [*self.base, *argv])
        self.assertEqual(code, 0)
        return out

    def submit(self, key, extra=None):
        apple = self.cli("call", key, "search_assets", '{"query": "apple", "role": "main"}')
        bowl = self.cli("call", key, "search_assets", '{"query": "bowl", "role": "target"}')
        roles = {
            "task_type": "put", "instruction": "Put the apple in the bowl",
            "main": {"name": "apple", "category": "apple", "model": apple["candidates"][0]["model"]},
            "target": {"name": "bowl", "category": "bowl", "model": bowl["candidates"][-1]["model"]},
            **(extra or {}),
        }
        self.assertTrue(self.cli("call", key, "propose_layout", json.dumps(roles))["ok"])
        self.assertTrue(self.cli("call", key, "submit_task", '{"decisions": []}')["ok"])

    def review_fails(self, key, render):
        (self.verdicts / "t.json").write_text(json.dumps({
            "task": "t", "task_id": key, "verdict": "fail", "render_id": render, "iteration": 1,
            "findings": [{"code": "NOT_VISIBLE", "object": "apple", "reason": "behind the bowl"}]}))
        return ingest(self.verdicts, self.sessions)

    def state(self, key):
        return json.loads((self.sessions / f"{key}.json").read_text())

    def test_review_loop_reopens_then_escalates(self):
        key = self.cli("start", "Put the apple in the bowl", "--ranking-id", "R")["task_id"]
        self.submit(key)
        self.assertEqual(self.review_fails(key, "r1")["reopened"], 1)
        self.assertIsNone(self.state(key)["outcome"])
        # Ingesting the same verdict twice changes nothing.
        self.assertEqual(self.review_fails(key, "r1")["reopened"], 0)
        self.assertEqual(len(self.state(key)["reviews"]), 1)

        resumed = self.cli("start", "Put the apple in the bowl", "--ranking-id", "R")
        self.assertEqual(resumed["review_findings"][0]["code"], "NOT_VISIBLE")
        first = self.cli("call", key, "search_assets", '{"query": "apple", "role": "main"}')
        self.assertIn("review_findings", first)
        second = self.cli("call", key, "search_assets", '{"query": "apple", "role": "main"}')
        self.assertNotIn("review_findings", second)
        self.assertEqual(run_main(cli.main, [*self.base, "status", "--quiet"])[1]["reviewed"], 1)

        self.submit(key, {"camera_skip": 1})
        self.assertNotIn("review_findings", self.state(key))
        self.assertEqual(self.state(key)["outcome"]["status"], "submitted")

        out = self.review_fails(key, "r2")
        if MAX_ITERATIONS == 2:
            self.assertEqual(out["escalated"], 1)
            self.assertEqual(self.state(key)["outcome"]["status"], "escalated")
            self.assertFalse(self.cli("call", key, "validate_draft")["ok"])
            self.cli("export", "--output", str(self.root / "out"), "--report", str(self.root / "r.json"))
            self.assertEqual(list((self.root / "out").glob("*/default.yaml")), [])

    def test_a_pass_is_recorded_and_leaves_the_task_submitted(self):
        key = self.cli("start", "Put the apple in the bowl", "--ranking-id", "R")["task_id"]
        self.submit(key)
        (self.verdicts / "t.json").write_text(json.dumps({
            "task": "t", "task_id": key, "verdict": "pass", "render_id": "r1", "iteration": 1, "findings": []}))
        self.assertEqual(ingest(self.verdicts, self.sessions)["passed"], 1)
        self.assertEqual(self.state(key)["outcome"]["status"], "submitted")
        self.assertEqual(self.state(key)["reviews"][0]["verdict"], "pass")


class CameraSkipTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset = build_fixture(Path(self._tmp.name) / "ds")

    def tearDown(self):
        self._tmp.cleanup()

    def cameras(self, **extra):
        result = layout.build_document({**PUT_ROLES, **extra}, dataset=self.dataset)
        return result["document"]["camera_extrinsics"], result["audit"]

    def test_skip_zero_keeps_the_cameras_and_skip_one_moves_to_another_framing_pair(self):
        base, _ = self.cameras()
        same, _ = self.cameras(camera_skip=0)
        moved, audit = self.cameras(camera_skip=1)
        self.assertEqual(base, same)
        self.assertNotEqual(base, moved)
        self.assertEqual(audit["camera_skip"], 1)
        self.assertIsInstance(audit["camera_redraws"], int)
        self.assertTrue(all(layout.looks_at_workspace(pose) for pose in moved.values()))


if __name__ == "__main__":
    unittest.main()
