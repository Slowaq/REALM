"""The generator agent's six tools.

The surface is deliberately small, and what is ABSENT is the point: there is no `set_position`,
no `set_scale`, no `write_yaml`. A model cannot take over the solver's job because no tool accepts
a coordinate. Everything positional is produced by `layout.py` calling the same solver the batch
generator uses.

Each tool returns JSON-serializable data. `dispatch()` is transport-agnostic so the same functions
serve the Anthropic tool-runner, the offline dry-run driver, and the tests.

`report_ungroundable` matters more than it looks. Without an explicit give-up path a model
substitutes a plausible-sounding asset rather than admit one does not exist -- which is how
"orange-handled tool" became a silent mismatch that a human had to catch and hard-code.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from tooling.task_authoring.agent import catalog as asset_catalog
from tooling.task_authoring.agent import layout
from tooling.task_authoring.agent.corrections import DEFAULT_CORRECTIONS_DIR, record_correction
from tooling.task_authoring.validation import validate


TASK_TYPE_ENUM = list(layout.TASK_TYPES)
#: Nouns that name a PrimitiveObject cube, which the config can colour exactly.
PRIMITIVE_WORDS = frozenset({"block", "cube"})

ASSET_REF = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "unique name across all roles in this task"},
        "category": {"type": "string", "description": "a category returned by search_assets"},
        "model": {"type": "string", "description": "a model id returned by search_assets"},
        "primitive": {
            "type": "string",
            "description": "set to 'block' for a solid-coloured cube instead of a catalogue asset",
        },
        "rgba": {
            "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
            "description": "colour for a primitive, 0..1",
        },
        "extent": {"type": "number", "description": "primitive cube edge in metres (0.01..0.2)"},
    },
    "required": ["name"],
    "additionalProperties": False,
}


def tool_schemas() -> list[dict]:
    """The six tool definitions, in Anthropic tool-use form."""
    return [
        {
            "name": "search_assets",
            "description": (
                "Search the indexed OmniGibson catalogue. Returns real assets only: a category "
                "absent from the result does not exist and must not be invented. Bounding boxes "
                "are the asset's natural extents in metres, XYZ, before any authoring scale."
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
        },
        {
            "name": "list_scene_regions",
            "description": (
                "List the scene/support regions a task can be authored on, with their dimensions. "
                "Pass a region_index to propose_layout to choose one."
            ),
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
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
                    "task_type": {"type": "string", "enum": TASK_TYPE_ENUM},
                    "instruction": {"type": "string", "description": "the final instruction text"},
                    "main": ASSET_REF,
                    "target": ASSET_REF,
                    "source": {
                        **ASSET_REF,
                        "description": (
                            "the object a pick task removes from, or the support a stack uses; "
                            "authored as an immutable with the declared relation"
                        ),
                    },
                    "distractors": {"type": "array", "items": ASSET_REF, "maxItems": 4},
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
                    "region_index": {
                        "type": "integer",
                        "description": (
                            "from list_scene_regions. Omit it unless the task needs a specific "
                            "surface: the solver then picks a region from this task's own seed, "
                            "which keeps the family spread across scenes"
                        ),
                    },
                    "decisions": {
                        "type": "array", "items": {"type": "string"},
                        "description": "short notes on substitutions or instruction rewrites",
                    },
                },
                "required": ["task_type", "instruction", "main"],
                "additionalProperties": False,
            },
        },
        {
            "name": "validate_draft",
            "description": (
                "Run REALM's authoring rules over the last layout proposed. Returns structured "
                "findings; each carries a `fix` with the concrete correction when the rule "
                "determines one. An empty error list means the draft is shippable."
            ),
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "submit_task",
            "description": (
                "Submit the validated draft as this instruction's answer. Rejected if the draft "
                "does not validate clean, with the findings returned."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array", "items": {"type": "string"},
                        "description": "what you substituted, shortened or declined, and why",
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "report_ungroundable",
            "description": (
                "Give up on this instruction honestly. Use when no real asset can ground a noun "
                "the instruction depends on, or when placement is impossible for every honest "
                "role assignment. This is a correct answer, not a failure."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "blocking_terms": {
                        "type": "array", "items": {"type": "string"},
                        "description": "the instruction's nouns that could not be grounded",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    ]


class Session:
    """Per-instruction agent state: the catalogue, the last solved draft, and the outcome.

    Holding the solved document here rather than re-solving on every `validate_draft` call is what
    keeps the tool loop honest: the model can only validate and submit the draft that a
    `propose_layout` call actually produced.
    """

    def __init__(
        self,
        instruction: str,
        *,
        dataset: Path | None = None,
        seed: int = layout.__dict__.get("DEFAULT_SEED", 100),
        assets_by_category: dict | None = None,
        corrections_dir: Path | None = DEFAULT_CORRECTIONS_DIR,
        ranking_id: str | None = None,
        catalog: Path | None = None,
    ) -> None:
        self.instruction = instruction
        self.dataset = dataset if dataset is not None else layout.DEFAULT_DATASET
        # Each task draws from its own deterministic stream. One shared seed gave all tasks the same
        # camera pair and the same sampled clutter; deriving from the content key keeps every task
        # reproducible while making the family varied.
        self.seed = task_seed(seed, instruction, ranking_id)
        self.ranking_id = ranking_id
        self.corrections_dir = corrections_dir
        if assets_by_category is None:
            # Raises CatalogError rather than handing the agent an empty catalogue: an empty index
            # declines every instruction and reads as a grounding failure, not a missing dataset.
            assets_by_category, _ = asset_catalog.resolve(dataset=dataset, catalog=catalog)
        self.assets = assets_by_category
        self.documents: list[dict] = []
        self.document: dict | None = None
        self.submission: dict | None = None
        self.ungroundable: dict | None = None
        self.corrections_applied: list[dict] = []
        self.validation_rounds = 0

    @property
    def regions(self) -> list[dict]:
        return layout.usable_regions()

    def search_assets(self, query: str, role: str, max_footprint_xy=None) -> dict:
        query = str(query).strip().lower()
        exact = self.assets.get(query, [])
        category = query
        if not exact:
            # Word-boundary fallback so "spoon" reaches teaspoon and "can" reaches can_of_soda (not
            # box_of_cane_sugar) -- and the result still names the REAL category, so the model
            # cannot drift off the catalogue.
            match = asset_catalog.match_category(query, self.assets)
            matches = [match] if match else []
            if not matches and query.rstrip("s") in PRIMITIVE_WORDS:
                return {
                    "query": query, "role": role, "found": False, "candidates": [],
                    "note": (
                        "a block/cube is not a catalogue asset: author it as a primitive -- "
                        "{\"name\": \"yellow_block\", \"primitive\": \"block\", \"rgba\": [r, g, b, 1]} "
                        "in propose_layout. Its colour is guaranteed, so colour words may stay."
                    ),
                }
            if not matches:
                return {
                    "query": query, "role": role, "found": False,
                    "candidates": [],
                    "note": (
                        f"no indexed category matches {query!r}. It does not exist. Do not invent "
                        f"it: search a synonym, or call report_ungroundable."
                    ),
                }
            category = matches[0]
            exact = self.assets[category]
        candidates = []
        for asset in exact:
            bbox = [round(float(value), 5) for value in asset["bbox"]]
            if max_footprint_xy and (bbox[0] > float(max_footprint_xy[0]) or bbox[1] > float(max_footprint_xy[1])):
                continue
            candidates.append({"category": category, "model": str(asset["model"]), "bbox": bbox})
        candidates.sort(key=lambda item: (item["bbox"][0] * item["bbox"][1], item["model"]))
        return {
            "query": query, "role": role, "category": category, "found": bool(candidates),
            "candidates": candidates[:12],
            "note": (
                "use `category` and `model` verbatim in propose_layout"
                if candidates else "every candidate exceeded max_footprint_xy; try a larger ceiling"
            ),
        }

    def list_scene_regions(self) -> dict:
        return {
            "regions": [
                {
                    "region_index": index,
                    "scene": region["scene"],
                    "support": region["support"],
                    "width": round(float(region["width"]), 4),
                    "depth": round(float(region["depth"]), 4),
                }
                for index, region in enumerate(self.regions)
            ],
            "note": "pass region_index to propose_layout; out-of-range is reported, never clamped",
        }

    def propose_layout(self, **roles) -> dict:
        requested = roles.pop("region_index", None)
        decisions = roles.pop("decisions", None)
        if requested is None:
            # No region named: try the usable regions in a task-seeded order, so the family spreads
            # across scenes instead of every task landing on whichever region the agent likes.
            order = list(range(len(self.regions)))
            random.Random(self.seed + 2).shuffle(order)
        else:
            order = [int(requested)]
        result = None
        for position, region_index in enumerate(order):
            outcome = self._solve(roles, region_index)
            if outcome.get("ok") or position == len(order) - 1:
                result = outcome
                break
        if not result.get("ok"):
            return result
        return self._accept(result["result"], decisions)

    def _solve(self, roles: dict, region_index: int) -> dict:
        try:
            result = layout.build_document(
                json.loads(json.dumps(roles)),
                region_index=region_index,
                seed=self.seed,
                dataset=self.dataset,
                assets_by_category=self.assets,
                corrections_dir=self.corrections_dir,
                provenance={
                    "task_id": _task_id_for(self.instruction, self.ranking_id),
                    "ranking_id": self.ranking_id,
                    "original_instruction": self.instruction,
                },
            )
        except layout.LayoutError as error:
            return {"ok": False, "reason": str(error)}
        except ValueError as error:
            return {"ok": False, "reason": str(error)}
        return {"ok": True, "result": result}

    def _accept(self, result: dict, decisions) -> dict:
        self.document = result["document"]
        if self.ranking_id:
            self.document.setdefault("provenance", {})["ranking_id"] = self.ranking_id
        if decisions:
            self.document.setdefault("provenance", {})["decisions"] = [str(item) for item in decisions]
        self.documents.append(self.document)
        self.validation_rounds = 0
        return {
            "ok": True,
            "audit": result["audit"],
            "objects": _summarize(self.document),
            "note": (
                "geometry solved. Positions, orientations and scales are final -- do not restate "
                "them. Call validate_draft next."
            ),
        }

    def validate_draft(self) -> dict:
        if self.document is None:
            return {"ok": False, "reason": "no layout proposed yet; call propose_layout first"}
        self.validation_rounds += 1
        report = validate(self.document, task=str(self.instruction)[:60])
        errors = [finding for finding in report.as_dict()["findings"] if finding["severity"] == "error"]
        errors += self._semantic_findings()
        return {
            "ok": not errors,
            "error_count": len(errors),
            "findings": errors,
            "warnings": [finding for finding in report.as_dict()["findings"] if finding["severity"] == "warning"],
            "round": self.validation_rounds,
            "note": (
                "apply each error's `fix` through the tool that owns it, then validate again"
                if errors else "clean: call submit_task"
            ),
        }

    def _semantic_findings(self) -> list[dict]:
        """Agent-level checks the geometry validator cannot make, because they need the catalogue."""
        phantoms = asset_catalog.phantom_nouns(self.document or {}, self.assets)
        if not phantoms:
            return []
        return [{
            "code": "INSTRUCTION_PHANTOM_OBJECT",
            "severity": "error",
            "message": (
                f"the instruction names {phantoms} but no such object is in the scene. Either add "
                f"it as a role (a `source` for 'from X', a `target` for 'into/onto X') or rewrite "
                f"the instruction so it promises only objects the scene contains."
            ),
            "obj": None,
            "path": "instruction",
            "fix": {"remove_or_ground_words": phantoms},
        }]

    def submit_task(self, decisions=None) -> dict:
        if self.document is None:
            return {"ok": False, "reason": "no layout proposed yet; call propose_layout first"}
        report = validate(self.document, task=str(self.instruction)[:60])
        errors = [item for item in report.as_dict()["findings"] if item["severity"] == "error"]
        errors += self._semantic_findings()
        if errors:
            self.validation_rounds += 1
            return {
                "ok": False,
                "reason": f"draft has {len(errors)} error(s); fix them before submitting",
                "findings": errors,
            }
        if decisions:
            self.document.setdefault("provenance", {})["decisions"] = [str(item) for item in decisions]
        self.submission = {
            "instruction": self.instruction,
            "document": self.document,
            "validation": report.as_dict(),
        }
        return {"ok": True, "accepted": True, "task_id": _task_id_for(self.instruction, self.ranking_id)}

    def report_ungroundable(self, reason: str, blocking_terms=None) -> dict:
        self.ungroundable = {
            "instruction": self.instruction,
            "reason": str(reason),
            "blocking_terms": [str(item) for item in (blocking_terms or [])],
        }
        return {"status": "recorded", "note": "the instruction is dropped from the family"}

    def record_vision_correction(self, *, code: str, action: str, obj=None, to=None, reason="", iteration=1) -> Path:
        """Persist one applied correction against this task's content key."""
        if self.document is None:
            raise ValueError("no layout proposed yet")
        return record_correction(
            self.document, code=code, action=action, obj=obj, to=to, reason=reason,
            source="vision", iteration=iteration, corrections_dir=self.corrections_dir,
        )


def _summarize(document: dict) -> dict:
    def brief(config: dict) -> dict:
        return {
            "name": config.get("name"),
            "category": config.get("category") or config.get("primitive_type"),
            "model": config.get("model"),
            "bbox": config.get("bounding_box"),
        }

    return {
        "main": [brief(item) for item in document.get("main_objects") or []],
        "target": [brief(item) for item in document.get("target_objects") or []],
        "distractors": [brief(item) for item in document.get("distractors") or []],
        "immutables": [brief(item) for item in document.get("immutables") or []],
        "instruction": document.get("instruction"),
        "task_type": document.get("task_type"),
    }


def _task_id_for(instruction: str, ranking_id: str | None) -> str:
    """Content key of the ORIGINAL instruction -- the same key run.py caches and the store uses."""
    from tooling.task_authoring.agent.corrections import task_id

    return task_id(str(instruction), ranking_id)


def task_seed(base_seed: int, instruction: str, ranking_id: str | None) -> int:
    """Per-task RNG seed: the family's base seed offset by the task's content key."""
    return int(base_seed) + int(_task_id_for(instruction, ranking_id)[:8], 16) % 1_000_000


def dispatch(session: Session, name: str, arguments: dict | None) -> dict:
    """Route one tool call. Unknown tool names are reported, not raised: a model that hallucinates
    a tool should get a correctable error, not end the run."""
    arguments = arguments or {}
    handlers = {
        "search_assets": session.search_assets,
        "list_scene_regions": session.list_scene_regions,
        "propose_layout": session.propose_layout,
        "validate_draft": session.validate_draft,
        "submit_task": session.submit_task,
        "report_ungroundable": session.report_ungroundable,
    }
    handler = handlers.get(name)
    if handler is None:
        return {"error": f"unknown tool {name!r}; available: {', '.join(sorted(handlers))}"}
    return handler(**arguments)


def dumps(result: dict) -> str:
    return json.dumps(result, indent=2, default=str)
