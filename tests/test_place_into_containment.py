"""Pin what PLACE_INTO's containment test accepts and rejects, on the host, with no GPU.

TaskProgressionMixin._is_inside passes when realm.geometry.aabb_containment reports xy coverage of
at least OmniGibson's Overlaid.OVERLAP_AREA_PERCENTAGE (0.5 in 3.9.1, pinned here because importing
omnigibson boots Isaac) and a depth fraction of at least PLACE_INSIDE_MIN_DEPTH_FRACTION. Every
case is a box layout taken from a shipped `put` task; the rejections are the false successes the
old Inside-or-OnTop check credited.

    uv run python -m pytest -q tests/test_place_into_containment.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realm.config.shared import PLACE_INSIDE_MIN_DEPTH_FRACTION  # noqa: E402
from realm.geometry import aabb_containment  # noqa: E402

OG_OVERLAP_AREA_PERCENTAGE = 0.5


def box(center, extent):
    lo = [c - e / 2 for c, e in zip(center, extent)]
    hi = [c + e / 2 for c, e in zip(center, extent)]
    return lo, hi


def is_inside(obj, cavity):
    xy_coverage, depth_fraction = aabb_containment(*obj, *cavity)
    return xy_coverage >= OG_OVERLAP_AREA_PERCENTAGE and depth_fraction >= PLACE_INSIDE_MIN_DEPTH_FRACTION


MUG = box((0, 0, 0.04), (0.115, 0.090, 0.080))               # no meta link: whole AABB
BOWL_CAVITY = box((0, 0, 0.04), (0.18, 0.18, 0.06))          # fillable link of a 0.20x0.20x0.07 bowl
TRAY_CAVITY = box((0, 0, 0.03), (0.21, 0.33, 0.04))          # fillable link of a 0.23x0.35x0.05 tray
WINEGLASS_CAVITY = box((0, 0, 0.13), (0.08, 0.08, 0.09))     # glass bowl above a 0.085 stem


@pytest.mark.parametrize("obj, cavity", [
    # Scissors standing in the mug: more than half sticks out above the rim, so a volume-overlap
    # fraction (~0.38) would reject it.
    (box((0, 0, 0.11), (0.08, 0.02, 0.21)), MUG),
    # Scissors leaning in the mug, lowest point on the floor.
    (box((0.02, 0, 0.10), (0.12, 0.03, 0.19)), MUG),
    # Green block resting in the bowl: IoU with the cavity would be ~0.03.
    (box((0, 0, 0.03), (0.04, 0.04, 0.04)), BOWL_CAVITY),
    # Banana lying in the tray.
    (box((0, 0.02, 0.03), (0.15, 0.04, 0.035)), TRAY_CAVITY),
    # Small object in a wineglass: the stem is not part of the cavity.
    (box((0, 0, 0.11), (0.04, 0.04, 0.04)), WINEGLASS_CAVITY),
], ids=["scissors-upright-in-mug", "scissors-leaning-in-mug", "block-in-bowl", "banana-in-tray",
        "object-in-wineglass"])
def test_genuine_placements_are_inside(obj, cavity):
    assert is_inside(obj, cavity)


@pytest.mark.parametrize("obj, cavity", [
    # Scissors lying across the mug's rim: OnTop, but reaches no depth.
    (box((0, 0, 0.09), (0.08, 0.21, 0.02)), MUG),
    # Banana across the bowl's rim, sagging a little into it.
    (box((0, 0, 0.065), (0.22, 0.04, 0.035)), BOWL_CAVITY),
    # Banana with one end dipped into the tray, most of it hanging off the side.
    (box((0.15, 0, 0.04), (0.15, 0.04, 0.05)), TRAY_CAVITY),
    # Resting on the table beside the mug.
    (box((0.12, 0, 0.01), (0.08, 0.21, 0.02)), MUG),
], ids=["scissors-across-mug-rim", "banana-across-bowl-rim", "banana-hanging-off-tray",
        "beside-mug"])
def test_partial_or_rim_placements_are_rejected(obj, cavity):
    assert not is_inside(obj, cavity)
