"""Pin what PLACE_INTO's containment test accepts and rejects, on the host, with no GPU.

TaskProgressionMixin._is_inside marks which of the object's points are in the container's cavity
(its `fillable` volume, else its AABB), then passes when realm.geometry.containment_fractions reports
at least OmniGibson's Overlaid.OVERLAP_AREA_PERCENTAGE (0.5 in 3.9.1, pinned here because importing
omnigibson boots Isaac) of the below-rim points in the cavity, and a depth fraction of at least
PLACE_INSIDE_MIN_DEPTH_FRACTION. Each case is a point layout taken from a shipped `put` task or a
simulated placement; the rejections are false successes the old Inside-or-OnTop check credited or
an earlier version of this one did.

    uv run python -m pytest -q tests/test_place_into_containment.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realm.config.shared import PLACE_INSIDE_CONTACT_MARGIN, PLACE_INSIDE_MIN_DEPTH_FRACTION  # noqa: E402
from realm.geometry import containment_fractions, points_in_box, pull_toward_centre  # noqa: E402

OG_OVERLAP_AREA_PERCENTAGE = 0.5


def box_points(center, extent, n=4):
    axes = [np.linspace(c - e / 2, c + e / 2, n) for c, e in zip(center, extent)]
    return np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 3)


def segment_points(start, end, n=30):
    return np.linspace(start, end, n)


def scissors(tip, pivot, handles_center):
    """Blades as a segment from @tip to @pivot, handle loops as an 8 x 2 x 8 cm box."""
    return np.vstack([segment_points(tip, pivot), box_points(handles_center, (0.08, 0.02, 0.08))])


class BoxCavity:
    """A container without a `fillable` link: its whole AABB is the cavity."""

    def __init__(self, center, extent):
        c, e = np.asarray(center, dtype=float), np.asarray(extent, dtype=float)
        self.lo, self.hi = c - e / 2, c + e / 2

    def contains(self, points):
        return points_in_box(points, self.lo, self.hi)


class FlaredCavity:
    """A bowl's `fillable` volume: a cone frustum, radius @r_floor at @z_floor widening to @r_rim at
    @z_rim -- the convex hull OG's check_points_in_volume tests against."""

    def __init__(self, r_floor, r_rim, z_floor, z_rim):
        self.r_floor, self.r_rim = r_floor, r_rim
        self.lo, self.hi = np.array([-r_rim, -r_rim, z_floor]), np.array([r_rim, r_rim, z_rim])

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        t = (points[:, 2] - self.lo[2]) / (self.hi[2] - self.lo[2])
        radius = self.r_floor + np.clip(t, 0, 1) * (self.r_rim - self.r_floor)
        return (t >= 0) & (t <= 1) & (np.hypot(points[:, 0], points[:, 1]) <= radius)


def is_inside(points, cavity):
    """Mirrors TaskProgressionMixin._is_inside."""
    in_cavity = cavity.contains(pull_toward_centre(points, PLACE_INSIDE_CONTACT_MARGIN))
    rim_fraction, depth_fraction = containment_fractions(points, in_cavity, cavity.lo[2], cavity.hi[2])
    return rim_fraction >= OG_OVERLAP_AREA_PERCENTAGE and depth_fraction >= PLACE_INSIDE_MIN_DEPTH_FRACTION


MUG = BoxCavity((0, 0, 0.04), (0.115, 0.090, 0.080))
TRAY = BoxCavity((0, 0, 0.03), (0.21, 0.33, 0.04))
WINEGLASS = BoxCavity((0, 0, 0.13), (0.08, 0.08, 0.09))           # glass bowl above a 0.085 stem
BOWL = FlaredCavity(r_floor=0.05, r_rim=0.095, z_floor=0.01, z_rim=0.065)


@pytest.mark.parametrize("points, cavity", [
    # Scissors standing in the mug: most of their height is above the rim.
    (scissors((0, 0, 0.005), (0, 0, 0.13), (0, 0, 0.17)), MUG),
    # Scissors leaning in the mug, handles overhanging the rim. In simulation an AABB-footprint
    # test scored this 0.23-0.37 and a whole-object point share 0.08: both rejected scissors
    # visibly resting in the mug.
    (scissors((-0.015, 0, 0.005), (0.02, 0, 0.125), (0.045, 0, 0.165)), MUG),
    # Green block resting on the bowl's floor.
    (box_points((0, 0, 0.025), (0.03, 0.03, 0.03)), BOWL),
    # The same block as a primitive cube: 8 vertices, the bottom 4 a millimetre into the floor. In
    # simulation the volume test put exactly half of them outside until contact was allowed for.
    (box_points((0, 0, 0.024), (0.03, 0.03, 0.03), n=2), BOWL),
    # Banana lying in the tray.
    (box_points((0, 0.02, 0.03), (0.15, 0.04, 0.035)), TRAY),
    # Small object in a wineglass: the stem is not part of the cavity.
    (box_points((0, 0, 0.11), (0.04, 0.04, 0.04)), WINEGLASS),
], ids=["scissors-upright-in-mug", "scissors-leaning-in-mug", "block-in-bowl",
        "block-corners-on-bowl-floor", "banana-in-tray", "object-in-wineglass"])
def test_genuine_placements_are_inside(points, cavity):
    assert is_inside(points, cavity)


@pytest.mark.parametrize("points, cavity", [
    # Scissors lying across the mug's rim: OnTop, but reaching no depth.
    (box_points((0, 0, 0.09), (0.08, 0.21, 0.02)), MUG),
    # Scissors leaning against the OUTSIDE of the mug, tip on the table.
    (scissors((0.075, 0, 0.0), (0.06, 0, 0.12), (0.05, 0, 0.16)), MUG),
    # Green block on the table, half tucked under the bowl's flared rim: over the bowl's footprint
    # but outside its volume. Credited in simulation by a footprint-box version of this check.
    (box_points((0.08, 0, 0.015), (0.03, 0.03, 0.03)), BOWL),
    # Banana across the bowl's rim, sagging a little into it.
    (box_points((0, 0, 0.0675), (0.22, 0.04, 0.035)), BOWL),
    # Banana with one end dipped into the tray, most of it hanging off the side.
    (box_points((0.15, 0, 0.04), (0.15, 0.04, 0.05)), TRAY),
    # Resting on the table beside the mug.
    (box_points((0.12, 0, 0.01), (0.08, 0.21, 0.02)), MUG),
], ids=["scissors-across-mug-rim", "scissors-leaning-outside-mug", "block-under-bowl-rim",
        "banana-across-bowl-rim", "banana-hanging-off-tray", "beside-mug"])
def test_partial_or_rim_placements_are_rejected(points, cavity):
    assert not is_inside(points, cavity)


def test_nothing_in_the_cavity_has_no_depth():
    points = box_points((0.5, 0, 0.0), (0.01, 0.01, 0.01))
    rim_fraction, depth_fraction = containment_fractions(points, MUG.contains(points), MUG.lo[2], MUG.hi[2])
    assert rim_fraction == 0.0 and depth_fraction == -np.inf


def test_parts_above_the_rim_are_ignored():
    # Same blades in the mug; handles raised far above it change nothing.
    def fractions(points):
        return containment_fractions(points, MUG.contains(points), MUG.lo[2], MUG.hi[2])

    low = scissors((0, 0, 0.005), (0, 0, 0.13), (0, 0, 0.17))
    high = scissors((0, 0, 0.005), (0, 0, 0.13), (0.3, 0, 0.6))
    assert fractions(low) == fractions(high)
