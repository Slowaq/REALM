"""Coordinate frames: REALM's authored `relative_bbox_position` <-> the solver's robot frame.

REALM (env_config._apply_object_cfg) turns an authored relative position r into scene coordinates
as  W = (x_min + mx(r.x), y_min + r.y),  where  mx(x) = width - x  when the robot yaw is in
[90, 270] degrees and  mx(x) = x  otherwise.  The solver works in a right-handed robot frame
(a = metres forward from the robot base, b = metres to the robot's left), so RoboLab's
"+X is front, +Y is left" vocabulary lines up with what the robot camera sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tooling.task_authoring.authoring import load_scene_regions
from tooling.task_authoring.generate_realm_droid100 import (
    ELLIPTICAL_SUPPORTS,
    SUPPORT_EDGE_CLEARANCE,
    UNSAFE_SCENE_REGIONS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENES_YAML = REPO_ROOT / "realm" / "config" / "scenes" / "scenes.yaml"


@dataclass(frozen=True)
class RegionFrame:
    region: dict

    def __post_init__(self):
        yaw = float(self.region["robot_rot"][2])
        if abs(yaw / 90.0 - round(yaw / 90.0)) > 1e-6:
            raise ValueError(
                f"{self.region['id']}: robot yaw {yaw} is not a multiple of 90 deg; robot-relative "
                "left/right/front/behind would be diagonal on the support, which this PoC does not handle")

    @property
    def id(self) -> str:
        return self.region["id"]

    @property
    def yaw(self) -> int:
        return int(round(float(self.region["robot_rot"][2]))) % 360

    @property
    def mirrored(self) -> bool:
        # Same test as env_config, which asserts yaw >= 0 and compares the raw config value.
        return 90 <= float(self.region["robot_rot"][2]) <= 270

    @property
    def elliptical(self) -> bool:
        return self.region["support"] in ELLIPTICAL_SUPPORTS

    @property
    def swaps_axes(self) -> bool:
        """True when the robot's forward axis is the scene's Y axis."""
        return self.yaw in (90, 270)

    def _forward_left(self):
        theta = math.radians(self.yaw)
        forward = (round(math.cos(theta)), round(math.sin(theta)))
        left = (-forward[1], forward[0])
        return forward, left

    def world(self, x_rel: float, y_rel: float) -> tuple[float, float]:
        width = self.region["width"]
        x = width - x_rel if self.mirrored else x_rel
        return self.region["x_min"] + x, self.region["y_min"] + y_rel

    def authored(self, world_x: float, world_y: float) -> tuple[float, float]:
        x = world_x - self.region["x_min"]
        if self.mirrored:
            x = self.region["width"] - x
        return x, world_y - self.region["y_min"]

    def to_solver(self, x_rel: float, y_rel: float) -> tuple[float, float]:
        forward, left = self._forward_left()
        wx, wy = self.world(x_rel, y_rel)
        dx, dy = wx - self.region["robot_pos"][0], wy - self.region["robot_pos"][1]
        return forward[0] * dx + forward[1] * dy, left[0] * dx + left[1] * dy

    def from_solver(self, a: float, b: float) -> tuple[float, float]:
        forward, left = self._forward_left()
        wx = self.region["robot_pos"][0] + a * forward[0] + b * left[0]
        wy = self.region["robot_pos"][1] + a * forward[1] + b * left[1]
        return self.authored(wx, wy)

    def solver_bounds(self, inset: float = SUPPORT_EDGE_CLEARANCE) -> tuple[float, float, float, float]:
        corners = [
            self.to_solver(x, y)
            for x in (inset, self.region["width"] - inset)
            for y in (inset, self.region["depth"] - inset)
        ]
        a_values, b_values = [c[0] for c in corners], [c[1] for c in corners]
        return min(a_values), max(a_values), min(b_values), max(b_values)

    def solver_footprint(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        """World-axis footprint (along scene X, scene Y) -> (along forward, along left)."""
        return (world_xy[1], world_xy[0]) if self.swaps_axes else (world_xy[0], world_xy[1])


def usable_regions(scenes_yaml: Path = SCENES_YAML) -> list[RegionFrame]:
    """The DROID100 generator's tabletop pool, minus regions whose robot yaw is not axis-aligned."""

    frames = []
    for region in load_scene_regions(scenes_yaml):
        if region["width"] < 0.4 or region["depth"] < 0.4 or region["z"] <= 0:
            continue
        if (region["scene"], region["support"]) in UNSAFE_SCENE_REGIONS:
            continue
        try:
            frames.append(RegionFrame(region))
        except ValueError:
            continue
    return frames


def find_region(region_id: str | None, scenes_yaml: Path = SCENES_YAML) -> RegionFrame:

    frames = usable_regions(scenes_yaml)
    if region_id is None:
        return frames[0]
    wanted = region_id.replace(" ", "")
    for frame in frames:
        if frame.id.replace(" ", "") == wanted:
            return frame
    raise ValueError(f"unknown or unusable region {region_id!r}; usable: {[f.id for f in frames]}")
