"""Open a REALM task scene in the GUI without a policy. Usage: python /app/view_task.py [task_cfg_path]"""
import sys

import omnigibson as og
from omnigibson.macros import gm
from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.sim_config import set_sim_config

task_cfg_path = sys.argv[1] if len(sys.argv) > 1 else "IMPACT/scissors_into_mug/default.yaml"

set_sim_config(robot="DROID_mounted")

# In GUI mode OmniGibson opens one extra viewport per camera sensor while loading and renders
# immediately to dock it, which intermittently deadlocks the renderer. Launch the GUI normally,
# then treat loading as headless so no sensor viewports are created.
_launch = og.launch


def _launch_then_skip_sensor_viewports(*args, **kwargs):
    sim = _launch(*args, **kwargs)
    gm.HEADLESS = True
    return sim


og.launch = _launch_then_skip_sensor_viewports

env = RealmEnvironmentDynamic(
    task_cfg_path=task_cfg_path,
    perturbations=["Default"],
    robot="DROID_mounted",
)
env.reset()
gm.HEADLESS = False

for obj in env.omnigibson_env.scene.objects:
    print("OBJ", obj.name, obj.get_position_orientation()[0].tolist(), obj.aabb_extent.tolist(), flush=True)

while True:
    og.sim.render()
