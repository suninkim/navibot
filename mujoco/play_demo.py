import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(sys.path)
import time

import numpy as np
import yaml
from controller import Controller

import mujoco
import mujoco.viewer
from utils import get_demo_data

paused = False


def key_callback(keycode):
    if chr(keycode) == " ":
        global paused
        paused = not paused


if __name__ == "__main__":

    robot = mujoco.MjModel.from_xml_path("assets/robots/mjcf/franka/panda.xml")
    data = mujoco.MjData(robot)

    control_cfg_path = "mujoco/controller_config/franka_controller.yaml"
    with open(control_cfg_path, "r") as stream:
        control_cfg = yaml.safe_load(stream)
    controller = Controller(control_cfg["controller"])

    demo_data = get_demo_data("data/test.hdf5")["demo_0"]

    episode = 0
    cnt = 0
    forward = True

    return_init_step = 1000
    num_samples = len(demo_data["actions"])
    with mujoco.viewer.launch_passive(robot, data, key_callback=key_callback) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            if not paused:

                if cnt < num_samples:
                    action = demo_data["actions"][cnt]
                    data.qpos[:7] = data.qpos[:7] + action
                    cnt += 1
                    mujoco.mj_step(robot, data)
                else:
                    init_qpos = data.qpos[:7]
                    final_qpos = [0, 0, 0, 0, 0, 0, 0]

                    result = controller.generate_traj(
                        "joint",
                        init_qpos,
                        final_qpos,
                        final_qpos,
                        final_qpos,
                        min_duration=10,
                    )
                    for i in range(return_init_step):
                        data.qpos[:7] = result[1][i]
                        mujoco.mj_step(robot, data)
                        viewer.sync()
                    cnt = 0

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    data.time % 2
                )

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = robot.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
