import time

import numpy as np
import yaml
from controller import Controller

import mujoco
import mujoco.viewer

paused = False


def key_callback(keycode):
    if chr(keycode) == " ":
        global paused
        paused = not paused


if __name__ == "__main__":

    robot = mujoco.MjModel.from_xml_path("../assets/robots/mjcf/navibot/navi.xml")
    data = mujoco.MjData(robot)

    control_cfg_path = "controller_config/navibot_controller.yaml"
    with open(control_cfg_path, "r") as stream:
        control_cfg = yaml.safe_load(stream)
    controller = Controller(control_cfg["controller"])

    init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    final_qpos = [0, 0, 0.5, 1.0, -0.7, 0.0, 0.0, -0.5, -1.0, 0.7]

    result = controller.generate_traj(
        "joint", init_qpos, init_qpos, final_qpos, init_qpos, min_duration=10
    )
    cnt = 0
    forward = True
    with mujoco.viewer.launch_passive(robot, data, key_callback=key_callback) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            if not paused:
                data.qpos[7:] = result[1][cnt]
                if cnt < 999:
                    cnt += 1
                else:
                    if not forward:
                        result = controller.generate_traj(
                            "joint",
                            init_qpos,
                            init_qpos,
                            final_qpos,
                            init_qpos,
                            min_duration=10,
                        )
                    else:
                        result = controller.generate_traj(
                            "joint",
                            final_qpos,
                            init_qpos,
                            init_qpos,
                            init_qpos,
                            min_duration=10,
                        )
                    forward = not forward
                    cnt = 0

                mujoco.mj_step(robot, data)
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
