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
from utils import DataCollector

paused = False
record_start = False
record_stop = False


def key_callback(keycode):
    if chr(keycode) == " ":
        global paused
        paused = not paused

    if chr(keycode) == "C":
        global record_start

        if record_start:
            print("Recording stop!")
            global record_stop
            record_stop = True
        else:
            print("Recording start!")

        record_start = not record_start


if __name__ == "__main__":

    robot = mujoco.MjModel.from_xml_path("assets/robots/mjcf/franka/panda.xml")
    data = mujoco.MjData(robot)

    control_cfg_path = "mujoco/controller_config/franka_controller.yaml"
    with open(control_cfg_path, "r") as stream:
        control_cfg = yaml.safe_load(stream)
    controller = Controller(control_cfg["controller"])

    data_collector = DataCollector("test", "demo", "test.hdf5")

    init_qpos = [0, 0, 0, 0, 0, 0, 0]
    final_qpos = [0, 0, 0.5, 1.0, -0.7, 1.0, 1.5]

    result = controller.generate_traj(
        "joint", init_qpos, init_qpos, final_qpos, init_qpos, min_duration=10
    )
    episode = 0
    cnt = 0
    forward = True
    data_collector.reset()
    with mujoco.viewer.launch_passive(robot, data, key_callback=key_callback) as viewer:
        start = time.time()
        while not data_collector.is_stopped():
            step_start = time.time()

            if not paused:
                prev_pos = data.qpos[:7]
                next_pos = result[1][cnt]
                delta_pos = next_pos - prev_pos
                data.qpos[:7] = data.qpos[:7] + delta_pos

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
                if record_start:
                    data_collector.add("obs/joint_pos", np.array([prev_pos]))
                    data_collector.add("actions", np.array([delta_pos]))
                    data_collector.add("next_obs/joint_pos", np.array([data.qpos[:7]]))
                    data_collector.add("rewards", np.array([0]))
                    data_collector.add("dones", np.array([False]))

                if record_stop:
                    data_collector.flush()
                    record_stop = False

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    data.time % 2
                )

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = robot.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    viewer.close()
