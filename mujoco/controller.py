import time

import numpy as np
import ruckig

import mujoco
import mujoco.viewer


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_joint = 7
        self.control_freq = 100
        self._configure_ruckig()

    def _configure_ruckig(self):
        self.rkg = {}
        self.rkg_ruckig = {}
        self.rkg_trajectory = {}
        # motor
        n_joint = self.num_joint  # len(self.robot.motors)
        self.rkg["joint"] = ruckig.InputParameter(n_joint)
        vel_limit = self.cfg["joint_vel_limit"]
        acc_limit = self.cfg["joint_acc_limit"]
        jerk_limit = self.cfg["joint_jerk_limit"]
        # logging.info(
        #     f"vel_limit={vel_limit}, acc_limit={acc_limit}, jerk_limit={jerk_limit}"
        # )
        self.vel_limit = np.array(vel_limit)
        self.rkg["joint"].max_velocity = self.vel_limit
        self.rkg["joint"].max_acceleration = np.array(acc_limit)
        self.rkg["joint"].max_jerk = np.array(jerk_limit)
        self.rkg_ruckig["joint"] = ruckig.Ruckig(n_joint)
        self.rkg_trajectory["joint"] = ruckig.Trajectory(n_joint)
        # hand
        w_dim = 3
        v_dim = 3
        w_limit = self.cfg["w_limit"]
        v_limit = self.cfg["v_limit"]
        wa_limit = self.cfg["wa_limit"]
        va_limit = self.cfg["va_limit"]
        jerk_limit = self.cfg["jerk_limit"]
        # self.rkg["T"] = ruckig.InputParameter(w_dim + v_dim)
        # self.rkg["T"].max_velocity = np.array([w_limit] * w_dim + [v_limit] * v_dim)
        # self.rkg["T"].max_acceleration = np.array(
        #     [wa_limit] * w_dim + [va_limit] * v_dim
        # )
        # self.rkg["T"].max_jerk = np.array([jerk_limit] * (w_dim + v_dim))
        # self.rkg_ruckig["T"] = ruckig.Ruckig(w_dim + v_dim)
        # self.rkg_trajectory["T"] = ruckig.Trajectory(w_dim + v_dim)

    def generate_traj(
        self, space: str, init_pos, init_vel, final_pos, final_vel, min_duration
    ):
        if space == "T":
            n_dim = 6
        elif space == "joint":
            # n_dim = len(self.robot.motors)
            n_dim = self.num_joint
        else:
            raise Exception(f"Invalid space={space}, self.rkg.keys()={self.rkg.keys()}")
        # log_assert(
        #     (
        #         (n_dim,)
        #         == init_pos.shape
        #         == init_vel.shape
        #         == final_pos.shape
        #         == final_vel.shape
        #     ),
        #     f"{init_pos.shape}, {init_vel.shape}, {final_pos.shape}, {final_vel.shape}",
        # )
        # log_assert(0 < min_duration, f"min_duration={min_duration}")
        self.rkg[space].current_position = init_pos
        self.rkg[space].current_velocity = init_vel
        self.rkg[space].current_acceleration = np.zeros(n_dim)
        self.rkg[space].target_position = final_pos
        self.rkg[space].target_velocity = final_vel
        self.rkg[space].target_acceleration = np.zeros(n_dim)
        self.rkg[space].minimum_duration = min_duration
        result = self.rkg_ruckig[space].calculate(
            self.rkg[space], self.rkg_trajectory[space]
        )
        if result == ruckig.Result.ErrorInvalidInput:
            # logging.error("=======================================")
            # # space: str, init_pos, init_vel, final_pos, final_vel, min_duration
            # logging.error(f"space=\n\t{space}")
            # logging.error(f"init_pos=\n\t{init_pos}")
            # logging.error(f"init_vel=\n\t{init_vel}")
            # logging.error(f"final_pos=\n\t{final_pos}")
            # logging.error(f"final_vel=\n\t{final_vel}")
            # logging.error(f"min_duration=\n\t{min_duration}")
            # logging.error("=======================================")
            raise Exception("[Ruckig] Invalid input!")
        pos_traj = []
        vel_traj = []
        ts = np.arange(0, self.rkg_trajectory[space].duration, 1 / self.control_freq)
        for t in ts:
            p, v, _ = self.rkg_trajectory[space].at_time(t)
            pos_traj.append(p)
            vel_traj.append(v)
        return (
            self.rkg_trajectory[space].duration,
            np.vstack(pos_traj),
            np.vstack(vel_traj),
        )


paused = False


def key_callback(keycode):
    if chr(keycode) == " ":
        global paused
        paused = not paused


if __name__ == "__main__":

    m = mujoco.MjModel.from_xml_path("../assets/robots/mjcf_navi/navi.xml")
    d = mujoco.MjData(m)

    control_cfg = {
        "w_limit": [1.0] * 10,
        "v_limit": [1.0] * 10,
        "wa_limit": [1.0] * 10,
        "va_limit": [1.0] * 10,
        "jerk_limit": [1.0] * 10,
        "joint_vel_limit": [1.0] * 10,
        "joint_acc_limit": [1.0] * 10,
        "joint_jerk_limit": [1.0] * 10,
    }

    controller = Controller(control_cfg)

    init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    final_qpos = [0, 0, 1, 0.78, 0.22, 0, 0, 1, 0.78, 0.22]

    result = controller.generate_traj(
        "joint", init_qpos, init_qpos, final_qpos, init_qpos, min_duration=10
    )
    print(len(result[1][0]))
    cnt = 0
    forward = False
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            print(f"d.qpos[:]: {d.qpos[:]}")
            print(f"result[cnt][:]: {result[1][cnt]} ")
            d.qpos[7:] = result[1][cnt]
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
            if not paused:
                mujoco.mj_step(m, d)
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

            # print(m.names)
            # print(len(d.qpos))
            # print(d.joint("joint_left_leg1"))
            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
