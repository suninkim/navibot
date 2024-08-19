import numpy as np
import ruckig


class Controller:
    def __init__(self):
        a = 1

        self.num_joint = 10

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
        self.rkg["T"] = ruckig.InputParameter(w_dim + v_dim)
        self.rkg["T"].max_velocity = np.array([w_limit] * w_dim + [v_limit] * v_dim)
        self.rkg["T"].max_acceleration = np.array(
            [wa_limit] * w_dim + [va_limit] * v_dim
        )
        self.rkg["T"].max_jerk = np.array([jerk_limit] * (w_dim + v_dim))
        self.rkg_ruckig["T"] = ruckig.Ruckig(w_dim + v_dim)
        self.rkg_trajectory["T"] = ruckig.Trajectory(w_dim + v_dim)

    def generate_traj(
        self, space: str, init_pos, init_vel, final_pos, final_vel, min_duration
    ):
        if space == "T":
            n_dim = 6
        elif space == "joint":
            n_dim = len(self.robot.motors)
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
