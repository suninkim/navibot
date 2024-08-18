# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/bipeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from typing import Optional

import torch
from omni.isaac.lab.app import AppLauncher
from tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import  EnvBase, Transform, TransformedEnv
from torchrl.envs.utils import check_env_specs, step_mdp


def make_base_cassie_env(tack_cfg, env_name="CassieEnv", frame_skip=1):
    env = CassieEnv(tack_cfg)

    return env


class CassieEnv(EnvBase):
    def __init__(self, task_cfg, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()
        super().__init__(device=device, batch_size=[])

        self.task_cfg = task_cfg

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.sim_setting()
        self._make_spec(td_params)

        """Rest everything follows."""

        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import Articulation
        from omni.isaac.lab.sim import SimulationContext

        ##
        # Pre-defined configs
        ##
        from omni.isaac.lab_assets.cassie import CASSIE_CFG  # isort:skip
        from omni.isaac.lab_assets import H1_CFG  # isort:skip
        from omni.isaac.lab_assets import G1_CFG  # isort:skip

        # Load kit helper
        self.sim = SimulationContext(
            sim_utils.SimulationCfg(device=self.task_cfg["sim_params"]["sim_device"], dt=self.task_cfg["sim_params"]["sim_dt"])
        )
        # Set main camera
        self.sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

        # Spawn things into stage
        # Ground-plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        # Lights
        cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

        self.origins = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        # Robots
        self.cassie = Articulation(CASSIE_CFG.replace(prim_path="/World/Cassie"))
        self.robots = [self.cassie]

        # Play the simulator
        self.sim.reset()

        # Now we are ready!
        print("[INFO]: Setup complete...")

        # Define simulation stepping
        self.sim_dt = self.sim.get_physics_dt()
        self.sim_time = 0.0
        self.count = 0
        self.max_step_size = self.task_cfg["setting"]["max_step_size"]

    def sim_setting(self):
        # add argparse arguments
        parser = argparse.ArgumentParser(
            description="This script demonstrates how to simulate bipedal robots."
        )
        # # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        # # parse the arguments
        args_cli = parser.parse_args()

        # # launch omniverse app
        app_launcher = AppLauncher(args_cli)
        self.simulation_app = app_launcher.app

    def _reset(self, tensordict=None):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        # reset counters
        self.sim_time = 0.0
        self.count = 0
        for index, robot in enumerate(self.robots):
            # reset dof state
            joint_pos, joint_vel = (
                robot.data.default_joint_pos,
                robot.data.default_joint_vel,
            )
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += self.origins[index]
            robot.write_root_state_to_sim(root_state)
            robot.reset()
            out = TensorDict(
                {
                    "joint_pos": joint_pos,
                    "joint_vel": joint_vel,
                    "root_state": root_state,
                    "params": tensordict["params"],
                },
                tensordict.shape,
            )

        print(">>>>>>>> Reset!")

        return out

    def _step(self, tensordict):
        target_pos = tensordict["joint_pos"] + tensordict["params"]["action_scale"]* tensordict["action"]

        # apply action to the robot
        for robot in self.robots:
            robot.set_joint_position_target(target_pos)
            robot.write_data_to_sim()

        # perform step
        self.sim.step()
        # update sim-time
        self.sim_time += self.sim_dt
        self.count += 1

        # update buffers
        for robot in self.robots:
            robot.update(self.sim_dt)
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            root_state = robot.data.default_root_state.clone()

            reward = self.cal_reward()
            done = self.is_done()
            # done = torch.zeros_like(reward, dtype=torch.bool)
            out = TensorDict(
                {
                    "joint_pos": joint_pos,
                    "joint_vel": joint_vel,
                    "root_state": root_state,
                    "params": tensordict["params"],
                    "reward": reward,
                    "done": done,
                },
                tensordict.shape,
            )

        if done:
            init_td = self._reset()
            for key in init_td.keys():
                out[key] = init_td[key]

        return out
    
    def rand_step(self, tensordict):
        tensordict["action"]=self.rand_action()
        return self._step(tensordict)
    
    def rand_action(self):
        return self.action_spec.rand()

    def cal_reward(self):
        return 1

    def get_init_state(self):
        return self.reset()

    def is_done(self):
        if self.count % self.max_step_size == 0:
            return True
        return False

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def make_composite_from_td(self, td):
        # custom function to convert a ``tensordict`` in a similar spec structure
        # of unbounded values.
        composite = CompositeSpec(
            {
                key: (
                    self.make_composite_from_td(tensor)
                    if isinstance(tensor, TensorDictBase)
                    else UnboundedContinuousTensorSpec(
                        dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
                    )
                )
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite


    def _make_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = CompositeSpec(
            joint_pos=BoundedTensorSpec(
                low=-torch.pi,
                high=torch.pi,
                shape=(),
                dtype=torch.float32,
            ),
            joint_vel=BoundedTensorSpec(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=self.make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = BoundedTensorSpec(
            low=-td_params["params", "max_torque"],
            high=td_params["params", "max_torque"],
            shape=(12,),
            dtype=torch.float32,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    def gen_params(self, g=10.0, batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_speed": 8,
                        "max_torque": 1.0,
                        "dt": 0.05,
                        "g": g,
                        "m": 1.0,
                        "l": 1.0,
                        "action_scale": 0.5,
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td


if __name__ == "__main__":
    # run the main function
    task_cfg_path = f"config/task/cassie.yaml"
    import yaml
    with open(task_cfg_path, "r") as stream:
        task_cfg = yaml.safe_load(stream)

    cassie_env = CassiEnv(task_cfg)

    tensor_dict = cassie_env.get_init_state()
    while cassie_env.simulation_app.is_running():
        new_tensor_dict = cassie_env.rand_step(tensor_dict)
        tensor_dict = new_tensor_dict
    # close sim app
    cassie_env.simulation_app.close()
