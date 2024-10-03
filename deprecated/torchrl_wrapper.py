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

from omni.isaac.lab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on creating a quadruped base environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase, Transform, TransformedEnv
# from torchrl.data import Bounded, Composite
from torchrl.envs.utils import check_env_specs, step_mdp

from omni.isaac.lab.envs import ManagerBasedRLEnv

def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: (
                make_composite_from_td(tensor)
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

class TorchRLVecEnvWrapper(EnvBase):
    batch_locked = False
    
    def __init__(self, env: ManagerBasedRLEnv,  rl_device: str, td_params=None):
        if td_params is None:
            td_params = self.gen_params()
            
        self._batch_size = torch.Size([10])
        super().__init__(device=rl_device, batch_size=[10])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        self.env = env
        self._rl_device = rl_device
        self._sim_device = env.unwrapped.device
        

    def _reset(self, tensordict=None):
        a = 1

    def _step(self, tensordict):
        action = (
            tensordict["joint_pos"]
            + tensordict["params"]["action_scale"] * tensordict["action"]
        )

        obs, rew, terminated, truncated, info = self.env.step(action)

        out = self.process_obs(obs, rew, terminated, truncated, tensordict)

        return out

    def process_obs(self, obs, rew, terminated, truncated, tensordict):
        next_dict = TensorDict(
            {
                "joint_pos": obs["joint_pos"],
                "joint_vel": obs["joint_vel"],
                "params": tensordict["params"],
                "reward": rew,
                "done": terminated | truncated,
            },
            tensordict.shape,
        )
        tensordict["next"] = next_dict
        return tensordict

    def rand_step(self, tensordict):
        tensordict["action"] = self.rand_action()
        return self._step(tensordict)

    def rand_action(self):
        return self.action_spec.rand()

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
        

    def _make_spec(self, td_params):
        self.__dict__["_input_spec"] = None
        self.__dict__["_output_spec"] = None
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                low=-torch.pi,
                high=torch.pi,
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
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
        
    def make_composite_from_td(td):
        # custom function to convert a ``tensordict`` in a similar spec structure
        # of unbounded values.
        composite = CompositeSpec(
            {
                key: (
                    make_composite_from_td(tensor)
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
    
    def _set_seed(self, seed: int):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    @staticmethod
    def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
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

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs

    @property
    def device(self) -> str:
        """Returns the base environment simulation device."""
        return self.unwrapped.device
        


def make_base_cassie_env(tack_cfg, env_name="CassieEnv", frame_skip=1):
    env = CassieEnv(tack_cfg)

    return env




def main():
    from manager_based_rl import CassieEnvCfg
    # setup base environment
    env_cfg = CassieEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # setup RL environment
    wrapped_env = TorchRLVecEnvWrapper(env, env.device)

    # simulate physics
    # tensor_dict = wrapped_env.reset()
    obs, _ = wrapped_env.env.reset()
    while simulation_app.is_running():
        joint_positions = torch.randn_like(wrapped_env.env.action_manager.action)
            # step the environment
        obs, rew, terminated, truncated, info = wrapped_env.env.step(joint_positions)
    #     with torch.inference_mode():
    #         # sample random actions
    #         joint_positions = torch.randn_like(wrapped_env.action_manager.action)
    #         # step the environment
    #         new_tensor_dict = wrapped_env.step(joint_positions)
    #         tensor_dict = new_tensor_dict
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter

    # close the environment
    wrapped_env.env.close()
    
if __name__ == "__main__":
    
    main()
    # # run the main function
    # task_cfg_path = f"config/task/cassie.yaml"
    # import yaml

    # with open(task_cfg_path, "r") as stream:
    #     task_cfg = yaml.safe_load(stream)

    # cassie_env = CassiEnv(task_cfg)
    
    #     """Main function."""
    
   