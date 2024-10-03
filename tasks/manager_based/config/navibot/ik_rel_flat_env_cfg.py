# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .rough_env_cfg import NavibotRoughEnvCfg

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


@configclass
class NavibotActionsCfg:
    """Action specifications for the MDP."""

    left_leg_joint_pos = mdp.DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    right_leg_joint_pos = mdp.DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)



@configclass
class NavibotIkEnvCfg(NavibotRoughEnvCfg):

    actions: NavibotActionsCfg= NavibotActionsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # rewards
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.feet_air_time.weight = 5.0
        # self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = ["hip_rotation_.*"]
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.actions.left_leg_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_hip_left","joint_thigh_left", "joint_knee_left", "joint_shin_left", "joint_toe_left"],
            body_name="toe_left",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.08]),
        )


        self.actions.right_leg_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_hip_right","joint_thigh_right", "joint_knee_right", "joint_shin_right", "joint_toe_right"],
            body_name="toe_right",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.08]),
        )


