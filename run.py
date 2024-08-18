import argparse

import torch
import yaml
from tasks import CassieEnv, make_base_cassie_env
from algo import PPO
from torchrl.collectors import SyncDataCollector


parser = argparse.ArgumentParser()
parser.add_argument("--task_cfg", type=str, default="cassie")
parser.add_argument("--train_cfg", type=str, default="ppo")
args = parser.parse_args()

# Read task config
task_cfg_path = f"config/task/{args.task_cfg}.yaml"
with open(task_cfg_path, "r") as stream:
    task_cfg = yaml.safe_load(stream)

# Read training config
train_cfg_path = f"config/train/{args.train_cfg}.yaml"
with open(train_cfg_path, "r") as stream:
    train_cfg = yaml.safe_load(stream)

# Env
cassie_env = CassieEnv(task_cfg)

# Agent
agent = PPO(train_cfg, cassie_env)

tensor_dict = cassie_env.get_init_state()
while cassie_env.simulation_app.is_running():
    action = agent.get_action(tensor_dict)

    tensor_dict["action"] = cassie_env.rand_action()    # to check
    new_tensor_dict = cassie_env.step(tensor_dict)
    agent.push_transition(tensor_dict)
    tensor_dict = new_tensor_dict

    # agent.train()
# close sim app
cassie_env.simulation_app.close()
