from env import CassiEnv
from algo import PPO
import argparse
import torch

import yaml


# Read training config

# Read env config


# Agent
agent = PPO()

# Env
cassie_env = CassiEnv()

state = cassie_env.get_init_state()
while cassie_env.simulation_app.is_running():
    action = agent.get_action(state)
    next_state, reward, terminal = cassie_env.step(action)
    transition = [state, action, reward, next_state, terminal]
    state = next_state

    agent.push_transition(transition)
    agent.train()

# close sim app
cassie_env.simulation_app.close()