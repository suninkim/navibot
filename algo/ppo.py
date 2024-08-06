import torchrl
import torch
from torch import multiprocessing

class PPO:
    def __init__(self):
        
        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )

        a=1

    def get_action(self, state):
        action = 1
        return action

    def push_transition(transition):
        a=1

    def train(self):
        a=1