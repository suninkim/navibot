import multiprocessing
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm


class PPO:
    def __init__(self):

        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )

        num_cells = 256
        lr = 3e-4
        self.max_grad_norm = 1.0

        self.frames_per_batch = 1000
        self.total_frames = 10_000

        self.sub_batch_size = 64
        self.num_epochs = 10
        clip_epsilon = 0.2
        gamma = 0.99
        lmbda = 0.95
        entropy_eps = 1e-4

        base_env = GymEnv("InvertedDoublePendulum-v4", device=self.device)
        self.env = TransformedEnv(
            base_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )
        self.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        print("observation_spec:", self.env.observation_spec)
        print("reward_spec:", self.env.reward_spec)
        print("input_spec:", self.env.input_spec)
        print("action_spec (as defined by input_spec):", self.env.action_spec)

        rollout = self.env.rollout(3)
        print("rollout of three steps:", rollout)
        print("Shape of the rollout TensorDict:", rollout.batch_size)

        actor_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(2 * self.env.action_spec.shape[-1], device=self.device),
            NormalParamExtractor(),
        )
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.env.action_spec.space.low,
                "high": self.env.action_spec.space.high,
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
        )

        value_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(1, device=self.device),
        )

        self.value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )
        print("Running policy:", self.policy_module(self.env.reset()))
        print("Running value:", self.value_module(self.env.reset()))
        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        self.advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=self.value_module, average_gae=True
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.total_frames // self.frames_per_batch, 0.0
        )

    def get_action(self, state):
        action = 1
        return action

    def push_transition(transition):
        a = 1

    def train(self):
        self.logs = defaultdict(list)
        pbar = tqdm(total=self.total_frames)
        eval_str = ""

        # We iterate over the self.collectoruntil it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(self.collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(self.num_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(self.frames_per_batch // self.sub_batch_size):
                    subdata = self.replay_buffer.sample(self.sub_batch_size)
                    loss_vals = self.loss_module(subdata.to(self.device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.max_grad_norm
                    )
                    self.optim.step()
                    self.optim.zero_grad()

            self.logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = f"average reward={self.logs['reward'][-1]: 4.4f} (init={self.logs['reward'][0]: 4.4f})"
            self.logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {self.logs['step_count'][-1]}"
            self.logs["lr"].append(self.optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {self.logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = self.env.rollout(1000, self.policy_module)
                    self.logs["eval reward"].append(
                        eval_rollout["next", "reward"].mean().item()
                    )
                    self.logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    self.logs["eval step_count"].append(
                        eval_rollout["step_count"].max().item()
                    )
                    eval_str = (
                        f"eval cumulative reward: {self.logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {self.logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {self.logs['eval step_count'][-1]}"
                    )
                    del eval_rollout
            pbar.set_description(
                ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
            )

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            self.scheduler.step()

    def plot_graph(self):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(self.logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(self.logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(self.logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()


if __name__ == "__main__":
    ppo = PPO()
    ppo.train()
    ppo.plot_graph()
