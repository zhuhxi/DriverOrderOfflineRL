import sys
sys.path.append('/tmp-data/zhx/DriverOrderOfflineRL/tianshou')
sys.path.append('/tmp-data/zhx/DriverOrderOfflineRL/gym')

import argparse
import datetime
import os
import pickle
import pprint

import numpy as np
import torch
from torch import nn
import h5py
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.policy import DiscreteCQLPolicy
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger, WandbLogger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="OrderFilter")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-quantiles", type=int, default=32)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=2)
    parser.add_argument("--min-q-weight", type=float, default=10.)
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--update-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512, 512])
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--frames-stack", type=int, default=1)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_driver.QRDQN")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--load-buffer-name", type=str, default="/tmp-data/yanhaoyue/workspace/RL/data/support_feature_buffer.h5"
    )
    parser.add_argument(
        "--buffer-from-rl-unplugged", action="store_true", default=True
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args

args = get_args()
args

def load_buffer(buffer_path: str) -> ReplayBuffer:
    with h5py.File(buffer_path, "r") as dataset:
        buffer = ReplayBuffer.from_data(
            obs=dataset["observations"],
            act=dataset["actions"],
            rew=dataset["rewards"],
            done=dataset["terminals"],
            obs_next=dataset["next_observations"]
        )
    return buffer

class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(state_shape, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_shape), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, state_shape)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, output_dim),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state
class QRDQN(DQN):
    """Reference: Distributional Reinforcement Learning with Quantile \
    Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        num_quantiles: int = 32,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        self.action_num = np.prod(action_shape)
        super().__init__(state_shape, self.action_num * num_quantiles, device)
        self.num_quantiles = num_quantiles

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.action_num, self.num_quantiles)
        return obs, state
    
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# buffer
if args.buffer_from_rl_unplugged:
    buffer = load_buffer(args.load_buffer_name)
else:
    assert os.path.exists(args.load_buffer_name), \
        "Please run atari_dqn.py first to get expert's data buffer."
    if args.load_buffer_name.endswith(".pkl"):
        buffer = pickle.load(open(args.load_buffer_name, "rb"))
    elif args.load_buffer_name.endswith(".hdf5"):
        buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
    else:
        print(f"Unknown buffer format: {args.load_buffer_name}")
        exit(0)
print("Replay buffer size:", len(buffer), flush=True)

args.state_shape = buffer.obs.shape[1]
args.action_shape = 6 # 0， 1， 2， 3， 4， 5
print("Observations shape:", args.state_shape)
print("Actions shape:", args.action_shape)
print("device:", args.device)

# model
net = QRDQN(args.state_shape, args.action_shape, args.num_quantiles, args.device)
optim = torch.optim.Adam(net.parameters(), lr=args.lr)

# define policy
policy = DiscreteCQLPolicy(
    net,
    optim,
    args.gamma,
    args.num_quantiles,
    args.n_step,
    args.target_update_freq,
    min_q_weight=args.min_q_weight,
).to(args.device)

# load a previous policy
if args.resume_path:
    policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    print("Loaded agent from: ", args.resume_path)
    
# log
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
args.algo_name = "cql"
log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
log_path = os.path.join(args.logdir, log_name)

# logger
if args.logger == "wandb":
    logger = WandbLogger(
        save_interval=1,
        name=log_name.replace(os.path.sep, "__"),
        run_id=args.resume_id,
        config=args,
        project=args.wandb_project,
    )
writer = SummaryWriter(log_path)
writer.add_text("args", str(args))
if args.logger == "tensorboard":
    logger = TensorboardLogger(writer)
else:  # wandb
    logger.load(writer)
    

def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

def stop_fn(mean_rewards):
    return False

# watch agent's performance
def watch():
    print("Setup test envs ...")
    policy.eval()
    policy.set_eps(args.eps_test)
    test_envs.seed(args.seed)
    print("Testing agent ...")
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    pprint.pprint(result)
    rew = result["rews"].mean()
    print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
    
def save_checkpoint_fn(epoch, env_step, gradient_step):
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    ckpt_path = os.path.join(log_path, str(epoch) + ".pth")
    torch.save({"model": policy.state_dict()}, ckpt_path)
    return ckpt_path

if args.watch:
    watch()
    exit(0)
import wandb
for epoch in range(1, args.epoch + 1):
    wandb.log(policy.update(0, buffer, batch_size=args.batch_size, repeat=1))
    save_checkpoint_fn(epoch, 0, 0)
