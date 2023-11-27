import sys
sys.path.append('/home/zhx/word/DriverOrderOfflineRL/cage-challenge-1/CybORG')
sys.path.append('/home/zhx/word/DriverOrderOfflineRL/tianshou')

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.env import SubprocVectorEnv

from typing import Any, Dict, Optional, Sequence, Tuple, Union


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cyborg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--step-per-epoch", type=int, default=100)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=1)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="cyborg.dqn")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
        help="use intrinsic curiosity module with this lr scale"
    )
    parser.add_argument(
        "--icm-reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward"
    )
    parser.add_argument(
        "--icm-forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM"
    )
    return parser.parse_args(args=[])


args = get_args()

import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Shared.Actions import *
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers import *

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# env
CYBORG = CybORG(path,'sim', agents={'Red': RedMeanderAgent})
env = ChallengeWrapper(env=CYBORG, agent_name="Blue", max_steps=args.step_per_epoch)
train_envs = SubprocVectorEnv([lambda: ChallengeWrapper(env=CybORG(path,'sim', agents={'Red': RedMeanderAgent}), 
                                                        agent_name="Blue", max_steps=args.step_per_epoch) for _ in range(1)])
test_envs = SubprocVectorEnv([lambda: ChallengeWrapper(env=CybORG(path,'sim', agents={'Red': RedMeanderAgent}), 
                                                       agent_name="Blue", max_steps=args.step_per_epoch) for _ in range(1)])

args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
print("Observations shape:", args.state_shape)
print("Actions shape:", args.action_shape)

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
            nn.Linear(state_shape, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, action_shape)
        )
#         with torch.no_grad():
#             self.output_dim = np.prod(self.net(torch.zeros(1, state_shape)).shape[1:])
#         if not features_only:
#             self.net = nn.Sequential(
#                 self.net, nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
#                 nn.Linear(512, np.prod(action_shape))
#             )
#             self.output_dim = np.prod(action_shape)
#         elif output_dim is not None:
#             self.net = nn.Sequential(
#                 self.net, nn.Linear(self.output_dim, output_dim),
#                 nn.ReLU(inplace=True)
#             )
#             self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state
    
    
# define model
net = DQN(args.state_shape[0], args.action_shape, args.device).to(args.device)
optim = torch.optim.Adam(net.parameters(), lr=args.lr)
# define policy
policy = DQNPolicy(
    net,
    optim,
    args.gamma,
    args.n_step,
    target_update_freq=args.target_update_freq
)

if args.icm_lr_scale > 0:
    feature_net = DQN(
        args.state_shape[0], args.action_shape, args.device, features_only=True
    )
    action_dim = np.prod(args.action_shape)
    feature_dim = feature_net.output_dim
    icm_net = IntrinsicCuriosityModule(
        feature_net.net,
        feature_dim,
        action_dim,
        hidden_sizes=[512],
        device=args.device
    )
    icm_optim = torch.optim.Adam(icm_net.parameters(), lr=args.lr)
    policy = ICMPolicy(
        policy, icm_net, icm_optim, args.icm_lr_scale, args.icm_reward_scale,
        args.icm_forward_loss_weight
    ).to(args.device)
    
# load a previous policy
if args.resume_path:
    policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    print("Loaded agent from: ", args.resume_path)
    

# replay buffer: `save_last_obs` and `stack_num` can be removed together
# when you have enough RAM
buffer = VectorReplayBuffer(
    args.buffer_size,
    buffer_num=len(train_envs),
    ignore_obs_next=True,
    save_only_last_obs=False,
    stack_num=args.frames_stack
)

# collector
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)

# log
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
args.algo_name = "dqn_icm" if args.icm_lr_scale > 0 else "dqn"
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
    return mean_rewards >= 20

def train_fn(epoch, env_step):
    # nature DQN setting, linear decay in the first 1M steps
    if env_step <= 1e6:
        eps = args.eps_train - env_step / 1e6 * \
            (args.eps_train - args.eps_train_final)
    else:
        eps = args.eps_train_final
    policy.set_eps(eps)
    if env_step % 1000 == 0:
        logger.write("train/env_step", env_step, {"train/eps": eps})

def test_fn(epoch, env_step):
    policy.set_eps(args.eps_test)

def save_checkpoint_fn(epoch, env_step, gradient_step):
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    ckpt_path = os.path.join(log_path, "checkpoint.pth")
    torch.save({"model": policy.state_dict()}, ckpt_path)
    return ckpt_path

# watch agent's performance
def watch():
    print("Setup test envs ...")
    policy.eval()
    policy.set_eps(args.eps_test)
    test_envs.seed(args.seed)
    if args.save_buffer_name:
        print(f"Generate buffer with size {args.buffer_size}")
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(test_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack
        )
        collector = Collector(policy, test_envs, buffer, exploration_noise=True)
        result = collector.collect(n_step=args.buffer_size)
        print(f"Save buffer into {args.save_buffer_name}")
        # Unfortunately, pickle will cause oom with 1M buffer size
        buffer.save_hdf5(args.save_buffer_name)
    else:
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(
            n_episode=args.test_num, render=args.render
        )
    rew = result["rews"].mean()
    print(f"Mean reward (over {result['n/ep']} episodes): {rew}")
    
if args.watch:
    watch()
    exit(0)

# test train_collector and start filling replay buffer
train_collector.collect(n_step=args.batch_size * args.training_num)
# trainer
result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    args.epoch,
    args.step_per_epoch,
    args.step_per_collect,
    args.test_num,
    args.batch_size,
    train_fn=train_fn,
    test_fn=test_fn,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    logger=logger,
    update_per_step=args.update_per_step,
    test_in_train=False,
    resume_from_log=args.resume_id is not None,
    save_checkpoint_fn=save_checkpoint_fn,
)

pprint.pprint(result)
watch()

