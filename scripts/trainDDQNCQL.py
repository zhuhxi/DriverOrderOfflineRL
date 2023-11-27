import sys
sys.path.append('/tmp-data/zhx/DriverOrderOfflineRL/tianshou')
sys.path.append('/tmp-data/zhx/DriverOrderOfflineRL/gym')

import pandas as pd
from time import *
from pprint import pprint
import os
current_dir = os.getcwd()

# buffer_data = pd.read_csv('/tmp-data/yanhaoyue/workspace/RL/data/support_feature_buffer.csv', sep='\t')

import gym
import numpy as np
from collections import deque, namedtuple
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb
import argparse
import glob
import random
from types import SimpleNamespace
import ast

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add a batch of experiences to memory."""
        for i in range(len(states)):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)
    
    def create_dataloader(self):
        """Create a DataLoader from the replay buffer."""
        states = torch.tensor([e.state for e in self.memory], dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in self.memory], dtype=torch.long, device=self.device)
        rewards = torch.tensor([e.reward for e in self.memory], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([e.next_state for e in self.memory], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in self.memory], dtype=torch.long, device=self.device)

        dataset = TensorDataset(states, actions.unsqueeze(1), rewards.unsqueeze(1), next_states, dones.unsqueeze(1))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    
def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = current_dir + '/trained_models/' + args.run_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
    wandb.save(save_dir + save_name + str(ep) + ".pth")

def collect_data(dataset, buffer_source):
    s, a, r, s_, d = buffer_source
    dataset.add(s, a, r, s_, d)
    
import sys
sys.path.append('/tmp-data/yanhaoyue/workspace/RL/CQL/CQL-DQN')
from agent import CQLAgent

# config
parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--run_name", type=str, default="CQL-DDQN", help="Run name, default: CQL-DQN")
parser.add_argument("--env", type=str, default="order_filter", help="env name")
parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes, default: 200")
parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Maximal training dataset size, default: 100_000")
parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
parser.add_argument("--eps_frames", type=int, default=1e4, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e5")
parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")

config = SimpleNamespace()
# args = parser.parse_args()
config.run_name = "11-22-CQL-DDQN"
config.env = 'order_filter'
config.episodes = 10000
config.buffer_size = 1_000_000
config.seed = 1
config.min_eps = 1e-4
config.eps_frames = 1e4
config.log_video = 0
config.save_every = 1
config.eval_every = 100

np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)

batches = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)

eps = 1.
d_eps = 1 - config.min_eps
steps = 0
average10 = deque(maxlen=10)
total_steps = 0

# state size
agent = CQLAgent(state_size=109,
                    action_size=6,
                    device=device)


import pickle

# 保存DataLoader
def save_dataloader(dataloader, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataloader, f)

# 加载DataLoader
def load_dataloader(filename):
    with open(filename, 'rb') as f:
        dataloader = pickle.load(f)
    return dataloader

dataloader = load_dataloader('support_feature_ddqn.pkl')
# buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=32, device=device)

# s = buffer_data['s'].apply(ast.literal_eval).values
# a = buffer_data['a'].values
# r = buffer_data['r'].values
# s_ = buffer_data['s_'].apply(ast.literal_eval).values
# d = buffer_data['d'].values

# buffer.add_batch(s, a, r, s_, d)

# dataloader = buffer.create_dataloader()

pprint(config)

with wandb.init(project="CQL-DDQN", name=config.run_name, config=config, dir=current_dir):
    wandb.watch(agent.network, log="gradients", log_freq=10)

    for i in range(1, config.episodes+1):

        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experience
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            loss, cql_loss, bellmann_error = agent.learn((states, actions, rewards, next_states, dones))
            batches += 1

            if batches % config.eval_every == 0:

                wandb.log({
                           "Q Loss": loss,
                           "CQL Loss": cql_loss,
                           "Bellmann error": bellmann_error,
                           "Episode": i,
                           "Batches": batches})

        if (i %10 == 0) and config.log_video:
            mp4list = glob.glob('video/*.mp4')
            if len(mp4list) > 1:
                mp4 = mp4list[-2]
                wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

        if i % config.save_every == 0:
            save(config, save_name='', model=agent.network, wandb=wandb, ep=i)
            print("Episode: ", i, "Batches: ", batches)
            

