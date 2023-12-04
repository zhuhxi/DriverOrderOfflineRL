import sys
sys.path.append('/home/zhx/word/work/DriverOrderOfflineRL/cage-challenge-1/CybORG')

import ray
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import EnvCompatibility

from typing import Optional, Tuple, Union
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from gymnasium.core import ObsType

import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Shared.Actions import *
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers import *
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

# seed
np.random.seed(1)
torch.manual_seed(1)

# env
CYBORG = CybORG(path,'sim', agents={'Red': RedMeanderAgent})
class Wrapper(gym.Env):
    def __init__(self, env:ChallengeWrapper) -> None:
        self.env = env
        self.action_space = spaces.Discrete(54)
        self.observation_space = spaces.Box(low=-1.0, high=3.0, shape=((52,)), dtype=np.float32)
    
    def step(self, action=None):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

class CyborgEnv(gym.Env):
    def __init__(self, config=None) -> None:
        self.env = EnvCompatibility(Wrapper(ChallengeWrapper(env=CybORG(path,'sim', agents={'Red': RedMeanderAgent}), agent_name="Blue", max_steps=100)))
        self.action_space = spaces.Discrete(54)
        self.observation_space = spaces.Box(low=-1.0, high=3.0, shape=((52,)), dtype=np.float32)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        return self.env.step(action=action)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        return self.env.reset(seed=seed, options=options)

