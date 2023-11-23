import pandas as pd
import matplotlib.pyplot as plt
from time import *
from pprint import pprint
import os
import tornado.web
import json
import asyncio

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

feature = pd.read_csv('all_feature.txt', sep=' ', header=None)
feature_name = feature[0].values.tolist()

def load(model, save_dir=None, ep=None):
    import os
    import glob
    save_dir = '/tmp-data/zhx/DriverOrderOfflineRL/scripts/trained_models/11-22-CQL-DDQN/11-22-CQL-DDQN40.pth'

    if os.path.exists(save_dir):
        model.load_state_dict(torch.load(save_dir))
        print("Model loaded successfully.")
        print("Model path: ", save_dir)
    else:
        print("Model path does not exist. No model loaded.")
        
## agent
import sys
sys.path.append('/tmp-data/yanhaoyue/workspace/RL/CQL/CQL-DQN')
from agent import CQLAgent

# config
parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--run_name", type=str, default="CQL-DDQN", help="Run name, default: CQL-DQN")
parser.add_argument("--env", type=str, default="order_filter", help="env name")
parser.add_argument("--episodes", type=int, default=400, help="Number of episodes, default: 200")
parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
parser.add_argument("--eps_frames", type=int, default=1e4, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e5")
parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")

config = SimpleNamespace()
# args = parser.parse_args()
config.run_name = "11-22-CQL-DDQN_test"
config.env = 'order_filter'
config.episodes = 400
config.buffer_size = 1_000_000
config.seed = 1
config.min_eps = 1e-4
config.eps_frames = 1e4
config.log_video = 0
config.save_every = 1
config.eval_every = 100

## test core
np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)

batches = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eps = 1.
d_eps = 1 - config.min_eps
steps = 0
average10 = deque(maxlen=10)
total_steps = 0

agent = CQLAgent(state_size=109,
                    action_size=6,
                    device=device)

load(agent.network)

class predict(tornado.web.RequestHandler):
    def initialize(self):
        self.set_header("Content-Type", "application/json")

    async def post(self):
        data = json.loads(self.request.body)
        print("Predicting...,data:",data)
        try:
            features = [data[name] for name in feature_name]
            features = np.array(features)
            print("=======features=======",len(features),features)
            prediction = agent.get_action(state=features, epsilon=0)
            print("prediction",int(prediction[0]))
            resp_data = {
                "prediction": int(prediction[0])}
            self.write(json.dumps(resp_data))
        except KeyError:
            print("Key does not exist")
            self.write(json.dumps({"prediction": "-1"}))


def make_app():
    return tornado.web.Application([
        (r"/predict", predict),
    ])


async def main():
    tornado.log.enable_pretty_logging()
    app = make_app()
    app.listen(5000)
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()
    
if __name__ == "__main__":
    print("====server start=====")
    asyncio.run(main())