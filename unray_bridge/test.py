import random

import ray
#from ray import air, tune
#from ray.tune.schedulers import PopulationBasedTraining
#from ray.tune.registry import register_env
from src.unray.envs.spaces import BridgeSpaces
from src.unray.envs.base_env import SingleAgentEnv
from src.unray.envs.base_env import MultiAgentEnv
from src.unray.unray_config import UnrayConfig
from ray.rllib.algorithms.ppo import PPOConfig


import pprint

import numpy as np

high = np.array(
    [10000]*36,
    dtype=np.float64,
)

env_config  = {
        "agent-1":{
            "observation": BridgeSpaces.Box(-high, high),
            "action":BridgeSpaces.MultiDiscrete([2,3]),
            "can_show": 36, # Amount of observations int obs stack
            "can_see": 0, # Amount of observations required in training 
            "obs_order": {   
                "agent-1": [i for i in range(36)]
            }
        },
        "agent-2":{
            "observation": BridgeSpaces.Box(-high, high),
            "action":BridgeSpaces.MultiDiscrete([2,3]),
            "can_show": 36, # Amount of observations int obs stack
            "can_see": 0, # Amount of observations required in training 
            "obs_order": {   
                "agent-2": [i for i in range(36)]
            }
        }
    }

ppo_config = PPOConfig()

ppo_config = ppo_config.training(gamma=0.99, lr=0.0001, clip_param=0.2, lambda_=0.95)
ppo_config = ppo_config.resources(num_gpus=1)
ppo_config = ppo_config.rollouts(num_rollout_workers=0)

# Create unray object
unray_config = UnrayConfig()

# Path
#path = "C:/Users/Valentina/Documents/1_Universidad/AI/4_Tests/NPC/R2" #"E:/Universidad/Codigo/Nullspace/UE5/AgentGardenProject/Models/soccer-v2"

# Create instance of single agent environment
env = MultiAgentEnv(env_config, "soccer-multi")

# Create algo instance
algo = unray_config.configure_algo(ppo_config, env)

#algo.restore(path) #= Algorithm.from_checkpoint(path)
# Train
for i in range (51):
    print("Iteración:"f" '{i}'")
    result = algo.train()
    print(f"EPISODE REWARD MEAN | {result['episode_reward_mean']} |")

    #if i % 5 == 0:

        #save_result = algo.save(path)#("C:/Users/gonza/AppData/Local/Temp/tmp10hjh2wd")
    #    print("An Algorithm checkpoint has been created inside directory: "f"'{save_result}'.")