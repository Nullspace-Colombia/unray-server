import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np

class ParrotEnv(gym.Env):

    def __init__(self, config=None):
          self.observation_space = config.get(
              "obs_act_space",
              gym.spaces.Box(-1.0, 1.0,(1,), np.float32)
          )
          self.action_space = self.observation_space
          self._cur_obs = None
          self._episode_len = 0

    def reset(self,*,seed=None, options=None):
        self._episode_len = 0
        self._cur_obs = self.observation_space.sample()
        return self._cur_obs, {}

    def step(self, action):
        self._episode_len +=1
        terminated = truncated = self._episode_len >= 10
        reward = -sum(abs(self._cur_obs - action))
        return self._cur_obs, reward, terminated, truncated, {}


config = (
    PPOConfig()
    .environment(
        ParrotEnv
    )
)

ppo_w_custom_env = config.build_algo()
ppo_w_custom_env.train()