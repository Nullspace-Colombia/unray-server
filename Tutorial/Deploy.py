from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule

rl_module = RLModule.from_checkpoint(
    Path("C:/Users/gonza/ray_results/PPO_2025-06-27_11-55-51/PPO_Pendulum-v1_96fd9_00000_0_lr=0.0010_2025-06-27_11-55-54/checkpoint_000000")
    / "learner_group"
    / "learner"
    / "rl_module"
    / "default_policy"
)

env = gym.make("Pendulum-v1", render_mode="human")

episode_return = 0.0
done = False

obs, info = env.reset()

while not done:

    env.render()

    obs_batch = torch.from_numpy(obs).unsqueeze(0)
    model_outputs = rl_module.forward_inference({"obs": obs_batch})

    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

    greedy_action = np.clip(
        action_dist_params[0:1],
        a_min=env.action_space.low[0],
        a_max=env.action_space.high[0],
    )

    obs, reward, terminated, truncated, info = env.step(greedy_action)

    episode_return += reward
    done = terminated or truncated

print(f"Reached episode return of {episode_return}")