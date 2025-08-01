from ray.rllib.algorithms.ppo import PPOConfig
from ray import train, tune


config =(
    PPOConfig().environment("Pendulum-v1")
    .training(lr=tune.grid_search([0.001, 0.0005, 0.0001]))
)

tuner = tune.Tuner(
    config.algo_class,
    param_space=config,
    run_config=train.RunConfig(
        stop={"env_runners/episode_return_mean": -1100.0}
    )
)

results = tuner.fit()


best_result = results.get_best_result("env_runners/episode_return_mean", mode="max")
best_checkpoint = best_result.checkpoint
print("Best checkpoint path:", best_checkpoint.path)