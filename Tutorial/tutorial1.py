from ray.rllib.algorithms.ppo import PPOConfig
from pprint import pprint
from ray import train, tune


config =(
    PPOConfig().environment("Pendulum-v1")
)


config.env_runners(num_env_runners=2)

config.training(
    lr=0.0002,
    train_batch_size_per_learner=2000,
    num_epochs=10
)
config.evaluation(
    evaluation_interval=1,

    evaluation_num_env_runners=2,

    evaluation_duration_unit="episodes",

    evaluation_duration=10,
)

ppo_with_evaluation = config.build_algo()

for _ in range(3):
    pprint(ppo_with_evaluation.train())