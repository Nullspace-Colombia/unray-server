from functools import partial
import gymnasium as gym
import numpy as np
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.tcp_client_inference_env_runner import (
    TcpClientInferenceEnvRunner,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls
from  tcp_client_multiagent_env_runner import (TcpClientMultiAgentEnvRunner)
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy # Or the policy class for your algo


parser = add_rllib_example_script_args(
    default_reward=450.0, default_iters=200, default_timesteps=2000000
)
parser.set_defaults(
    enable_new_api_stack=True,
    num_env_runners=2,
)
parser.add_argument(
    "--port",
    type=int,
    default=5555,
    help="Port to listen for external env connections (per env_runner worker).",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Definir espacios comunes
    obs_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    # Políticas
    policy_ids = ["policy_1", "policy_2"]

    # Mapeo agentes → políticas
    def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
        if agent_id in ["agent_0", "agent_1"]:
            return "policy_1"
        else:
            return "policy_2"

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            observation_space=obs_space,
            action_space=act_space,
            env_config={"port": args.port},  # se incrementa automáticamente por worker_index
        )
        .env_runners(
            env_runner_cls=TcpClientMultiAgentEnvRunner,
            rollout_fragment_length=400,
        )
        .training(
            num_epochs=10,
            vf_loss_coeff=0.01,
        )
        .multi_agent(
            policies={
                policy_ids[0]: (
                None, gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32), gym.spaces.Discrete(2), {}),
                policy_ids[1]: (
                None, gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32), gym.spaces.Discrete(2), {}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["policy_1", "policy_2"],
        )
        .rl_module(model_config=DefaultModelConfig(vf_share_layers=True))

    )

    run_rllib_example_script_experiment(base_config, args)

