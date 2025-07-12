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

    # Los espacios comunes ya no son necesarios aquí, se definirán por política.
    # obs_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    # act_space = gym.spaces.Discrete(2)

    # Políticas
    # Definimos 4 políticas, una para cada agente, si es que cada agente tendrá una política diferente.
    # Si varios agentes compartirán política, agrupa los agent_id bajo la misma policy_id.
    policy_ids = ["policy_agent1", "policy_agent2", "policy_agent3", "policy_agent4"]

    # Mapeo agentes → políticas
    # Asignamos una política específica a cada agente.
    def policy_mapping_fn(agent_id: str, episode=None, **kwargs):
        if agent_id == "agent_1":
            return "policy_agent1"
        elif agent_id == "agent_2":
            return "policy_agent2"
        elif agent_id == "agent_3":
            return "policy_agent3"
        elif agent_id == "agent_4":
            return "policy_agent4"
        else:
            raise ValueError(f"Unknown agent ID: {agent_id}")

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            # No especificamos observation_space ni action_space aquí,
            # ya que serán definidos por cada política individualmente.
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
                # Agente 1: Acciones Discrete(3) Observaciones Box(6) (-1 a 1)
                policy_ids[0]: (
                    None,  # Usar la clase de política predeterminada (e.g., PPOTorchPolicy)
                    gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
                    gym.spaces.Discrete(3),
                    {},
                ),
                # Agente 2: Acciones Box(2) (-1 a 1) Observaciones Multibinary(8)
                policy_ids[1]: (
                    None,
                    gym.spaces.MultiBinary(8),
                    gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    {},
                ),
                # Agente 3: Acciones Multidiscrete([3,3]) Observaciones Discrete(30)
                policy_ids[2]: (
                    None,
                    gym.spaces.Discrete(30),
                    gym.spaces.MultiDiscrete([3, 3]),
                    {},
                ),
                # Agente 4: Acciones Multibinary(2) Observaciones Multidiscrete([30,30,30,30,30,30,30])
                policy_ids[3]: (
                    None,
                    gym.spaces.MultiDiscrete([30, 30, 30, 30, 30, 30, 30]),
                    gym.spaces.MultiBinary(2),
                    {},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            # Entrenar todas las políticas definidas.
            policies_to_train=policy_ids,
        )
        .rl_module(model_config=DefaultModelConfig(vf_share_layers=True))
    )

    run_rllib_example_script_experiment(base_config, args)