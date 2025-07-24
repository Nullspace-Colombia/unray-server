import base64
import math
from collections import defaultdict
import gzip
import json
import pathlib
import socket
import tempfile
import threading
import time
from typing import Collection, DefaultDict, List, Optional, Union, Dict, Any

import gymnasium as gym
import numpy as np
import onnxruntime
from gymnasium.spaces import MultiDiscrete, Box, Discrete

from ray.rllib.core import (
    Columns,
    COMPONENT_RL_MODULE,
    DEFAULT_AGENT_ID,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils.external_env_protocol import RLlink as rllink
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    WEIGHTS_SEQ_NO,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.numpy import softmax
from ray.rllib.utils.postprocessing import episodes
from ray.rllib.utils.typing import EpisodeID, ModelWeights, ResultDict, StateDict
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

torch, _ = try_import_torch()


@ExperimentalAPI
class TcpClientMultiAgentEnvRunner(EnvRunner, Checkpointable):
    """An EnvRunner communicating with an external env through a TCP socket.

    This implementation assumes:
    - Only one external client ever connects to this env runner.
    - The external client performs inference locally through an ONNX model. Thus,
    samples are sent in bulk once a certain number of timesteps has been executed on the
    client's side (no individual action requests).
    - A copy of the RLModule is kept at all times on the env runner, but never used
    for inference, only as a data (weights) container.
    TODO (sven): The above might be inefficient as we have to store basically two
     models, one in this EnvRunner, one in the env (as ONNX).
    - There is no environment and no connectors on this env runner. The external env
    is responsible for generating all the data to create episodes.
    """

    @override(EnvRunner)
    def __init__(self, *, config, **kwargs):
        """
        Initializes a TcpClientInferenceEnvRunner instance.

        Args:
            config: The AlgorithmConfig to use for setup.

        Keyword Args:
            port: The base port number. The server socket is then actually bound to
                `port` + self.worker_index.
        """
        super().__init__(config=config)

        self.worker_index: int = kwargs.get("worker_index", 0)

        self._weights_seq_by_agent_no = {
            "policy_1": 0,
            "policy_2": 0,
            "policy_3": 0,
            "policy_4": 0,
        }

        # Build the module from its spec.
        module_spec = self.config.get_multi_rl_module_spec(
            spaces=self.get_spaces(), inference_only=True
        )
        self.module = module_spec.build()

        self.host = "localhost"
        self.port = int(self.config.env_config.get("port", 5555)) + self.worker_index
        self.server_socket = None
        self.client_socket = None
        self.address = None

        self.metrics = MetricsLogger()

        self._episode_chunks_to_return: Optional[List[MultiAgentEpisode]] = None
        self._done_episodes_for_metrics: List[MultiAgentEpisode] = []
        self._ongoing_episodes_for_metrics: DefaultDict[
            EpisodeID, List[MultiAgentEpisode]
        ] = defaultdict(list)

        self._sample_lock = threading.Lock()
        self._on_policy_lock = threading.Lock()
        self._blocked_on_state_by_agent = {
            "policy_1": False,
            "policy_2": False,
            "policy_3": False,
            "policy_4": False
        }

        # Start a background thread for client communication.
        self.thread = threading.Thread(
            target=self._client_message_listener, daemon=True
        )
        self.thread.start()

    @override(EnvRunner)
    def assert_healthy(self):
        """Checks that the server socket is open and listening."""
        assert (
            self.server_socket is not None
        ), "Server socket is None (not connected, not listening)."

    @override(EnvRunner)
    def sample(self, **kwargs):
        while True:
            with self._sample_lock:
                if self._episode_chunks_to_return is not None:
                    num_env_steps = 0
                    num_episodes_completed = 0
                    for eps in self._episode_chunks_to_return:
                        if eps.is_done:
                            self._done_episodes_for_metrics.append(eps)
                            num_episodes_completed += 1
                        else:
                            self._ongoing_episodes_for_metrics[eps.id_].append(eps)
                        num_env_steps += len(eps)

                    ret = self._episode_chunks_to_return
                    self._episode_chunks_to_return = None

                    if ret:
                        sample_episode = ret[0]
                        sample_obs = sample_episode.get_observations(indices=[-1], return_list=True)[0]
                        MultiAgentEnvRunner._increase_sampled_metrics(
                            self, num_env_steps, sample_obs, sample_episode
                        )

                    return ret
            time.sleep(0.01)

    @override(EnvRunner)
    def get_metrics(self) -> ResultDict:
        # Compute per-episode metrics (only on already completed episodes).
        for eps in self._done_episodes_for_metrics:
            assert eps.is_done
            episode_length = len(eps)
            agent_steps = defaultdict(
                int,
                {str(aid): len(sa_eps) for aid, sa_eps in eps.agent_episodes.items()},
            )
            episode_return = eps.get_return()
            episode_duration_s = eps.get_duration_s()

            agent_episode_returns = defaultdict(
                float,
                {
                    str(sa_eps.agent_id): sa_eps.get_return()
                    for sa_eps in eps.agent_episodes.values()
                },
            )
            module_episode_returns = defaultdict(
                float,
                {
                    sa_eps.module_id: sa_eps.get_return()
                    for sa_eps in eps.agent_episodes.values()
                },
            )

            # Don't forget about the already returned chunks of this episode.
            if eps.id_ in self._ongoing_episodes_for_metrics:
                for eps2 in self._ongoing_episodes_for_metrics[eps.id_]:
                    return_eps2 = eps2.get_return()
                    episode_length += len(eps2)
                    episode_return += return_eps2
                    episode_duration_s += eps2.get_duration_s()

                    for sa_eps in eps2.agent_episodes.values():
                        return_sa = sa_eps.get_return()
                        agent_steps[str(sa_eps.agent_id)] += len(sa_eps)
                        agent_episode_returns[str(sa_eps.agent_id)] += return_sa
                        module_episode_returns[sa_eps.module_id] += return_sa

                del self._ongoing_episodes_for_metrics[eps.id_]

            self._log_episode_metrics(
                episode_length,
                episode_return,
                episode_duration_s,
                agent_episode_returns,
                module_episode_returns,
                dict(agent_steps),
            )

        # Now that we have logged everything, clear cache of done episodes.
        self._done_episodes_for_metrics.clear()

        # Return reduced metrics.
        return self.metrics.reduce()

    def get_spaces(self):
        return {
            INPUT_ENV_SPACES: (self.config.observation_space, self.config.action_space),
            **{
                mid: (o, self.config.action_space[mid])
                for mid, o in self.config.observation_space.items()
            },
        }

    @override(MultiAgentEnvRunner)
    def stop(self):
        """Closes the client and server sockets."""
        self._close_sockets_if_necessary()

    @override(Checkpointable)
    def get_ctor_args_and_kwargs(self):
        return (
            (),  # *args
            {"config": self.config},  # **kwargs
        )

    @override(Checkpointable)
    def get_checkpointable_components(self):
        return [
            (COMPONENT_RL_MODULE, self.module),
        ]

    @override(Checkpointable)
    def get_state(
        self,
        components: Optional[Union[str, Collection[str]]] = None,
        *,
        not_components: Optional[Union[str, Collection[str]]] = None,
        **kwargs,
    ) -> StateDict:
        return {}

    @override(Checkpointable)
    def set_state(self, state: StateDict) -> None:
        # Update the RLModule state.

        #print ("State ", state)

        if COMPONENT_RL_MODULE in state:
            # A missing value for WEIGHTS_SEQ_NO or a value of 0 means: Force the
            # update.
            weights_seq_no = state.get(WEIGHTS_SEQ_NO, 0)
            print("w_seq: ",weights_seq_no)
            # Only update the weigths, if this is the first synchronization or
            # if the weights of this `EnvRunner` lacks behind the actual ones.


            rl_module_state = state[COMPONENT_RL_MODULE]
            if isinstance(rl_module_state, dict):

                for policy_id, policy_state in rl_module_state.items():
                    print("policy_id: ",policy_id)

                    if weights_seq_no == 0 or self._weights_seq_by_agent_no[policy_id] < weights_seq_no:
                        if policy_id not in self.module:
                            raise ValueError(f"Policy {policy_id} not found in module.")
                        self.module[policy_id].set_state(policy_state)
                        # Update our weights_seq_no, if the new one is > 0.
                        if weights_seq_no > 0:
                            self._weights_seq_by_agent_no[policy_id] = weights_seq_no

                        print("w_seq - ", policy_id, ": ", self._weights_seq_by_agent_no[policy_id])

                    if self._blocked_on_state_by_agent[policy_id] is True:
                        print("Blocked on state - ", policy_id, ": ", self._blocked_on_state_by_agent[policy_id])
                        self._send_set_state_message(policy_id, only_get_state = False)
                        self._blocked_on_state_by_agent[policy_id] = False
            else:
                raise ValueError("Expected multi-policy state dictionary.")




    def _client_message_listener(self):
        """Entry point for the listener thread."""

        # Set up the server socket and bind to the specified host and port.
        self._recycle_sockets()

        # Enter an endless message receival- and processing loop.
        while True:
            # As long as we are blocked on a new state, sleep a bit and continue.
            # Do NOT process any incoming messages (until we send out the new state
            # back to the client).
            #if all(self._blocked_on_state_by_agent.values()):
            #    time.sleep(0.01)
            #    continue

            try:
                # Blocking call to get next message.
                msg_type, msg_body = _get_message(self.client_socket)

                # Process the message received based on its type.
                # Initial handshake.
                if msg_type == rllink.PING:
                    self._send_pong_message()

                # Episode data from the client.
                elif msg_type in [
                    rllink.EPISODES,
                    rllink.EPISODES_AND_GET_STATE,
                ]:
                    #print(msg_body)
                    self._process_episodes_message(msg_type, msg_body)

                # Client requests the state (model weights).
                elif msg_type == rllink.GET_STATE:
                    self._send_set_state_message([], only_get_state=True)

                # Clients requests some (relevant) config information.
                elif msg_type == rllink.GET_CONFIG:
                    self._send_set_config_message()

            except ConnectionError as e:
                print(f"Messaging/connection error {e}! Recycling sockets ...")
                self._recycle_sockets(5.0)
                continue

    def _recycle_sockets(self, sleep: float = 0.0):
        # Close all old sockets, if they exist.
        self._close_sockets_if_necessary()

        time.sleep(sleep)

        # Start listening on the configured port.
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow reuse of the address.
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        # Listen for a single connection.
        self.server_socket.listen(1)
        print(f"Waiting for client to connect to port {self.port}...")

        self.client_socket, self.address = self.server_socket.accept()
        print(f"Connected to client at {self.address}")

    def _close_sockets_if_necessary(self):
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()

    def _send_pong_message(self):
        _send_message(self.client_socket, {"type": rllink.PONG.name})

    def _process_episodes_message(self, msg_type, msg_body):
        agent_ids = list(msg_body["episodes"][0]["agents"].keys())
        for agent in agent_ids:
            if msg_type == rllink.EPISODES_AND_GET_STATE:
                self._blocked_on_state_by_agent[agent] = True

        episodes = []

        for episode_data in msg_body["episodes"]:

            timesteps = len(next(iter(episode_data["agents"].values()))["actions"])  # asumimos longitud uniforme
            observations = [
                {
                    agent_id: np.asarray(agent_data["obs"][t], dtype=np.float32)
                    for agent_id, agent_data in episode_data["agents"].items()
                }
                for t in range(timesteps + 1)
            ]

            actions = [
                {
                    agent_id: np.asarray(agent_data["actions"][t], dtype=np.float32)
                    for agent_id, agent_data in episode_data["agents"].items()
                }
                for t in range(timesteps)
            ]

            rewards = [
                {
                    agent_id: np.asarray(agent_data["rewards"][t], dtype=np.float32)
                    for agent_id, agent_data in episode_data["agents"].items()
                }
                for t in range(timesteps)
            ]

            extra_model_outputs = []

            for t in range(timesteps):
                timestep_dict = {}
                for agent_id, agent_data in episode_data["agents"].items():
                    timestep_dict[agent_id] = {
                        "action_dist_inputs": np.asarray(agent_data["extra_model_outputs"]["action_dist_inputs"][t],
                                                         dtype=np.float32),
                        "action_logp": np.asarray(agent_data["extra_model_outputs"]["action_logp"][t],
                                                  dtype=np.float32),
                    }
                extra_model_outputs.append(timestep_dict)

            act_space = {
                "policy_1": gym.spaces.Discrete(3),
                "policy_2": gym.spaces.Box(0, 1.0, shape=(2,), dtype=np.float32),
                "policy_3": gym.spaces.MultiDiscrete([2, 2], dtype=np.int32),
                "policy_4": gym.spaces.Discrete(3)
            }

            extra_model_outputs = []

            for t in range(timesteps):
                timestep_dict = {}
                for agent_id, agent_data in episode_data["agents"].items():
                    space = act_space[agent_id]
                    logits = agent_data["extra_model_outputs"]["action_dist_inputs"][t]
                    logp = agent_data["extra_model_outputs"]["action_logp"][t]

                    # Valida dimensiones
                    logits = np.asarray(logits, dtype=np.float32)
                    logp = np.asarray(logp, dtype=np.float32)

                    if isinstance(space, Discrete):
                        assert logits.shape == (
                            space.n,), f"{agent_id}: expected shape {(space.n,)}, got {logits.shape}"
                    elif isinstance(space, Box):
                        expected = 2 * space.shape[0]
                        assert logits.shape == (
                            expected,), f"{agent_id}: expected shape {(expected,)}, got {logits.shape}"
                    elif isinstance(space, MultiDiscrete):
                        expected = np.sum(space.nvec)
                        assert logits.shape == (
                            expected,), f"{agent_id}: expected shape {(expected,)}, got {logits.shape}"
                    else:
                        raise ValueError(f"Unsupported action space for agent {agent_id}")

                    timestep_dict[agent_id] = {
                        "action_dist_inputs": logits,
                        "action_logp": logp,
                    }

                extra_model_outputs.append(timestep_dict)
            # Construir diccionarios finales de terminación y truncamiento
            terminateds = {
                agent_id: data.get("is_terminated", False)
                for agent_id, data in episode_data.items()
            }
            terminateds["__all__"] = all(terminateds.values())

            truncateds = {
                agent_id: data.get("is_truncated", False)
                for agent_id, data in episode_data.items()
            }
            truncateds["__all__"] = all(truncateds.values())

            POLICY_BY_AGENT = {
                "policy_1": "policy_1",
                "policy_2": "policy_2",
                "policy_3": "policy_3",
                "policy_4": "policy_4",
            }
            agents_data = episode_data["agents"]

            module_ids = {
                agent_id: POLICY_BY_AGENT[agent_id]  # o policy_mapping_fn(agent_id)
                for agent_id in agents_data.keys()
            }


            # Crear el MultiAgentEpisode
            episode = MultiAgentEpisode(
                observation_space=self.config.observation_space,
                action_space=self.config.action_space,
                observations=observations,
                actions=actions,
                rewards=rewards,
                extra_model_outputs=extra_model_outputs,
                terminateds=terminateds,
                truncateds=truncateds,
                len_lookback_buffer=0,
                agent_module_ids=module_ids,
            )

            episodes.append(episode)

        with self._sample_lock:
            if isinstance(self._episode_chunks_to_return, list):
                self._episode_chunks_to_return.extend(episodes)
            else:
                self._episode_chunks_to_return = episodes

    def get_multi_agent_module_ids(self):
        multi_agent_config = self.config.multi_agent()  # llamado correcto
        return list(multi_agent_config["policies"].keys())

    def get_spaces_for_module(self, module_id):
        multi_agent_config = self.config.multi_agent()
        policy_spec = multi_agent_config["policies"][module_id]
        # Puede venir explícito o usar los de config base
        obs_space = policy_spec[1] if policy_spec[1] is not None else self.config.observation_space
        act_space = policy_spec[2] if policy_spec[2] is not None else self.config.action_space

        return {
            "obs": obs_space,
            "action": act_space
        }

    """    def get_rl_module_spec_for_module(self, module_id, spaces, inference_only=False):
            # Usa la clase del módulo base desde el config
            module_class = self.module.__class__
    
            return RLModuleSpec(
                module_class=module_class,
                observation_space=spaces["obs"],
                action_space=spaces["action"],
                model_config=self.config.model,
                catalog_class=self.config.get("catalog_class", None),
                inference_only=inference_only,
            )
    """

    def _send_set_state_message(self, policy_ids, only_get_state):
        onnx_models = {}

        for module_id in self.get_multi_agent_module_ids():

            if module_id not in policy_ids and not only_get_state:
                continue

            spaces = self.get_spaces_for_module(module_id)
            obs_space = spaces["obs"]

            # Exportar a ONNX
            with tempfile.TemporaryDirectory() as dir:
                onnx_file = pathlib.Path(dir) / f"{module_id}_model.onnx"
                dummy_obs = torch.randn(1, *obs_space.shape)

                torch.onnx.export(
                    self.module[module_id],
                    {"batch": {"obs": dummy_obs}},
                    onnx_file,
                    export_params=True,
                )

                with open(onnx_file, "rb") as f:
                    compressed = gzip.compress(f.read())
                    encoded = base64.b64encode(compressed).decode("utf-8")
                    onnx_models[module_id] = encoded

        # Enviar todos los modelos al cliente
        _send_message(
            self.client_socket,
            {
                "type": rllink.SET_STATE.name,
                "onnx_files": onnx_models,
                WEIGHTS_SEQ_NO: str(self._weights_seq_by_agent_no),
            },
        )

    def _send_set_config_message(self):
        _send_message(
            self.client_socket,
            {
                "type": rllink.SET_CONFIG.name,
                "env_steps_per_sample": self.config.get_rollout_fragment_length(
                    worker_index=self.worker_index
                ),
                "force_on_policy": True,
            },
        )

    def _log_episode_metrics(
        self,
        length,
        ret,
        sec,
        agents=None,
        modules=None,
        agent_steps=None,
    ):
        # Log general episode metrics.

        # Use the configured window, but factor in the parallelism of the EnvRunners.
        # As a result, we only log the last `window / num_env_runners` steps here,
        # b/c everything gets parallel-merged in the Algorithm process.
        win = max(
            1,
            int(
                math.ceil(
                    self.config.metrics_num_episodes_for_smoothing
                    / (self.config.num_env_runners or 1)
                )
            ),
        )

        self.metrics.log_dict(
            {
                EPISODE_LEN_MEAN: length,
                EPISODE_RETURN_MEAN: ret,
                EPISODE_DURATION_SEC_MEAN: sec,
                **(
                    {
                        # Per-agent returns.
                        "agent_episode_returns_mean": agents,
                        # Per-RLModule returns.
                        "module_episode_returns_mean": modules,
                        "agent_steps": agent_steps,
                    }
                    if agents is not None
                    else {}
                ),
            },
            window=win,
        )
        # For some metrics, log min/max as well.
        self.metrics.log_dict(
            {
                EPISODE_LEN_MIN: length,
                EPISODE_RETURN_MIN: ret,
            },
            reduce="min",
            window=win,
        )
        self.metrics.log_dict(
            {
                EPISODE_LEN_MAX: length,
                EPISODE_RETURN_MAX: ret,
            },
            reduce="max",
            window=win,
        )


def _send_message(sock_, message: dict):
    """Sends a message to the client with a length header."""
    body = json.dumps(message).encode("utf-8")
    header = str(len(body)).zfill(8).encode("utf-8")
    try:
        sock_.sendall(header + body)
        print("Sending message..")
        #print("Header:", header)
        print("Type:", message["type"])
        #print("Full Message (hex):", (header + body).hex(' ', 1))  # Byte por byte en hex
        #print("As UTF-8 string:", (header + body).decode('utf-8', errors='replace'))
    except Exception as e:
        raise ConnectionError(
            f"Error sending message {message} to server on socket {sock_}! "
            f"Original error was: {e}"
        )


def _get_message(sock_):
    """Receives a message from the client following the length-header protocol."""
    try:
        # Read the length header (8 bytes)
        header = _get_num_bytes(sock_, 8)
        msg_length = int(header.decode("utf-8"))
        # Read the message body
        body = _get_num_bytes(sock_, msg_length)
        # Decode JSON.
        message = json.loads(body.decode("utf-8"))
        # Check for proper protocol.
        if "type" not in message:
            raise ConnectionError(
                "Protocol Error! Message from peer does not contain `type` field."
            )
        return rllink(message.pop("type")), message
    except Exception as e:
        raise ConnectionError(
            f"Error receiving message from peer on socket {sock_}! "
            f"Original error was: {e}"
        )


def _get_num_bytes(sock_, num_bytes):
    """Helper function to receive a specific number of bytes."""
    data = b""
    while len(data) < num_bytes:
        packet = sock_.recv(num_bytes - len(data))
        if not packet:
            raise ConnectionError(f"No data received from socket {sock_}!")
        data += packet
    return data
