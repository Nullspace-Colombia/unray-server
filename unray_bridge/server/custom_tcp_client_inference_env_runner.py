import base64
from collections import defaultdict
import gzip
import json
import pathlib
import socket
import tempfile
import threading
import time
from typing import Collection, DefaultDict, List, Optional, Union

import gymnasium as gym
import numpy as np
import onnxruntime

from ray.rllib.core import (
    Columns,
    COMPONENT_RL_MODULE,
    DEFAULT_AGENT_ID,
    DEFAULT_MODULE_ID,
)
from ray.rllib.env import INPUT_ENV_SPACES
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.tcp_client_inference_env_runner import TcpClientInferenceEnvRunner, _send_message
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
from ray.rllib.utils.typing import EpisodeID, StateDict
from sympy.codegen import Print

torch, _ = try_import_torch()


@ExperimentalAPI
class CustomTcpClientInferenceEnvRunner(TcpClientInferenceEnvRunner):

    @override(TcpClientInferenceEnvRunner)
    def _process_episodes_message(self, msg_type, msg_body):
        # On-policy training -> we have to block until we get a new `set_state` call
        # (b/c the learning step is done and we can sent new weights back to all
        # clients).
        if msg_type == rllink.EPISODES_AND_GET_STATE:
            self._blocked_on_state = True

        print("msg type: ", msg_type)
        #print("msg body: ", msg_body)
        print("Blocked on state :: Process", self._blocked_on_state)
        agent_ids = list(msg_body["episodes"][0]["agents"].keys())

        episodes = []
        for episode_data in msg_body["episodes"]:
            default_data = episode_data["agents"][agent_ids[0]]
            episode = SingleAgentEpisode(
                observation_space=self.config.observation_space,
                observations=[np.array(o) for o in default_data[Columns.OBS]],
                action_space=self.config.action_space,
                actions=default_data[Columns.ACTIONS],
                rewards=default_data[Columns.REWARDS],
                extra_model_outputs={
                    Columns.ACTION_DIST_INPUTS: [
                        np.array(a) for a in default_data["extra_model_outputs"][Columns.ACTION_DIST_INPUTS]
                    ],
                    Columns.ACTION_LOGP: default_data["extra_model_outputs"][Columns.ACTION_LOGP],
                },
                terminated=default_data["is_terminated"],
                truncated=default_data["is_truncated"],
                len_lookback_buffer=0,
            )
            episodes.append(episode.to_numpy())

        # Push episodes into the to-be-returned list (for `sample()` requests).
        with self._sample_lock:
            if isinstance(self._episode_chunks_to_return, list):
                self._episode_chunks_to_return.extend(episodes)
            else:
                self._episode_chunks_to_return = episodes

    @override(TcpClientInferenceEnvRunner)
    def _send_set_state_message(self):
        with tempfile.TemporaryDirectory() as dir:
            onnx_file = pathlib.Path(dir) / "_temp_model.onnx"
            torch.onnx.export(
                self.module,
                {
                    "batch": {
                        "obs": torch.randn(1, *self.config.observation_space.shape)
                    }
                },
                onnx_file,
                export_params=True,
            )
            with open(onnx_file, "rb") as f:
                compressed = gzip.compress(f.read())
                onnx_binary = base64.b64encode(compressed).decode("utf-8")
        _send_message(
            self.client_socket,
            {
                "type": rllink.SET_STATE.name,
                "onnx_file": onnx_binary,
                WEIGHTS_SEQ_NO: self._weights_seq_no,
            },
        )
