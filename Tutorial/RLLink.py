"""Example of running against a TCP-connected external env performing its own inference.

The example uses a custom EnvRunner (TcpClientInferenceEnvRunner) to allow
connections from one or more TCP clients to RLlib's EnvRunner actors, which act as
RL servers.
In this example, action inference for stepping the env is performed on the client's
side, meaning the client computes all actions itself, applies them to the env logic,
collects episodes of experiences, and sends these (in bulk) back to RLlib for training.
Also, from time to time, the updated model weights have to be sent from RLlib (server)
back to the connected clients.
Note that RLlib's new API stack does not yet support individual action requests, where
action computations happen on the RLlib (server) side.

This example:
    - demonstrates how RLlib can be hooked up to an externally running complex simulator
    through TCP connections.
    - shows how a custom EnvRunner subclass can be configured allowing users to
    implement their own logic of connecting to external processes and customize the
    messaging protocols.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --port 5555

Use the `--port` option to change the default port (5555) to some other value.
Make sure that you do the same on the client side.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see something like this on your terminal. Note that the dummy CartPole
client (which runs in a thread for the purpose of this example here) might throw
a disconnection error at the end, b/c RLlib closes the server socket when done training.

+----------------------+------------+--------+------------------+
| Trial name           | status     |   iter |   total time (s) |
|                      |            |        |                  |
|----------------------+------------+--------+------------------+
| PPO_None_3358e_00000 | TERMINATED |     40 |          32.2649 |
+----------------------+------------+--------+------------------+
+------------------------+------------------------+
|  episode_return_mean  |   num_env_steps_sample |
|                       |             d_lifetime |
|-----------------------+------------------------|
|                458.68 |                 160000 |
+-----------------------+------------------------+

From the dummy client (thread), you should see at the end:
```
ConnectionError: Error receiving message from peer on socket ...
```
"""
# client.py

import argparse
from ray.rllib.env.tcp_client_inference_env_runner import _dummy_client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy CartPole client for RLlib TCP server.")
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="Port to connect to the RLlib server. Should match server port + runner index.",
    )
    args = parser.parse_args()

    _dummy_client(port=args.port)
