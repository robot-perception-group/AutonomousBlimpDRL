import argparse
import datetime
import os

import numpy as np
from blimp_env.envs import PlanarNavigateEnv
from blimp_env.envs.script import close_simulation, close_simulation_on_marvin
import ray
from ray import tune
from ray.rllib.agents import impala
from ray.tune.registry import register_env

# exp setup
ENV = PlanarNavigateEnv
AGENT = impala
AGENT_NAME = "IMPALA"
exp_name_posfix = "test"

freq = 30  # steps per second
one_day_ts = 24 * 3600 * freq
days = 30
TIMESTEP = int(days * one_day_ts)

parser = argparse.ArgumentParser()
parser.add_argument("--gui", type=bool, default=False, help="Start with gazebo gui")
parser.add_argument("--num_gpus", type=bool, default=1, help="Number of gpu to use")
parser.add_argument(
    "--num_workers", type=int, default=5, help="Number of workers to use"
)

parser.add_argument(
    "--stop-timesteps", type=int, default=TIMESTEP, help="Number of timesteps to train."
)
parser.add_argument(
    "--fcnet_hiddens", type=list, default=[64, 64], help="fully connected layers"
)
parser.add_argument(
    "--vf_share_layers",
    type=bool,
    default=False,
    help="Whether layers should be shared for the value function.",
)
parser.add_argument(
    "--use_lstm",
    type=bool,
    default=False,
    help="Whether to wrap the model with an LSTM",
)
parser.add_argument(
    "--max_seq_len", type=int, default=7, help="Max seq len for training the LSTM"
)
parser.add_argument(
    "--lstm_cell_size", type=int, default=64, help="Size of the LSTM cell"
)


def env_creator(env_config):
    return ENV(env_config)


if __name__ == "__main__":
    env_name = ENV.__name__
    agent_name = AGENT_NAME
    exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix
    exp_time = datetime.datetime.now().strftime("%c")
    exp_path = os.path.join("./logs", exp_name, exp_time)

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=False)

    register_env(env_name, env_creator)
    env_config = {
        "seed": 123,
        "simulation": {
            "gui": args.gui,
            "auto_start_simulation": True,
        },
        "action": {
            "act_noise_stdv": 0.0,
        },
        "observaion": {
            "noise_stdv": 0.0,
        },
        "success_threshhold": 20,
        "tracking_reward_weights": np.array(
            [0.0, 0.0, 1.0, 0.0]
        ),  # z_diff, planar_dist, psi_diff, u_diff
        "reward_weights": np.array([1.0, 1.0, 0.0]),  # success, tracking, action
    }
    model_config = {
        # "custom_model": "my_model",
        "fcnet_hiddens": args.fcnet_hiddens,
        "fcnet_activation": "tanh",
        "vf_share_layers": args.vf_share_layers,
        # == LSTM ==
        "use_lstm": args.use_lstm,
        "max_seq_len": args.max_seq_len,
        "lstm_cell_size": args.lstm_cell_size,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        # == Attention Nets ==
        "use_attention": False,
        "attention_dim": 16,
    }
    config =AGENT.DEFAULT_CONFIG.copy()
    rollout_fragment_length = 1200
    train_batch_size = args.num_workers * rollout_fragment_length
    config.update(
        {
            "env": "my_env",
            "env_config": env_config,
            "num_gpus": args.num_gpus,
            "model": model_config,
            "num_workers": args.num_workers,  # parallelism
            "num_envs_per_worker": 1,
            "framework": "torch",
            # == AGENT config ==
            "rollout_fragment_length": rollout_fragment_length,
            "vtrace": True,  # vtrace use Importance Sampling to reduce off-policy discrepency
            "vtrace_clip_rho_threshold": 1.0,  # target or behaviour value func converge to
            "vtrace_clip_pg_rho_threshold": 1.0,  # convergence speed
            "train_batch_size": train_batch_size,
            "num_sgd_iter": 1,
            "replay_proportion": 1.0,
            "replay_buffer_num_slots": 10,
            "learner_queue_size": 16,
            "learner_queue_timeout": 1e6,
            "broadcast_interval": 1,
            "grad_clip": 40.0,
            "lr": 5e-4,
            "lr_schedule": None,
            "decay": 0.99,
        }
    )
    stop = {
        "timesteps_total": args.stop_timesteps,
    }

    print(config)
    close_simulation()
    results = tune.run(
        AGENT_NAME,
        config=config,
        stop=stop,
        local_dir=exp_path,
        checkpoint_freq=5000,
        checkpoint_at_end=True,
        reuse_actors=False,
    )
    ray.shutdown()
