import argparse
import random
import numpy as np
import ray
from blimp_env.envs import TestYawEnv
from blimp_env.envs.script import close_simulation
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import sample_from
from ray.tune.registry import register_env

from rl.rllib_script.agent.model import TorchBatchNormModel

# exp setup
ENV = TestYawEnv
AGENT = ppo
AGENT_NAME = "PPO"
exp_name_posfix = "badPID_LSTM_Clipparam"

days = 1
one_day_ts = 24 * 3600 * ENV.default_config()["policy_frequency"]
TIMESTEP = int(days * one_day_ts)

restore = None
resume = False

parser = argparse.ArgumentParser()
parser.add_argument("--gui", type=bool, default=False, help="Start with gazebo gui")
parser.add_argument("--num_gpus", type=bool, default=1, help="Number of gpu to use")
parser.add_argument(
    "--num_workers", type=int, default=7, help="Number of workers to use"
)
parser.add_argument(
    "--stop-timesteps", type=int, default=TIMESTEP, help="Number of timesteps to train."
)
parser.add_argument("--use_lstm", type=bool, default=True, help="enable lstm cell")
parser.add_argument(
    "--lstm_use_prev",
    type=bool,
    default=True,
    help="allow lstm use previous action and reward",
)


def env_creator(env_config):
    return ENV(env_config)


if __name__ == "__main__":
    env_name = ENV.__name__
    agent_name = AGENT_NAME
    exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix

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
        "observation": {
            "enable_rsdact_feedback": True,
        },
        "reward_weights": np.array([1.0, 1.0, 0]),  # success, tracking, action
        "enable_residual_ctrl": True,
        "reward_scale": 0.1,
        "clip_reward": False,
        "mixer_type": "absolute",
        "beta": 0.5,
        "pid_param": np.array([1.0, 0.0, 0.0]),  # bad:[1,0,0], good:[1,0,0.05]
    }

    ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)
    model_config = {
        "use_lstm": args.use_lstm,
        "lstm_cell_size": 64,
        "lstm_use_prev_action": args.lstm_use_prev,
        "lstm_use_prev_reward": args.lstm_use_prev,
        "custom_model": "bn_model",
        "custom_model_config": {},
    }

    config = AGENT.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": env_name,
            "env_config": env_config,
            "log_level": "INFO",
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,  # parallelism
            "num_envs_per_worker": 1,
            "framework": "torch",
            "model": model_config,
            # == Learning ==
            "gamma": 0.999,
            "lambda": 0.9,  # gae lambda
            "kl_coeff": 1.0,
            "horizon": 400,
            "rollout_fragment_length": 400,
            "train_batch_size": args.num_workers * 1600,
            "sgd_minibatch_size": 1024,
            "num_sgd_iter": 32,
            "lr": 1e-4,
            "lr_schedule": [
                [0, 1e-4],
                [args.stop_timesteps, 1e-12],
            ],
            "clip_param": tune.grid_search([0.2, 2.0, 10.0]),
            "vf_clip_param": tune.grid_search([10, None]),
            "grad_clip": tune.grid_search([0.5, 5.0, 50.0]),
            "observation_filter": "NoFilter",
            "batch_mode": "truncate_episodes",
        }
    )
    stop = {
        "timesteps_total": args.stop_timesteps,
    }

    print(config)
    if env_config["simulation"]["auto_start_simulation"]:
        close_simulation()

    results = tune.run(
        AGENT_NAME,
        name=exp_name,
        config=config,
        stop=stop,
        checkpoint_freq=2000,
        checkpoint_at_end=True,
        reuse_actors=False,
        restore=restore,
        resume=resume,
        max_failures=3,
        verbose=1,
    )
    ray.shutdown()
