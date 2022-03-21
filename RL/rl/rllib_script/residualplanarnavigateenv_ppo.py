import argparse

import ray
import rl.rllib_script.agent.model.ray_model
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from rl.rllib_script.util import find_nearest_power_of_two

# exp setup
ENV = ResidualPlanarNavigateEnv
AGENT = ppo
AGENT_NAME = "PPO"
exp_name_posfix = "test"

env_default_config = ENV.default_config()
duration = env_default_config["duration"]
simulation_frequency = env_default_config["simulation_frequency"]
policy_frequency = env_default_config["policy_frequency"]

days = 28
one_day_ts = 24 * 3600 * policy_frequency
TIMESTEP = int(days * one_day_ts)

restore = None

parser = argparse.ArgumentParser()
parser.add_argument("--gui", type=bool, default=False, help="Start with gazebo gui")
parser.add_argument("--num_gpus", type=bool, default=1, help="Number of gpu to use")
parser.add_argument(
    "--num_workers", type=int, default=7, help="Number of workers to use"
)
parser.add_argument(
    "--stop_timesteps", type=int, default=TIMESTEP, help="Number of timesteps to train."
)
parser.add_argument(
    "--resume", type=bool, default=False, help="resume the last experiment"
)
parser.add_argument(
    "--use_lstm",
    dest="use_lstm",
    default=False,
    action="store_true",
    help="enable lstm cell",
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
        "seed": tune.grid_search([123, 456, 789]),
        "seed": 123,
        "simulation": {
            "gui": args.gui,
            "auto_start_simulation": True,
        },
        "mixer_type": tune.grid_search(
            ["hybrid", "absolute"]
        ),  # absolute, relative, hybrid
        "mixer_param": (0.5, 0.7),  # alpha, beta
    }

    if args.use_lstm:
        custom_model = "bnrnn_model"
        custom_model_config = {
            "hidden_sizes": [64, 64],
            "lstm_cell_size": 64,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
        }
        print("BNRNN model selected")
    else:
        custom_model = "bn_model"
        custom_model_config = {
            "actor_sizes": [64, 64],
            "critic_sizes": [128, 128],
        }
        print("BN model selected")

    model_config = {
        "custom_model": custom_model,
        "custom_model_config": custom_model_config,
    }

    episode_ts = duration * policy_frequency / simulation_frequency
    train_batch_size = args.num_workers * 4 * episode_ts
    sgd_minibatch_size = find_nearest_power_of_two(train_batch_size / 10)

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
            "lambda": 0.9,
            "kl_coeff": 1.0,
            "horizon": episode_ts,
            "rollout_fragment_length": episode_ts,
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": 32,
            "lr": 1e-4,
            "lr_schedule": [
                [0, 1e-4],
                [args.stop_timesteps, 5e-6],
            ],
            "clip_param": 0.2,
            "vf_clip_param": 10,
            "grad_clip": 1.0,
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
        checkpoint_freq=500,
        checkpoint_at_end=True,
        reuse_actors=False,
        restore=restore,
        resume=args.resume,
        max_failures=3,
        verbose=1,
    )
    ray.shutdown()
