import argparse
import datetime
import os

from blimp_env.envs import PlanarNavigateEnv2
from blimp_env.envs.script import close_simulation, close_simulation_on_marvin
import ray
from ray import tune
from ray.rllib.agents import sac
from ray.tune.registry import register_env

# exp setup
ENV = PlanarNavigateEnv2
AGENT = "SAC"
exp_name_posfix = "test"

ts = 10
one_day_ts = 24 * 3600 * ts
days = 1
TIMESTEP = int(days * one_day_ts)


parser = argparse.ArgumentParser()
parser.add_argument("--gui", type=bool, default=False, help="Start with gazebo gui")
parser.add_argument("--num_gpus", type=bool, default=1, help="Number of gpu to use")
parser.add_argument(
    "--num_workers", type=int, default=1, help="Number of workers to use"
)

parser.add_argument(
    "--stop-timesteps", type=int, default=TIMESTEP, help="Number of timesteps to train."
)
parser.add_argument(
    "--fcnet_hiddens", type=list, default=[64, 64], help="fully connected layers"
)


def env_creator(env_config):
    return ENV(env_config)


if __name__ == "__main__":
    register_env("my_env", env_creator)

    env_name = ENV.__name__
    agent_name = AGENT
    exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix
    exp_time = datetime.datetime.now().strftime("%c")
    exp_path = os.path.join("./logs", exp_name, exp_time)

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=False)

    env_config = {
        "seed": 123,
        "simulation": {
            "gui": args.gui,
            "auto_start_simulation": True,
        },
    }
    model_config = {
        "fcnet_hiddens": args.fcnet_hiddens,
        "free_log_std": True,
    }

    config = sac.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": "my_env",
            "env_config": env_config,
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,  # parallelism
            "num_envs_per_worker": 1,
            "framework": "torch",
            # == SAC config ==
            # "Q_model": model_config,
            # "policy_model": model_config,
        }
    )
    stop = {
        "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    print(config)
    if env_config["simulation"]["auto_start_simulation"]:
        close_simulation()
        # close_simulation_on_marvin()
    results = tune.run(
        AGENT,
        config=config,
        stop=stop,
        local_dir=exp_path,
        checkpoint_freq=5000,
        checkpoint_at_end=True,
        reuse_actors=False,
    )
    ray.shutdown()
