import argparse
import datetime
import os
import random

import numpy as np
import ray
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation
from ray import tune
from ray.rllib.agents import sac
from ray.rllib.models import ModelCatalog
from ray.tune import run, sample_from
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from rl.rllib_script.agent.model import TorchBatchNormModel

# exp setup
ENV = ResidualPlanarNavigateEnv
AGENT = sac
AGENT_NAME = "SAC"
exp_name_posfix = "test"

ts = 6
one_day_ts = 24 * 3600 * ts
days = 21
TIMESTEP = int(days * one_day_ts)

restore=None

parser = argparse.ArgumentParser()
parser.add_argument("--gui", type=bool, default=False, help="Start with gazebo gui")
parser.add_argument("--num_gpus", type=bool, default=1, help="Number of gpu to use")
parser.add_argument(
    "--num_workers", type=int, default=7, help="Number of workers to use"
)
parser.add_argument(
    "--stop-timesteps", type=int, default=TIMESTEP, help="Number of timesteps to train."
)
parser.add_argument("--t_ready", type=int, default=1e4)
parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
parser.add_argument(
    "--criteria", type=str,
    default="timesteps_total")  # "training_iteration", "time_total_s"


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
    }

    ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)
    Q_model_config = {
        "custom_model": "bn_model",  
        "custom_model_config": {},
    }
    policy_model_config = {
        "custom_model": "bn_model", 
        "custom_model_config": {},
    }

    config = AGENT.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": env_name,
            "env_config": env_config,
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,  # parallelism
            "num_envs_per_worker": 1,
            "framework": "torch",
            # == Model ==
            "Q_model": Q_model_config,
            "policy_model": policy_model_config,
            # == Learning ==
            "tau": 5e-3,
            "timesteps_per_iteration": 600, 
            # === Replay buffer ===
            "buffer_size": int(1e6),
            "store_buffer_in_checkpoints": True,
            "prioritized_replay": True,
            # === Optimization ===
            "optimization": {
                "actor_learning_rate": 3e-4, 
                "critic_learning_rate": 3e-4, 
                "entropy_learning_rate": 3e-4, 
            },
            "grad_clip": 5,
            "learning_starts": TIMESTEP*0.05,
            "rollout_fragment_length": sample_from(
                lambda spec: random.randint(1, 5)),
            "train_batch_size": 256,
            "target_network_update_freq": sample_from(
                lambda spec: random.randint(0, 5)),
        }
    )
    stop = {
        "timesteps_total": args.stop_timesteps,
    }

    pb2 = PB2(
        time_attr=args.criteria,
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.t_ready,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the hyperparam search space
        hyperparam_bounds={
            "rollout_fragment_length": [1, 5],
            "target_network_update_freq": [0, 5],
        })

    print(config)
    if env_config["simulation"]["auto_start_simulation"]:
        close_simulation()
    results = tune.run(
        AGENT_NAME,
        name=exp_name,
        scheduler=pb2,
        config=config,
        stop=stop,
        checkpoint_freq=2500,
        checkpoint_at_end=True,
        reuse_actors=False,
        max_failures=5,
        restore=restore,
        verbose=1,
    )
    ray.shutdown()
