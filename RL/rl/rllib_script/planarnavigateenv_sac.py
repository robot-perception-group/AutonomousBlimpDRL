import argparse
import datetime
import os

from blimp_env.envs import PlanarNavigateEnv
from blimp_env.envs.script import close_simulation
from rl.rllib_script.agent.model import TorchBatchNormModel

import ray
from ray import tune
from ray.rllib.agents import sac
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import numpy as np 

# exp setup
ENV = PlanarNavigateEnv
AGENT = sac
AGENT_NAME = "SAC"
exp_name_posfix = "test"

ts = 6
one_day_ts = 24 * 3600 * ts
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


def env_creator(env_config):
    return ENV(env_config)


if __name__ == "__main__":
    register_env("my_env", env_creator)
    ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)

    env_name = ENV.__name__
    agent_name = AGENT_NAME
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
            "env": "my_env",
            "env_config": env_config,
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,  # parallelism
            "num_envs_per_worker": 1,
            "framework": "torch",
            # == AGENT config ==
            "Q_model": Q_model_config,
            "policy_model": policy_model_config,
            "tau": 5e-3,
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
            "grad_clip": None,
            "learning_starts": 1e4,
            "rollout_fragment_length": 1,
            "train_batch_size": 256,
            "target_network_update_freq": 0,
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
        config=config,
        stop=stop,
        local_dir=exp_path,
        checkpoint_freq=5000,
        checkpoint_at_end=True,
        reuse_actors=False,
    )
    ray.shutdown()
