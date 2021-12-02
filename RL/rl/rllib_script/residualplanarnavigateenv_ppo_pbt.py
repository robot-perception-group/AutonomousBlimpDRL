import argparse
import random

import ray
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import sample_from
from ray.tune.registry import register_env
from ray.tune.schedulers.pb2 import PB2
from rl.rllib_script.agent.model import TorchBatchNormModel

# exp setup
ENV = ResidualPlanarNavigateEnv
AGENT = ppo
AGENT_NAME = "PPO"
exp_name_posfix = "test"

ts = ENV.default_config()["policy_frequency"]
one_day_ts = 24 * 3600 * ts
days = 28
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
parser.add_argument("--t_ready", type=int, default=50000)
parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
parser.add_argument(
    "--criteria", type=str,
    default="timesteps_total")  # "training_iteration", "time_total_s",  "timesteps_total"


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
        "action": {"disable_servo":True,},
    }

    ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)
    model_config = {
        "custom_model": "bn_model",  
        "custom_model_config": {},
        "vf_share_layers": False,
    }

    config = AGENT.DEFAULT_CONFIG.copy()
    rollout_fragment_length = 200
    train_batch_size=args.num_workers*rollout_fragment_length
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
            "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
            "kl_coeff": 0.2,
            "horizon": ENV.default_config()["duration"],
            "rollout_fragment_length": rollout_fragment_length,
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
            "lr": sample_from(lambda spec: random.uniform(1e-4, 1e-5)),
            "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
            "vf_clip_param": 10.0,
            "observation_filter": "MeanStdFilter",
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
            "lambda": [0.9, 1.0],
            "clip_param": [0.1, 0.5],
            "lr": [1e-4, 1e-5],
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
        checkpoint_freq=2000,
        checkpoint_at_end=True,
        reuse_actors=False,
        max_failures=5,
        restore=restore,
        verbose=1,
    )
    ray.shutdown()
