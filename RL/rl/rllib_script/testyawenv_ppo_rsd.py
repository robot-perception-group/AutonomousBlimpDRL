import argparse
import random

import numpy as np
import ray
from blimp_env.envs import TestYawEnv
from blimp_env.envs.script import close_simulation

from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.tune import sample_from
from ray.tune.registry import register_env
from ray.tune.utils.placement_groups import PlacementGroupFactory
from rl.rllib_script.agent.model import TorchBatchNormModel

# exp setup
ENV = TestYawEnv
AGENT = ppo
AGENT_NAME = "PPO"
exp_name_posfix = "withSuccessR_RsdEffect"

days = 1
one_day_ts = 24 * 3600 * ENV.default_config()["policy_frequency"]
TIMESTEP = int(days * one_day_ts)

restore = None
resume = True

parser = argparse.ArgumentParser()
parser.add_argument("--gui", type=bool, default=False, help="Start with gazebo gui")
parser.add_argument("--num_gpus", type=bool, default=1, help="Number of gpu to use")
parser.add_argument(
    "--num_workers", type=int, default=7, help="Number of workers to use"
)
parser.add_argument(
    "--stop-timesteps", type=int, default=TIMESTEP, help="Number of timesteps to train."
)


def env_creator(env_config):
    return ENV(env_config)


def my_train_fn(config, reporter):
    total_ts = config["total_timesteps"]
    vf_explained_var = 0

    # Train for n iterations with high LR
    agent1 = PPOTrainer(env="CartPole-v0", config=config)
    while vf_explained_var < 0.9:
        result = agent1.train()
        vf_explained_var = result["vf_explained_var"]
        reporter(**result)
        phase1_time = result["timesteps_total"]
    state = agent1.save()
    agent1.stop()

    # Train for n iterations with low LR
    agent2 = PPOTrainer(env="CartPole-v0", config=config)
    agent2.restore(state)
    for _ in range(total_ts - phase1_time):
        result = agent2.train()
        result["phase"] = 2
        result["timesteps_total"] += phase1_time  # keep time moving forward
        reporter(**result)
    agent2.stop()


def training_workflow(config, reporter):
    """
    https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
    https://github.com/ray-project/ray/blob/master/rllib/examples/rollout_worker_custom_workflow.py
    """
    # Setup policy and policy evaluation actors
    env = gym.make(config["env"])
    policy = CustomPolicy(env.observation_space, env.action_space, {})
    workers = [
        RolloutWorker.as_remote().remote(
            env_creator=lambda c: gym.make("CartPole-v0"), policy=CustomPolicy
        )
        for _ in range(config["num_workers"])
    ]

    for _ in range(config["num_iters"]):
        # Broadcast weights to the policy evaluation workers
        weights = ray.put({DEFAULT_POLICY_ID: policy.get_weights()})
        for w in workers:
            w.set_weights.remote(weights)

        # Gather a batch of samples
        T1 = SampleBatch.concat_samples(ray.get([w.sample.remote() for w in workers]))

        # Update the remote policy replicas and gather another batch of samples
        new_value = policy.w * 2.0
        for w in workers:
            w.for_policy.remote(lambda p: p.update_some_value(new_value))

        # Gather another batch of samples
        T2 = SampleBatch.concat_samples(ray.get([w.sample.remote() for w in workers]))

        # Improve the policy using the T1 batch
        policy.learn_on_batch(T1)

        # Do some arbitrary updates based on the T2 batch
        policy.update_some_value(sum(T2["rewards"]))

        reporter(**collect_metrics(remote_workers=workers))


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
        "simulation": {
            "gui": args.gui,
            "auto_start_simulation": True,
        },
        "observation": {
            "enable_psi_vel": True,
            "enable_rsd_act_in_obs": True,
        },
        "reward_weights": np.array(
            [1.0, 1.0, 0, 0]
        ),  # success, tracking, action, bonus
        "enable_residual_ctrl": True,
        "enable_early_stopping": False,
        "reward_scale": 0.1,
        "clip_reward": False,
    }

    ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)
    model_config = {
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
            "lambda": 0.9,
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
            "clip_param": 0.2,
            "grad_clip": 0.5,
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