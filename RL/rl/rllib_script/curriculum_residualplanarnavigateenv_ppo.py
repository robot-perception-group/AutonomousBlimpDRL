import argparse

import numpy as np
import ray
import rl.rllib_script.agent.model
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation, spawn_simulation_on_different_port
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import sample_from
from ray.tune.registry import register_env
from rl.rllib_script.util import find_nearest_power_of_two

# exp setup
ENV = ResidualPlanarNavigateEnv
AGENT = ppo
TRAINER = PPOTrainer
AGENT_NAME = "PPO"
exp_name_posfix = "curriculum_test"

env_default_config = ENV.default_config()
duration = env_default_config["duration"]
simulation_frequency = env_default_config["simulation_frequency"]
policy_frequency = env_default_config["policy_frequency"]

days = 20
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
parser.add_argument("--use_lstm", type=bool, default=True, help="enable lstm cell")


def env_creator(env_config):
    return ENV(env_config)


def train_fn(
    config,
    reporter,
    phase,
    total_phase,
    passed_ts,
    total_ts,
    state=None,
):
    prev_phase_time = passed_ts
    trainer = TRAINER(env=env_name, config=config)
    if state is not None:
        trainer.restore(state)

    while passed_ts <= (total_ts * phase / total_phase):
        result = trainer.train()

        ts_total = result["timesteps_total"]
        passed_ts += ts_total

        result["phase"] = 1
        result["timesteps_total"] += prev_phase_time
        reporter(**result)

    state = trainer.save()
    trainer.stop()
    return state, passed_ts


def my_train_fn(config, reporter):
    total_ts = TIMESTEP
    passed_ts = 0
    total_phase = 3

    config.update({"lr": 1e-4})
    env_config = config["env_config"]
    env_config.update(
        {
            "beta": 0.3,
            "reward_weights": np.array([100, 1.0, 0.0]),
            "tracking_reward_weights": np.array([0.3, 0.4, 0.25, 0.05]),
            "success_threshhold": 10,
        }
    )
    env_config["simulation"].update(
        {
            "wind_speed": 0.5,
            "buoyancy_range": [0.95, 1.05],
        }
    )
    config.update({"env_config": env_config})
    state, passed_ts = train_fn(
        config=config,
        reporter=reporter,
        phase=1,
        total_phase=total_phase,
        passed_ts=passed_ts,
        total_ts=total_ts,
        state=None,
    )

    config.update({"lr": 5e-5})
    env_config = config["env_config"]
    env_config.update(
        {
            "beta": 0.4,
            "reward_weights": np.array([100, 0.9, 0.1]),
            "tracking_reward_weights": np.array([0.35, 0.35, 0.2, 0.1]),
            "success_threshhold": 7,
        }
    )
    env_config["simulation"].update(
        {
            "wind_speed": 0.8,
            "buoyancy_range": [0.9, 1.1],
        }
    )
    config.update({"env_config": env_config})
    state, passed_ts = train_fn(
        config=config,
        reporter=reporter,
        phase=2,
        total_phase=total_phase,
        passed_ts=passed_ts,
        total_ts=total_ts,
        state=state,
    )

    config.update({"lr": 1e-5})
    env_config = config["env_config"]
    env_config.update(
        {
            "beta": 0.5,
            "reward_weights": np.array([100, 0.85, 0.15]),
            "tracking_reward_weights": np.array([0.4, 0.3, 0.15, 0.15]),
            "success_threshhold": 5,
        }
    )
    env_config["simulation"].update(
        {
            "wind_speed": 1.2,
            "buoyancy_range": [0.85, 1.15],
        }
    )
    config.update({"env_config": env_config})
    state, passed_ts = train_fn(
        config=config,
        reporter=reporter,
        phase=3,
        total_phase=total_phase,
        passed_ts=passed_ts,
        total_ts=total_ts,
        state=state,
    )


if __name__ == "__main__":
    env_name = ENV.__name__
    agent_name = AGENT_NAME
    exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    register_env(env_name, env_creator)
    env_config = {
        "seed": 123,
        "DBG": True,
        "simulation": {
            "gui": args.gui,
            "auto_start_simulation": True,
            "enable_wind": True,
            "enable_wind_sampling": True,
            "wind_speed": 1.5,
            "enable_buoyancy_sampling": True,
            "enable_next_goal": True,
        },
        "observation": {
            "enable_rsdact_feedback": True,
            "enable_airspeed_sensor": True,
        },
        "action": {
            "disable_servo": False,
            "max_servo": -0.5,
        },
        "reward_weights": np.array([100, 1.0, 0.0]),  # success, tracking, action
        "tracking_reward_weights": np.array(
            [0.4, 0.3, 0.2, 0.1]
        ),  # z_diff, planar_dist, yaw_diff, vel_diff
        "success_threshhold": 5,  # [meters]
        "enable_residual_ctrl": True,
        "reward_scale": 0.01,
        "clip_reward": False,
        "mixer_type": "absolute",
        "beta": 0.4,
    }

    if args.use_lstm:
        custom_model = "bnrnn_model"
        custom_model_config = {
            "hidden_sizes": [64, 64],
            "lstm_cell_size": 64,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
        }
    else:
        custom_model = "bn_model"
        custom_model_config = {
            "actor_sizes": [64, 64],
            "critic_sizes": [128, 128],
        }
    model_config = {
        "custom_model": custom_model,
        "custom_model_config": custom_model_config,
    }

    if env_config["simulation"]["auto_start_simulation"]:
        close_simulation()

    train_batch_size = args.num_workers * 1600
    sgd_minibatch_size = find_nearest_power_of_two(train_batch_size / 10)
    episode_ts = duration * policy_frequency / simulation_frequency

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
    results = tune.run(
        my_train_fn,
        resources_per_trial=PPOTrainer.default_resource_request(config),
        name=exp_name,
        config=config,
        restore=restore,
        resume=args.resume,
        verbose=1,
    )
    ray.shutdown()
