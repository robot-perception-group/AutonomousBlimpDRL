import argparse
import os
import tempfile
from datetime import datetime

import numpy as np
import ray
import rl.rllib_script.agent.model
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation, spawn_simulation_on_different_port
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import sample_from
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from rl.rllib_script.util import find_nearest_power_of_two

# exp setup
ENV = ResidualPlanarNavigateEnv
AGENT = ppo
TRAINER = PPOTrainer
AGENT_NAME = "PPO"
exp_name_posfix = "disturbed_curriculum_multigoal"

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


def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def train_fn(
    config,
    reporter,
    phase,
    passed_ts,
    total_phase,
    total_ts,
    exp_name,
    state=None,
):
    close_simulation()

    prev_phase_time = passed_ts
    print(
        f"Start Training Phase_{phase}: total_phase={total_phase}, total_ts={total_ts}, passed_ts={passed_ts}"
    )

    trainer = TRAINER(
        env=config["env"],
        config=config,
        logger_creator=custom_log_creator(
            os.path.expanduser("~/ray_results/" + exp_name), f"phase_{phase}"
        ),
    )
    if state is not None:
        trainer.restore(state)
        print(f"Training Phase_{phase} restored from {state}")

    print(f"Training Phase_{phase} started with {passed_ts} time steps.")
    while passed_ts < int(total_ts * phase / total_phase):
        result = trainer.train()

        rollout_ts = result["timesteps_total"]
        passed_ts += rollout_ts

        # result["phase"] = phase
        # result["timesteps_total"] += prev_phase_time
        # reporter(**result)
        print(f"Episode finished with {rollout_ts} time steps.")

    state = trainer.save()
    trainer.stop()
    print(
        f"Training Phase_{phase} complete with {passed_ts} time steps. Trainer is saved at {state}"
    )
    return (state, passed_ts, reporter)


def curriculum_training(config, reporter):
    exp_name = config["env_config"].get("exp_name", "curriculum_training")
    total_ts = config["env_config"].get("stop_timesteps", TIMESTEP)
    passed_ts = 0
    total_phase = 3

    # =============================== Phase 1 =============================== #

    phase = 1
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
    env_config["target"].update({"trigger_dist": env_config["success_threshhold"]})
    config.update({"env_config": env_config})
    state, passed_ts, reporter = train_fn(
        config=config,
        reporter=reporter,
        phase=phase,
        total_phase=total_phase,
        passed_ts=passed_ts,
        total_ts=total_ts,
        exp_name=exp_name,
        state=None,
    )

    # =============================== Phase 2 =============================== #
    phase = 2
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
    env_config["target"].update({"trigger_dist": env_config["success_threshhold"]})
    config.update({"env_config": env_config})
    state, passed_ts, reporter = train_fn(
        config=config,
        reporter=reporter,
        phase=phase,
        passed_ts=passed_ts,
        total_phase=total_phase,
        total_ts=total_ts,
        exp_name=exp_name,
        state=state,
    )
    # =============================== Phase 3 =============================== #
    phase = 3
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
    env_config["target"].update({"trigger_dist": env_config["success_threshhold"]})
    config.update({"env_config": env_config})
    state, passed_ts, reporter = train_fn(
        config=config,
        reporter=reporter,
        phase=phase,
        total_phase=total_phase,
        passed_ts=passed_ts,
        total_ts=total_ts,
        exp_name=exp_name,
        state=state,
    )

    # =================================================================================


if __name__ == "__main__":
    env_name = ENV.__name__
    agent_name = AGENT_NAME
    exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init()

    register_env(env_name, env_creator)
    trigger_dist = 5
    env_config = {
        "seed": 123,
        "DBG": False,
        "stop_timesteps": args.stop_timesteps,  # smuggle this parmaeter to curriculum_training fn
        "exp_name": exp_name,  # smuggle this parmaeter to curriculum_training fn
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
        "target": {
            "trigger_dist": trigger_dist,
        },
        "reward_weights": np.array([100, 1.0, 0.0]),  # success, tracking, action
        "tracking_reward_weights": np.array(
            [0.4, 0.3, 0.2, 0.1]
        ),  # z_diff, planar_dist, yaw_diff, vel_diff
        "success_threshhold": trigger_dist,  # [meters]
        "enable_residual_ctrl": True,
        "reward_scale": 0.05,
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
            "clip_param": 0.2,
            "vf_clip_param": 10,
            "grad_clip": 1.0,
            "observation_filter": "NoFilter",
            "batch_mode": "truncate_episodes",
        }
    )

    print(config)
    results = tune.run(
        curriculum_training,
        resources_per_trial=PPOTrainer.default_resource_request(config),
        name=exp_name,
        config=config,
        restore=restore,
        resume=args.resume,
        verbose=0,
    )
    ray.shutdown()
