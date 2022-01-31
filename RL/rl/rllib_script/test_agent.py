import os
import pickle

import numpy as np
import ray
import rl.rllib_script.agent.model
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation, spawn_simulation_on_different_port
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

ENV = ResidualPlanarNavigateEnv

checkpoint_path = os.path.expanduser(
    "~/catkin_ws/src/AutonomousBlimpDRL/RL/rl/trained_model/PPO_ResidualPlanarNavigateEnv_LSTM_AbsMix/checkpoint_002700/checkpoint-2700"
)

simulation_mode = False  # if realworld exp or simulation
auto_start_simulation = False  # start simulation
online_training = True  # if training during test

# in realworld exp "auto_start_simulation" should always be false
if not simulation_mode:
    auto_start_simulation = False

duration = 1e20  # evaluation time steps
train_iter = 1e20  # training iterations if online training is enabled

run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

env_config = config["env_config"]
env_config.update(
    {
        "DBG": False,
        "evaluation_mode": True,
        "real_experiment": not simulation_mode,
        "seed": 123,
        "duration": duration,
        # "beta": 0.5,
    }
)
env_config["simulation"].update(
    {
        "gui": True,
        "auto_start_simulation": auto_start_simulation,
        "enable_meshes": True,
        "enable_wind": True,
        "enable_wind_sampling": False,
        "wind_speed": 1.5,
        "wind_direction": (1, 0),
        "enable_buoyancy_sampling": False,
        "position": (0, 0, 50),
    }
)
env_config["observation"].update(
    {
        "noise_stdv": 0.0 if not simulation_mode else 0.02,
    }
)
env_config["action"].update(
    {
        "act_noise_stdv": 0.0 if not simulation_mode else 0.05,
        # "disable_servo": False,
        # "max_servo": -0.5,
        # "max_thrust": 0.5,
    }
)


wp_list = [
    (40, 40, -30, 5),
    (40, -40, -30, 5),
    (-40, -40, -30, 5),
    (-40, 40, -30, 5),
]
target_dict = {
    "type": "MultiGoal",  # InteractiveGoal
    "target_name_space": "goal_",
    "trigger_dist": 10,
    "wp_list": wp_list,
}
if "target" in env_config:
    env_config["target"].update(target_dict)
else:
    env_config["target"] = target_dict

if auto_start_simulation:
    close_simulation()
    spawn_simulation_on_different_port(**env_config["simulation"])

env_config["simulation"]["auto_start_simulation"] = False
if online_training:
    config.update(
        {
            "create_env_on_driver": False,  # Make sure worker 0 has an Env.
            "num_workers": 0,
            "num_gpus": 1,
            "explore": True,
            "env_config": env_config,
            "horizon": 400,
            "rollout_fragment_length": 400,
            "train_batch_size": 1600,
            "sgd_minibatch_size": 128,
            "lr": 1e-4,
            "lr_schedule": None,
        }
    )
    print(config)
    ray.shutdown()
    ray.init()
    agent = ppo.PPOTrainer(config=config, env=ENV)
    agent.restore(checkpoint_path)
    for _ in range(int(train_iter)):
        result = agent.train()
        print(pretty_print(result))
        if result["timesteps_total"] >= duration:
            break
else:
    config.update(
        {
            "create_env_on_driver": True,  # Make sure worker 0 has an Env.
            "num_workers": 0,
            "num_gpus": 1,
            "horizon": duration,
            "rollout_fragment_length": duration,
            "explore": False,
            "env_config": env_config,
        }
    )
    print(config)
    ray.shutdown()
    ray.init()  # local_mode: single thread
    agent = ppo.PPOTrainer(config=config, env=ENV)
    agent.restore(checkpoint_path)

    n_steps = int(duration)
    total_reward = 0
    cell_size = config["model"]["custom_model_config"].get("lstm_cell_size", 64)
    state = [np.zeros(cell_size, np.float32), np.zeros(cell_size, np.float32)]
    prev_action = np.zeros(4)
    prev_reward = np.zeros(1)
    env = agent.workers.local_worker().env

    obs = env.reset()
    for steps in range(n_steps):
        action, state, _ = agent.compute_single_action(
            obs,
            state=state,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward
        prev_action = action
        prev_reward = reward

        if steps % 20 == 0:
            print(f"Steps #{steps} Average Reward: {total_reward/(steps+1)}")
            print(f"Steps #{steps} Action: {action}")
            print(f"Steps #{steps} Observation: {obs}")
