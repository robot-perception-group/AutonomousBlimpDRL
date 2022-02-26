import os
import pickle

import numpy as np
import ray
import sys
import rl.rllib_script.agent.model.ray_model
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation, spawn_simulation_on_different_port
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from blimp_env.envs.common.gazebo_connection import GazeboConnection

checkpoint_path = os.path.expanduser(
    "~/catkin_ws/src/AutonomousBlimpDRL/RL/rl/trained_model/PPO_ResidualPlanarNavigateEnv_9d24f_00000_0_2022-02-21_17-09-14/checkpoint_001080/checkpoint-1080"
)

robot_id = "0"
auto_start_simulation = True  # start simulation
duration = int(0.5 * 3600 * 10 * 7) + 24193600
# duration = 1e20
run_pid = False


num_workers = 7

real_experiment = True  # no reset
evaluation_mode = False  # fix robotid, don't support multiworker
online_training = False  # if training during test


traj = "square" if sys.argv[1] == 0 else "coil"
windspeed = 0.5 * float(sys.argv[2])

trigger_dist = 7
init_alt = 100

###########################################

ENV = ResidualPlanarNavigateEnv

run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

if run_pid:
    beta = 0.0
    disable_servo = True
else:
    beta = 0.5
    disable_servo = False


env_config = config["env_config"]
env_config.update(
    {
        "robot_id": robot_id,
        "DBG": False,
        "evaluation_mode": evaluation_mode,
        "real_experiment": real_experiment,
        "seed": 123,
        "duration": duration,
        "beta": beta,
        "success_threshhold": trigger_dist,  # [meters]
    }
)
env_config["simulation"].update(
    {
        "robot_id": int(robot_id),
        "gui": False,
        "auto_start_simulation": auto_start_simulation,
        "enable_meshes": True,
        "enable_wind": True,
        "enable_wind_sampling": True,
        "wind_speed": windspeed,
        "wind_direction": (1, 0),
        "enable_buoyancy_sampling": False,
        "position": (0, 0, init_alt),
    }
)
if "observation" in env_config:
    env_config["observation"].update(
        {
            "noise_stdv": 0.1,
        }
    )
else:
    env_config["observation"] = {
        "noise_stdv": 0.1,
    }

if "action" in env_config:
    env_config["action"].update(
        {
            "act_noise_stdv": 0.25,
            "disable_servo": disable_servo,
            # "max_servo": -0.5,
            "max_thrust": 0.5,
        }
    )
else:
    env_config["action"] = {
        "act_noise_stdv": 0.25,
        "disable_servo": disable_servo,
        # "max_servo": -0.5,
        "max_thrust": 0.7,
    }


def generate_coil(points, radius, speed=5):
    li = []
    nwp_layer = 8
    for i in range(points):
        x = radius * np.sin(i * 2 * np.pi / nwp_layer)
        y = radius * np.cos(i * 2 * np.pi / nwp_layer)
        wp = (x, y, -init_alt - 2 * i, speed)
        li.append(wp)
    return li


coil = generate_coil(8 * 2 - 1, 30)
square = [
    (40, 40, -init_alt, 3),
    (40, -40, -init_alt, 3),
    (-40, -40, -init_alt, 3),
    (-40, 40, -init_alt, 3),
]

if traj == "coil":
    wp_list = coil
elif traj == "square":
    wp_list = square
target_dict = {
    "type": "MultiGoal",  # InteractiveGoal
    "target_name_space": "goal_",
    "trigger_dist": trigger_dist,
    "wp_list": wp_list,
    "enable_random_goal": False,
}
if "target" in env_config:
    env_config["target"].update(target_dict)
else:
    env_config["target"] = target_dict


if online_training:
    config.update(
        {
            "create_env_on_driver": False,
            "num_workers": num_workers,
            "num_gpus": 1,
            "explore": False,
            "env_config": env_config,
            "horizon": 400,
            "rollout_fragment_length": 400,
            "train_batch_size": 5600,
            "sgd_minibatch_size": 512,
            "lr": 5e-4,
            "lr_schedule": None,
            "num_sgd_iter": 16,
        }
    )
else:
    config.update(
        {
            "create_env_on_driver": False,
            "num_workers": num_workers,
            "num_gpus": 1,
            "explore": False,
            "env_config": env_config,
            "horizon": 400,
            "rollout_fragment_length": 400,
            "train_batch_size": 5600,
            "sgd_minibatch_size": 512,
            "lr": 0,
            "lr_schedule": None,
            "num_sgd_iter": 0,
        }
    )

print(config)
ray.shutdown()
ray.init()
agent = ppo.PPOTrainer(config=config, env=ENV)
agent.restore(checkpoint_path)
for _ in range(int(duration)):
    result = agent.train()
    print(pretty_print(result))
    if result["timesteps_total"] >= duration:
        break
print("done")


# else:
#     config.update(
#         {
#             "create_env_on_driver": False,  # Make sure worker 0 has an Env.
#             "num_workers": num_workers,
#             "num_gpus": 1,
#             "horizon": duration,
#             "rollout_fragment_length": duration,
#             "explore": False,
#             "env_config": env_config,
#         }
#     )
#     print(config)
#     ray.shutdown()
#     ray.init()  # local_mode: single thread
#     agent = ppo.PPOTrainer(config=config, env=ENV)
#     agent.restore(checkpoint_path)

#     n_steps = int(duration)
#     total_reward = 0
#     cell_size = config["model"]["custom_model_config"].get("lstm_cell_size", 64)
#     state = [np.zeros(cell_size, np.float32), np.zeros(cell_size, np.float32)]
#     prev_action = np.zeros(4)
#     prev_reward = np.zeros(1)

#     gaz = GazeboConnection()
#     gaz.pause_sim()
#     gaz.reset_sim()
#     gaz.unpause_sim()
#     for steps in range(n_steps):
#         action, state, _ = agent.compute_single_action(
#             obs,
#             state=state,
#             prev_action=prev_action,
#             prev_reward=prev_reward,
#         )
#         obs, reward, done, info = env.step(action)
#         prev_action = action
#         prev_reward = reward
#         total_reward += reward

#         if steps % 20 == 0:
#             print(
#                 f"Steps #{steps} Total Reward: {total_reward}, Average Reward: {total_reward/(steps+1)}"
#             )
#             print(f"Steps #{steps} Action: {action}")
#             print(f"Steps #{steps} Observation: {obs}")
