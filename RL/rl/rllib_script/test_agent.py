import pickle
import os
import numpy as np
from blimp_env.envs import ResidualPlanarNavigateEnv
import ray
from ray.rllib.agents import ppo
from ray.rllib.evaluate import rollout
import rl.rllib_script.agent.model

from blimp_env.envs.script import close_simulation, spawn_simulation_on_different_port


checkpoint_path = os.path.expanduser(
    "~/ray_results/ResidualPlanarNavigateEnv_PPO_wind_LSTM_absMix/PPO_ResidualPlanarNavigateEnv_b32cf_00000_0_2022-01-14_20-10-56/checkpoint_002700/checkpoint-2700"
)

ENV = ResidualPlanarNavigateEnv


run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

auto_start_simulation = False
env_config = config["env_config"]
env_config.update(
    {
        "DBG": False,
        "evaluation_mode": True,
        "seed": 123,
        "robot_id": "0",
        "duration": 1000000000,
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
        "position": (0, 0, 30),
    }
)
env_config.update(
    {
        "target": {
            "type": "MultiGoal",  # InteractiveGoal
            "trigger_dist": 10,
            # "new_target_every_ts": 1200,
        },
    }
)
if auto_start_simulation:
    close_simulation()
    spawn_simulation_on_different_port(**env_config)

env_config["simulation"]["auto_start_simulation"] = False
config.update(
    {
        "create_env_on_driver": True,  # Make sure worker 0 has an Env.
        "num_workers": 0,
        "num_gpus": 0,
        # "evaluation_num_workers": 1,
        # "evaluation_interval": 1,
        # "evaluation_num_episodes": 1,
        # "evaluation_config": {
        #     "explore": False,
        #     "env_config": env_config,
        # },
    }
)
ray.shutdown()
ray.init(local_mode=True)
agent = ppo.PPOTrainer(config=config, env=ENV)
agent.restore(checkpoint_path)

num_episodes = 10
steps = 0
episodes = 0
for episodes in range(num_episodes):
    eval_result = agent.evaluate()["evaluation"]
    print(
        "Episode #{}: reward: {}".format(episodes, eval_result["episode_reward_mean"])
    )

# agent.stop()
