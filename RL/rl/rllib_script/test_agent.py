import pickle
import os
import numpy as np
from blimp_env.envs import ResidualPlanarNavigateEnv
import ray
from ray.rllib.agents import ppo
import rl.rllib_script.agent.model

from blimp_env.envs.script import close_simulation, spawn_simulation_on_different_port


checkpoint_path = os.path.expanduser(
    "~/ray_results/ResidualPlanarNavigateEnv_PPO_wind_LSTM_absMix/PPO_ResidualPlanarNavigateEnv_b32cf_00000_0_2022-01-14_20-10-56/checkpoint_002700/checkpoint-2700"
)

ENV = ResidualPlanarNavigateEnv
simulation_mode = True  # False if realworld exp
auto_start_simulation = True  # False if realworld exp or reusing env
duration = 1e20

run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

env_config = config["env_config"]
env_config.update(
    {
        "DBG": False,
        "evaluation_mode": True,
        "seed": 123,
        "duration": duration,
        "beta": 0.5,
    }
)
env_config["simulation"].update(
    {
        "gui": True,
        "auto_start_simulation": auto_start_simulation,
        "enable_meshes": True,
        "enable_wind": False,
        "enable_wind_sampling": False,
        "wind_speed": 1.5,
        "wind_direction": (1, 0),
        "enable_buoyancy_sampling": False,
        "position": (0, 0, 30),
    }
)
env_config["observation"].update(
    {
        "real_experiment": not simulation_mode,
        "noise_stdv": 0.0 if not simulation_mode else 0.02,
    }
)
env_config["action"].update(
    {
        "act_noise_stdv": 0.0 if not simulation_mode else 0.05,
        "disable_servo": True,
        "max_servo": -0.5,
        "max_thrust": 0.5,
    }
)
env_config["target"].update(
    {
        "type": "MultiGoal",  # InteractiveGoal
        "trigger_dist": 10,
        # "new_target_every_ts": 1200,
    }
)

if auto_start_simulation:
    close_simulation()
    spawn_simulation_on_different_port(**env_config["simulation"])

config.update(
    {
        "create_env_on_driver": True,  # Make sure worker 0 has an Env.
        "num_workers": 0,
        "num_gpus": 0,
        "rollout_fragment_length": duration,
        "explore": False,
    }
)
print(config)
ray.shutdown()
ray.init(local_mode=True)  # local_mode: single thread
agent = ppo.PPOTrainer(config=config, env=ENV)
agent.restore(checkpoint_path)

n_steps = int(duration)
total_reward = 0
cell_size = config["model"]["custom_model_config"]["lstm_cell_size"]
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

    if steps % 100 == 0:
        print(f"Steps #{steps} Average Reward: {total_reward/(steps+1)}")
