import pickle
import os
import numpy as np
from blimp_env.envs import ResidualPlanarNavigateEnv
import ray
from ray.rllib.agents import ppo
from ray.tune import sample_from
from ray.tune.registry import register_env
from rl.rllib_script.agent.model import TorchBatchNormModel, TorchBatchNormRNNModel


checkpoint_path = os.path.expanduser(
    "~/ray_results/ResidualPlanarNavigateEnv_PPO_wind_LSTM_absMix/PPO_ResidualPlanarNavigateEnv_b32cf_00000_0_2022-01-14_20-10-56/checkpoint_002700/checkpoint-2700"
)

ENV = ResidualPlanarNavigateEnv
env_config = {
    "DBG": True,
    "seed": 123,
    "simulation": {
        "gui": True,
        "auto_start_simulation": True,
        "enable_wind": True,
        "enable_wind_sampling": False,
        "wind_speed": 1.0,
        "wind_direction": (1, 0),
    },
    "action": {
        "disable_servo": False,
    },
    "target": {
        "type": "InteractiveGoal",
        "target_name_space": "goal_",
        "new_target_every_ts": 1200,
    },
    "reward_weights": np.array([100, 0.9, 0.1]),
}


run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

config.update(
    {
        "num_workers": 1,
        "num_gpus": 0,
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
    }
)

ray.shutdown()
ray.init()
# env = ENV(env_config)
agent = ppo.PPOTrainer(config=config, env=ENV)
agent.restore(checkpoint_path)

if config["model"]["custom_model"] == "bnrnn_model":
    cell_size = config["custom_model_config"]["lstm_cell_size"]
    state = [np.zeros(cell_size, np.float32), np.zeros(cell_size, np.float32)]


episode_reward = 0
done = False
obs = agent.env.reset()

for _ in range(10000):
    action, state, _ = agent.compute_action(obs, state)
    obs, reward, done, info = agent.env.step(action)
    episode_reward += reward
