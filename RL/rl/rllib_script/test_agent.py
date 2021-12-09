import pickle
import os
import numpy as np
from blimp_env.envs import ResidualPlanarNavigateEnv
import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import sample_from
from ray.tune.registry import register_env
from rl.rllib_script.agent.model import TorchBatchNormModel

checkpoint_path = os.path.expanduser(
    "~/ray_results/ResidualPlanarNavigateEnv_PPO_test/PPO_ResidualPlanarNavigateEnv_30925_00000_0_clip_param=0.15017,lambda=0.91278,lr=4.4654e-05_2021-12-03_18-41-05/checkpoint_000864/checkpoint-864"
)
run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

config["evaluation_num_workers"] = 1
config["evaluation_interval"] = 1  # <-- HERE: must set this to > 0!
config["num_workers"] = 1

ray.init()

ENV = ResidualPlanarNavigateEnv
env_config = {
    "seed": 123,
    "simulation": {
        "gui": True,
        "auto_start_simulation": False,
    },
    "action": {
        "disable_servo": True,
    },
    "reward_weights": np.array([100, 0.9, 0.0, 0.1]),
}


agent = ppo.PPOTrainer(config=config, env=ENV)
agent.restore(checkpoint_path)

env = ENV(env_config)

episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
