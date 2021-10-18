"""Continue training QRDQN agent, change:
- z position target is not projected target but the real target
- action is cut to maximum 50% throttle
- reward function: 
    increase z tracking and reduce xy tracking
    increase action penalty and reduce trakincg reward
    the success threshold change from 0.06 to 0.02
"""
import datetime
import os

from blimp_env.envs import (
    PlanarNavigateEnv,
)
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.script import close_simulation
from rl.agent.qrdqn_policy import MyMultiInputPolicy, MyQRDQNPolicy
from rl.script.config import generate_config, save_config
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import numpy as np

ENV = PlanarNavigateEnv
AGENT = QRDQN
robot_id = 0
exp_name_posfix = "continue"

task = ENV.default_config()["simulation"]["task"]
env_name = ENV.__name__
agent_name = AGENT.__name__
exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix

POLICY = MyQRDQNPolicy
sec = 2
hour = 3600 * sec
day = 24 * hour
TIMESTEP = 0.5 * day

path = "."
model_path = os.path.join(path, "final_model.zip")
buffer_path = os.path.join(path, "final_buffer.pkl")


def exp_training(exp_config, env_config, agent_config):
    close_simulation()
    env = ENV(env_config)
    model = AGENT.load(model_path, env=env)
    model.load_replay_buffer(buffer_path)
    print(model.replay_buffer.size())

    model.learn(**exp_config["learn"])
    model.save(exp_config["final_model_save_path"])
    model.save_replay_buffer(exp_config["final_replaybuffer_save_path"])


# exp_config
exp_time = datetime.datetime.now().strftime("%c")
exp_path = os.path.join("../logs", exp_name, exp_time)
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=os.path.join(exp_path, "checkpoint"),
    name_prefix=agent_name + "_model",
)
learn_config = {
    "total_timesteps": TIMESTEP,
    "callback": CallbackList([checkpoint_callback]),
    "eval_freq": 10000,
    "n_eval_episodes": 3,
    "eval_log_path": os.path.join(exp_path, "eval"),
}

exp_config = {
    "logdir": exp_path,
    "learn": learn_config,
    "final_model_save_path": os.path.join(exp_path, "final_model"),
    "final_replaybuffer_save_path": os.path.join(exp_path, "final_buffer"),
}

# env_config
robot_id = 0
env_config = {
    "simulation": {
        "gui": True,
    },
    "duration": 400,
    "reward_weights": np.array([1, 0.7, 0.3]),  # success, tracking, action
    "tracking_reward_weights": np.array(
        [0.2, 0.6, 0.2]
    ),  # z_diff, planar_dist, psi_diff
    "reward_scale": (1, 1),
    "success_threshhold": 0.05,
}

# agent_config
agent_config = {
    "learning_rate": 1e-4,
    "learning_starts": 3e4,
    "batch_size": 256,
    "tau": 0.95,
    "gamma": 0.999,
    "target_update_interval": 5e2,
    "exploration_fraction": 0.1,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"n_quantiles": 13, "net_arch": [64, 64]},
    "tensorboard_log": os.path.join(exp_path, "tb"),
}

# start training
config = generate_config(
    agent_name=agent_name,
    exp_config=exp_config,
    env_config=env_config,
    agent_config=agent_config,
)
print(config)
save_config(exp_path, config)
exp_training(**config)
