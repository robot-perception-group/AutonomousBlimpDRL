import datetime
import os

from blimp_env.envs import VerticalHoverGoalEnv
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.script import spawn_simulation
from rl.agent.qrdqn_policy import MyMultiInputPolicy
from rl.script.config import generate_config, save_config
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

ENV = VerticalHoverGoalEnv
AGENT = QRDQN

task = ENV.default_config()["task"]
env_name = ENV.__name__
agent_name = AGENT.__name__
exp_name = env_name + "_" + agent_name

POLICY = MyMultiInputPolicy
sec = 2
hour = 3600 * sec
day = 24 * hour
TIMESTEP = 0.5 * day


def exp_training(exp_kwargs, env_kwargs, agent_kwargs):
    spawn_simulation(n_env=1, gui=False, task=task)

    model = AGENT(
        policy=POLICY,
        env=ENV(env_kwargs),
        **agent_kwargs,
    )
    model.learn(**exp_kwargs["learn"])
    model.save(exp_kwargs["final_model_save_path"])
    model.save_replay_buffer(
        exp_kwargs["final_replaybuffer_save_path"]
    )  # call load_replay_buffer() for loading
    # GazeboConnection().unpause_sim()


# exp_kwargs
exp_time = datetime.datetime.now().strftime("%c")
exp_path = os.path.join("./logs", exp_name, exp_time)
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=os.path.join(exp_path, "checkpoint"),
    name_prefix=agent_name + "_model",
)
learn_kwargs = {
    "total_timesteps": TIMESTEP,
    "callback": CallbackList([checkpoint_callback]),
    "eval_freq": 10000,
    "n_eval_episodes": 3,
    "eval_log_path": os.path.join(exp_path, "eval"),
}

exp_kwargs = {
    "logdir": exp_path,
    "learn": learn_kwargs,
    "final_model_save_path": os.path.join(exp_path, "final_model"),
    "final_replaybuffer_save_path": os.path.join(exp_path, "final_buffer"),
}

# env_kwargs
env_kwargs = {
    "observation": {
        "type": "KinematicsGoal",
        "name_space": "machine_",
        "orientation_type": "euler",
        "action_feedback": True,
        "goal_obs_diff_feedback": True,
    },
}

# agent_kwargs
agent_kwargs = {
    "learning_rate": 1e-4,
    "learning_starts": 3e4,
    "batch_size": 256,
    "tau": 0.95,
    "gamma": 0.999,
    "target_update_interval": 5e2,
    "exploration_fraction": 5e-3,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"n_quantiles": 7, "net_arch": [64, 64]},
    "tensorboard_log": os.path.join(exp_path, "tb"),
}

# start training
meta_kwargs = generate_config(exp_kwargs, env_kwargs, agent_kwargs)
print(meta_kwargs)
save_config(exp_path, meta_kwargs)
exp_training(**meta_kwargs)
