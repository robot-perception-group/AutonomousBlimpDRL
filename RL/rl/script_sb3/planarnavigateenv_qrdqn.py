import datetime
import os

from blimp_env.envs import PlanarNavigateEnv
from rl.agent.qrdqn_policy import MyQRDQNPolicy
from rl.script.config import generate_config, save_config
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

# exp setup
ENV = PlanarNavigateEnv
AGENT = QRDQN
robot_id = 0
exp_name_posfix = "TrackRew"

env_name = ENV.__name__
agent_name = AGENT.__name__
exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix

POLICY = MyQRDQNPolicy
sec = 2
hour = 3600 * sec
day = 24 * hour
TIMESTEP = 7 * day

gui = True
close_prev_sim = True if robot_id == 0 else False


def exp_training(exp_config, env_config, agent_config):
    model = AGENT(
        policy=POLICY,
        env=ENV(env_config),
        **agent_config,
    )
    model.learn(**exp_config["learn"])
    model.save(exp_config["final_model_save_path"])
    model.save_replay_buffer(exp_config["final_replaybuffer_save_path"])


# exp_config
exp_time = datetime.datetime.now().strftime("%c")
exp_path = os.path.join("./logs", exp_name, exp_time)
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
close_prev_sim = True if robot_id == 0 else False
env_config = {
    "simulation": {
        "robot_id": str(robot_id),
        "ros_port": 11311,
        "gaz_port": 11351,
        "gui": True,
        "world": "basic",
        "task": "navigate_goal",
        "close_prev_sim": close_prev_sim,
        "auto_start_simulation": True,
    },
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
