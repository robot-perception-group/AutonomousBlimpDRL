import os

import pickle5 as pickle

# import pickle
import json
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.planar_navigate_env import Observation
from blimp_env.envs.script import close_simulation, spawn_simulation_on_different_port
import numpy as np
from rl.rllib_script.agent.model.tx2_model import TorchBatchNormRNNModel
from rl.rllib_script.agent.model.action_dist import TorchDiagGaussian
from rl.rllib_script.agent.torch_policy import MyTorchPolicy, ppo_surrogate_loss

robot_id = "1"
simulation_mode = False  # if realworld exp or simulation
online_training = False  # if training during test, currently disabled because cannot install ray on tx2
duration = 1e20
train_iter = 1e20
run_pid = False

checkpoint_path = os.path.expanduser(
    "~/src/AutonomousBlimpDRL/RL/rl/trained_model/PPO_ResidualPlanarNavigateEnv_9d24f_00000_0_2022-02-21_17-09-14/checkpoint_001080/"
)


trigger_dist = 7
wp_list = [
    (20, 20, -15, 3),
    (20, -20, -15, 3),
    (-20, -20, -15, 3),
    (-20, 20, -15, 3),
]
###########################################

ENV = ResidualPlanarNavigateEnv
dist_cls = TorchDiagGaussian

checkpoint_base_dir = os.path.dirname(checkpoint_path)

policy_path = os.path.join(checkpoint_base_dir, "mypolicy.pickle")
with open(policy_path, "rb") as f:
    mypolicy = pickle.load(f)

myconfig_path = os.path.join(checkpoint_base_dir, "myconfig.pickle")
with open(myconfig_path, "rb") as fh:
    config = pickle.load(fh)

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
        "evaluation_mode": True,
        "real_experiment": not simulation_mode,
        "seed": 123,
        "duration": duration,
        "beta": beta,
        # "reward_weights": np.array([100, 0.9, 0.1, 0.1]),
        "success_threshhold": trigger_dist,  # [meters]
    }
)
env_config["simulation"].update(
    {
        "robot_id": int(robot_id),
    }
)

obs_dict = {
    "noise_stdv": 0.0 if not simulation_mode else 0.02,
}
if "observation" in env_config:
    env_config["observation"].update(obs_dict)
else:
    env_config["observation"] = obs_dict

act_dict = {
    "act_noise_stdv": 0.0 if not simulation_mode else 0.0,
    "disable_servo": disable_servo,
    # "max_servo": -0.5,
    "max_thrust": 0.35,
}
if "action" in env_config:
    env_config["action"].update(act_dict)
else:
    env_config["action"] = act_dict

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


###########################################

weights = mypolicy["weights"]
observation_space = mypolicy["observation_space"]
action_space = mypolicy["action_space"]

num_outputs = dist_cls.required_model_output_shape(action_space, config)
model_config = config["model"]
name = config["model"]["custom_model"]

model = TorchBatchNormRNNModel(
    obs_space=observation_space,
    action_space=action_space,
    num_outputs=num_outputs,
    model_config=model_config,
    name=name,
)

loss = ppo_surrogate_loss
action_distribution_class = dist_cls
action_sampler_fn = None
action_distribution_fn = None
max_seq_len = 20
get_batch_divisibility_req = None

policy = MyTorchPolicy(
    observation_space=observation_space,
    action_space=action_space,
    config=config,
    model=model,
    loss=loss,
    action_distribution_class=action_distribution_class,
    action_sampler_fn=action_sampler_fn,
    action_distribution_fn=action_distribution_fn,
    max_seq_len=max_seq_len,
    get_batch_divisibility_req=get_batch_divisibility_req,
)
policy.set_weights(weights)

###########################################

env_config["simulation"]["auto_start_simulation"] = False

n_steps = int(duration)
total_reward = 0
cell_size = config["model"]["custom_model_config"].get("lstm_cell_size", 64)
state = [np.zeros(cell_size, np.float32), np.zeros(cell_size, np.float32)]
prev_action = np.zeros(4)
prev_reward = np.zeros(1)
env = ENV(env_config)

obs = env.reset()
for steps in range(n_steps):
    action, state, _ = policy.compute_single_action(
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
        print(
            f"Steps #{steps} Total Reward: {total_reward}, Average Reward: {total_reward/(steps+1)}"
        )
        print(f"Steps #{steps} Action: {action}")
        print(f"Steps #{steps} Observation: {obs}")
