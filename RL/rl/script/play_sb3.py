import argparse
import os
from blimp_env.envs import PlanarNavigateEnv
from blimp_env.envs.common.gazebo_connection import GazeboConnection

from blimp_env.envs.script import close_simulation
from sb3_contrib import QRDQN

# args
ENV = PlanarNavigateEnv
AGENT = QRDQN

# exp
time_steps = 100000

# model
default_model_path = (
    os.path.expanduser("~")
    + "/catkin_ws/src/AutonomousBlimpDRL/RL/rl/trained_model/final_model"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default=default_model_path, help="path to policy model"
)
parser.add_argument(
    "--time_steps",
    type=int,
    default=time_steps,
    help="time steps for testing, 0.5s/step",
)
parser.add_argument(
    "--task", type=str, default="square", help="square or hover_fixed_goal"
)
parser.add_argument(
    "--enable_wind",
    dest="enable_wind",
    action="store_true",
    help="apply wind to the blimp",
)
parser.set_defaults(enable_wind=False)
parser.add_argument(
    "--wind_speed",
    type=float,
    default=2.0,
    help="wind speed m/s",
)

args = parser.parse_args()

# environments"""
env_kwargs = {
    "DBG": True,
    "seed": 123,
    "simulation": {
        "robot_id": str(0),
        "ros_port": 11311,
        "gaz_port": 11351,
        "gui": True,
        "enable_meshes": True,
        "enable_wind": args.enable_wind,
        "wind_speed": args.wind_speed,
        "world": "basic",
        "task": args.task,
        "auto_start_simulation": True,
    },
    "observation": {
        "DBG_OBS": True,
    },
    "success_threshhold": 0.075,
}


# play
def play(env_kwargs, model_path, time_steps):
    close_simulation()
    env = ENV(env_kwargs)
    model = AGENT.load(model_path, env=env)
    print(model)

    obs = env.reset()
    for _ in range(time_steps):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

    print("play finished")
    if done:
        obs = env.reset()
        GazeboConnection().unpause_sim()


print(env_kwargs)
play(env_kwargs=env_kwargs, model_path=args.model_path, time_steps=args.time_steps)
