from blimp_env.envs import (
    # VerticalHoverGoal2ActEnv,
    # VerticalHoverGoalEnv,
    PlanarNavigateEnv,
    # PlanarNavigateEnv_v2,
    RealWorldPlanarNavigateEnv,
)
from blimp_env.envs.common.gazebo_connection import GazeboConnection

from blimp_env.envs.script import close_simulation
from sb3_contrib import QRDQN


ENV = PlanarNavigateEnv
AGENT = QRDQN

# exp
time_steps = 100000

# model
model_path = "."

# environments"""
env_kwargs = {
    "DBG": True,
    "seed": 123,
    "real_experiment": False,
    "simulation": {
        "robot_id": str(0),
        "ros_port": 11311,
        "gaz_port": 11351,
        "gui": True,
        "enable_meshes": True,
        "world": "basic",
        "task": "square",  # hover_fixed_goal
        "auto_start_simulation": True,
        "update_robotID_on_workerID": True,
    },
    "observation": {
        "DBG_OBS": True,
    },
    "success_threshhold": 0.1,
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


play(env_kwargs=env_kwargs, model_path=model_path, time_steps=time_steps)
