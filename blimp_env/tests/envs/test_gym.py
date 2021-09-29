import gym
import pytest
from blimp_env.envs import (
    NavigateEnv,
    NavigateGoalEnv,
    HoverGoalEnv,
    VerticalHoverGoalEnv,
    VerticalHoverGoal2ActEnv,
    PlanarNavigateEnv,
)

from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.script import spawn_simulation_for_testing, close_simulation

env_kwargs = {
    "simulation": {
        "gui": True,
        "enable_meshes": True,
        "auto_start_simulation": False,
    },
}

envs = [
    NavigateEnv,
    NavigateGoalEnv,
    HoverGoalEnv,
    VerticalHoverGoalEnv,
    VerticalHoverGoal2ActEnv,
    PlanarNavigateEnv,
]


@pytest.mark.parametrize("env", envs)
def test_env_step(env):
    env = env(env_kwargs)
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)

    GazeboConnection().unpause_sim()

    assert env.observation_space.contains(obs)


close_simulation()
spawn_simulation_for_testing(1, gui=True)
