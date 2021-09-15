import gym
import pytest
from blimp_env.envs.script import spawn_simulation_for_testing
from blimp_env.envs.common.gazebo_connection import GazeboConnection

envs = [
    "blimp_env:navigate-v0",
    "blimp_env:navigate_goal-v0",
    "blimp_env:hover_goal-v0",
    "blimp_env:vertical_goal-v0",
    "blimp_env:vertical_goal_2act-v0",
    "blimp_env:planar_navigate-v0",
]


@pytest.mark.parametrize("env_spec", envs)
def test_env_step(env_spec):
    env = gym.make(env_spec)

    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)

    GazeboConnection().unpause_sim()

    assert env.observation_space.contains(obs)


# spawn_simulation_for_testing(1, gui=True)
