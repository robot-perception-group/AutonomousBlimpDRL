import pytest
from blimp_env.envs import PlanarNavigateEnv2
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from stable_baselines3.common.env_checker import check_env
import copy
import numpy as np
import rospy

ENV = PlanarNavigateEnv2

PlanarNavigateEnv_kwargs = {
    "DBG": True,
    "simulation": {
        "gui": True,
        "enable_meshes": True,
        "auto_start_simulation": False,
    },
    "observation": {
        "type": "PlanarKinematics",
        "DBG_ROS": False,
        "DBG_OBS": True,
    },
}

env_kwargs = [PlanarNavigateEnv_kwargs]

env_kwargs0 = copy.deepcopy(env_kwargs)
env_kwargs1 = copy.deepcopy(env_kwargs)
env_kwargs2 = copy.deepcopy(env_kwargs)
env_kwargs3 = copy.deepcopy(env_kwargs)
env_kwargs4 = copy.deepcopy(env_kwargs)


@pytest.mark.parametrize("env_kwargs", env_kwargs)
def test_env_functions(env_kwargs):
    check_env(ENV(env_kwargs), warn=True)
    GazeboConnection().unpause_sim()


@pytest.mark.parametrize("env_kwargs", env_kwargs0)
def test_env_step(env_kwargs):
    env = ENV(env_kwargs)
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, rew, terminal, _ = env.step(action)

    GazeboConnection().unpause_sim()

    assert env.observation_space.contains(obs)
    assert rew >= -2 and rew <= 1
    assert isinstance(terminal, bool)


def test_compute_success_rew():
    env = ENV

    achieved_goal = np.ones(3)
    desired_goal = achieved_goal
    result = env.compute_success_rew(achieved_goal, desired_goal)
    expect = 1
    np.testing.assert_allclose(result, expect)

    achieved_goal = np.zeros(3)
    desired_goal = achieved_goal
    result = env.compute_success_rew(achieved_goal, desired_goal)
    expect = 1
    np.testing.assert_allclose(result, expect)

    achieved_goal = np.ones(3)
    desired_goal = -1 * np.ones(3)
    result = env.compute_success_rew(achieved_goal, desired_goal)
    expect = 0
    np.testing.assert_allclose(result, expect)

    for _ in range(10):
        achieved_goal = np.random.uniform(-1, 1, 3)
        desired_goal = np.random.uniform(-1, 1, 3)
        rew = env.compute_success_rew(achieved_goal, desired_goal)
        assert isinstance(rew, float)
        assert rew == 0.0 or rew == 1.0


def test_compute_tracking_rew():
    env = ENV

    obs_diff = -np.ones(3)
    weights = np.array([0.5, 0.4, 0.1])
    result = env.compute_tracking_rew(obs_diff, weights)
    expect = -1
    np.testing.assert_allclose(result, expect)

    obs_diff = -np.zeros(3)
    weights = np.array([0.5, 0.4, 0.1])
    result = env.compute_tracking_rew(obs_diff, weights)
    expect = 0
    np.testing.assert_allclose(result, expect)

    for _ in range(10):
        obs_diff = np.random.uniform(-1, 0, 3)
        weights = np.random.uniform(0, 1 / 3, 3)
        rew = env.compute_tracking_rew(obs_diff, weights)
        assert isinstance(rew, float)
        assert rew >= -1 and rew <= 0


def test_compute_psi_diff():
    env = ENV

    goal_pos = np.ones(3)
    obs_pos = np.zeros(3)
    obs_psi = 0
    result = env.compute_psi_diff(goal_pos, obs_pos, obs_psi)
    expect = 0.25
    np.testing.assert_allclose(result, expect)

    goal_pos = np.ones(3)
    obs_pos = -np.ones(3)
    obs_psi = 0.5
    result = env.compute_psi_diff(goal_pos, obs_pos, obs_psi)
    expect = -0.25
    np.testing.assert_allclose(result, expect)

    goal_pos = np.zeros(3)
    obs_pos = np.ones(3)
    obs_psi = 0
    result = env.compute_psi_diff(goal_pos, obs_pos, obs_psi)
    expect = -0.75
    np.testing.assert_allclose(result, expect)

    for _ in range(100):
        goal_pos = np.random.uniform(-1, 1, 3)
        obs_pos = np.random.uniform(-1, 1, 3)
        obs_psi = np.random.uniform(-1, 1)
        result = env.compute_psi_diff(goal_pos, obs_pos, obs_psi)
        assert isinstance(result, float)
        assert result >= -1 and result <= 1


@pytest.mark.parametrize("env_kwargs", env_kwargs4)
def test_check_publisher_connection(env_kwargs):
    env = ENV(env_kwargs)
    GazeboConnection().unpause_sim()
    connected = env.action_type.check_publishers_connection()
    assert connected == True


def exp_env_step(env_kwargs):
    from blimp_env.envs.script import close_simulation, close_simulation_on_marvin

    close_simulation()
    # close_simulation_on_marvin()
    env = ENV(env_kwargs)
    env.reset()
    while not rospy.is_shutdown():
        action = 1
        obs, _, _, _ = env.step(action)

    GazeboConnection().unpause_sim()


# exp_env_step(env_kwargs[0])
