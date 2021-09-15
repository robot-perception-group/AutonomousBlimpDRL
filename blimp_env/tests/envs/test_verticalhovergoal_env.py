import pytest
from blimp_env.envs import VerticalHoverGoalEnv
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from stable_baselines3.common.env_checker import check_env
import copy
import numpy as np
import rospy
import time

ENV = VerticalHoverGoalEnv

VertiHoverGoalEnv_kwargs = {
    "DBG": True,
    "observation": {
        "type": "KinematicsGoal",
        "name_space": "machine_",
        "orientation_type": "euler",
        "action_feedback": True,
        "goal_obs_diff_feedback": True,
        "real_experiment": False,
        "DBG_ROS": False,
        "DBG_OBS": False,
    },
    "action": {
        "type": "DiscreteMetaHoverAction",
        "name_space": "machine_",
        "flightmode": 3,
        "DBG_ACT": False,
    },
    "target": {
        "type": "GOAL",
        "target_name_space": "goal_",
        "orientation_type": "euler",
        "DBG_ROS": False,
    },
}
SimpleContinuousAction_kwargs = {
    "DBG": True,
    "action": {
        "type": "SimpleContinuousAction",
    },
}
DiscreteMetaAction_kwargs = {
    "DBG": True,
    "action": {
        "type": "DiscreteMetaAction",
    },
}
DiscreteMetaHoverAction_kwargs = {
    "DBG": True,
    "action": {
        "type": "DiscreteMetaHoverAction",
    },
}
SimpleDiscreteMetaHoverAction_kwargs = {
    "DBG": True,
    "action": {
        "type": "SimpleDiscreteMetaHoverAction",
    },
}


KinematicsGoal_kwargs = [
    {},
    VertiHoverGoalEnv_kwargs,
    SimpleContinuousAction_kwargs,
    DiscreteMetaAction_kwargs,
    DiscreteMetaHoverAction_kwargs,
    SimpleDiscreteMetaHoverAction_kwargs,
]

env_kwargs = KinematicsGoal_kwargs
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
    assert rew >= 0 and rew <= 1
    assert isinstance(terminal, bool)


@pytest.mark.parametrize("env_kwargs", env_kwargs1)
def test_compute_reward(env_kwargs):
    env = ENV(env_kwargs)
    GazeboConnection().unpause_sim()

    achieved_goal = np.ones(6)
    desired_goal = achieved_goal
    result = env.compute_reward(achieved_goal, desired_goal, {})
    expect = 1
    np.testing.assert_allclose(result, expect)

    achieved_goal = np.zeros(6)
    desired_goal = achieved_goal
    result = env.compute_reward(achieved_goal, desired_goal, {})
    expect = 1
    np.testing.assert_allclose(result, expect)

    achieved_goal = np.ones(6)
    desired_goal = -1 * np.ones(6)
    result = env.compute_reward(achieved_goal, desired_goal, {})
    expect = 0.001
    np.testing.assert_array_less(result, expect)

    achieved_goal = np.zeros((25, 6))
    desired_goal = np.zeros((25, 6))
    result = env.compute_reward(achieved_goal, desired_goal, {})
    expect = np.ones((25,))
    assert result.shape == (25,)
    np.testing.assert_allclose(result, expect)

    for _ in range(10):
        achieved_goal = np.random.uniform(-1, 1, 6)
        desired_goal = np.random.uniform(-1, 1, 6)
        rew = env.compute_reward(achieved_goal, desired_goal, {})
        assert isinstance(rew, float)
        assert rew >= 0 and rew <= 1


@pytest.mark.parametrize("env_kwargs", env_kwargs2)
def test_is_success(env_kwargs):
    env = ENV(env_kwargs)
    GazeboConnection().unpause_sim()

    achieved_goal = np.ones(6)
    desired_goal = achieved_goal
    result = env._is_success(achieved_goal, desired_goal, {})
    assert result == True

    achieved_goal = np.zeros(6)
    desired_goal = achieved_goal
    result = env._is_success(achieved_goal, desired_goal, {})
    assert result == True

    achieved_goal = np.zeros(6)
    desired_goal = np.ones(6)
    result = env._is_success(achieved_goal, desired_goal, {})
    assert result == False


@pytest.mark.parametrize("env_kwargs", env_kwargs3)
def test_far_from_goal(env_kwargs):
    env = ENV(env_kwargs)
    GazeboConnection().unpause_sim()

    result = env.far_from_goal(np.zeros(3), np.zeros(3))
    assert result == False

    result = env.far_from_goal(np.zeros(3), np.ones(3) * 0.5)
    assert result == False

    result = env.far_from_goal(np.zeros(3) - 0.001, np.ones(3))
    assert result == True

    result = env.far_from_goal(np.ones(3), -np.ones(3))
    assert result == True


def exp_env_step(env_kwargs):
    from blimp_env.envs.script import spawn_simulation

    spawn_simulation(1, gui=False, task=ENV.default_config()["task"])

    env = ENV(env_kwargs)
    env.reset()
    while not rospy.is_shutdown():
        # action = np.ones(env.action_type.act_dim) * 0
        action = 1
        obs, _, _, _ = env.step(action)

    GazeboConnection().unpause_sim()


# from blimp_env.envs.script import spawn_simulation

# spawn_simulation(1, gui=False, task=ENV.default_config()["task"])

# exp_env_step(env_kwargs[1])
