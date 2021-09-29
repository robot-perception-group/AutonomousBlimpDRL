import pytest
from blimp_env.envs import NavigateEnv
from stable_baselines3.common.env_checker import check_env
from blimp_env.envs.common.gazebo_connection import GazeboConnection
import copy
import rospy
import numpy as np

ENV = NavigateEnv

NavigateEnv_kwargs = {
    "DBG": True,
    "simulation": {
        "auto_start_simulation": False,
    },
    "action": {
        "type": "ContinuousAction",
    },
}
SimpleContinuousAction_kwargs = {
    "simulation": {
        "auto_start_simulation": False,
    },
    "action": {
        "type": "SimpleContinuousAction",
    },
}
DiscreteMetaAction_kwargs = {
    "simulation": {
        "auto_start_simulation": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
}
DiscreteMetaHoverAction_kwargs = {
    "simulation": {
        "auto_start_simulation": False,
    },
    "action": {
        "type": "DiscreteMetaHoverAction",
    },
}


env_kwargs = [
    {},
    NavigateEnv_kwargs,
    SimpleContinuousAction_kwargs,
    DiscreteMetaAction_kwargs,
    DiscreteMetaHoverAction_kwargs,
]


@pytest.mark.parametrize("env_kwargs", env_kwargs)
def test_NavigateEnv(env_kwargs):
    check_env(ENV(env_kwargs), warn=True)
    GazeboConnection().unpause_sim()


@pytest.mark.parametrize("env_kwargs", copy.deepcopy(env_kwargs))
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


def exp_vec_env(env_kwargs):
    """deprecated"""

    from blimp_env.envs.common.utils import my_vec_env
    from blimp_env.envs.script import spawn_simulation, close_simulation

    n_envs = 2
    close_simulation()
    spawn_simulation(n_env=n_envs, gui=True, task=ENV.default_config()["task"])
    rospy.init_node("RL_node", anonymous=False)

    env = my_vec_env(env_id=ENV, n_envs=n_envs)
    # check_env(env, warn=True)
    GazeboConnection().unpause_sim()


def exp_env_step(env_kwargs):
    from blimp_env.envs.script import spawn_simulation

    spawn_simulation(n_env=1, gui=False, task=ENV.default_config()["task"])
    env = ENV(env_kwargs)
    env.reset()
    while not rospy.is_shutdown():
        action = np.ones(8) * 0
        obs, _, _, _ = env.step(action)

    GazeboConnection().unpause_sim()


# exp_env_step(env_kwargs[1])
# exp_vec_env(env_kwargs[1])
