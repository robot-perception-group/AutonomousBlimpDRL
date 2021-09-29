import pytest

from blimp_env.envs.common.abstract import AbstractEnv
from blimp_env.envs.common.target import target_factory
from blimp_env.envs.common.gazebo_connection import GazeboConnection
import rospy
import numpy as np
import time


path_target_kwargs = {
    "type": "PATH",
    "target_name_space": "target_0",
    "DBG_ROS": True,
}
goal_target_kwargs = {
    "type": "GOAL",
    "target_name_space": "goal_0",
    "DBG_ROS": True,
}
planar_target_kwargs = {
    "type": "PlanarGoal",
    "name_space": "machine_0",
    "target_name_space": "goal_0",
    "DBG_ROS": True,
}

target_kwargs = [
    path_target_kwargs,
    goal_target_kwargs,
    planar_target_kwargs,
]

vec = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 2, 0),
    (0, -2, 0),
    (0, 0, 3),
    (0, 0, -3),
    (np.sqrt(3), -1, 0),
]


def expected_euler_from_vec(input):
    expected_results = {
        str(vec[0]): [0, 0, 0],
        str(vec[1]): [0, 0, 3.14159265],
        str(vec[2]): [0, 0, 1.5707963268],
        str(vec[3]): [0, 0, -1.5707963268],
        str(vec[4]): [0, -1.5707963268, 0],
        str(vec[5]): [0, 1.5707963268, 0],
        str(vec[6]): [0, 0, -0.5235987756],
    }

    return expected_results[str(input)]


@pytest.mark.parametrize("target_kwargs", target_kwargs)
@pytest.mark.parametrize("vec", vec)
def test_vec_to_euler(target_kwargs, vec):
    target_type = target_factory(AbstractEnv, target_kwargs)

    result = target_type.euler_from_vec(vec)
    expect = expected_euler_from_vec(vec)
    np.testing.assert_allclose(np.array(result), np.array(expect))


def exp_target(target_kwargs):
    from blimp_env.envs.script import spawn_simulation_for_testing

    spawn_simulation_for_testing(1, gui=False)

    rospy.init_node("test", anonymous=False)
    target_type = target_factory(AbstractEnv, target_kwargs)
    target_type.check_connection()

    for _ in range(10):
        goal = target_type.sample()
        print("goal:", goal)
        time.sleep(0.1)

    GazeboConnection().unpause_sim()

    assert target_type.space().contains(goal[0])


# exp_target(target_kwargs[1])
