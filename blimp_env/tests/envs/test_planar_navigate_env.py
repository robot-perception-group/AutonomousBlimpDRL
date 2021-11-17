import pytest
from blimp_env.envs import PlanarNavigateEnv
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from stable_baselines3.common.env_checker import check_env
import copy
import numpy as np

ENV = PlanarNavigateEnv

env_kwargs = {
    "DBG": True,
    "simulation": {
        "auto_start_simulation": False,
    },
    "observation": {
        "DBG_ROS": False,
        "DBG_OBS": True,
    },
    "action": {
        "DBG_ACT": True,
    },
    "target": {"DBG_ROS": False},
}

# ============== test env ==============#


def test_env_functions():
    check_env(ENV(copy.deepcopy(env_kwargs)), warn=True)
    GazeboConnection().unpause_sim()


def test_env_step():
    env = ENV(copy.deepcopy(env_kwargs))
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, rew, terminal, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert isinstance(rew, float)
        assert isinstance(terminal, bool)
        assert isinstance(info, dict)

        assert rew >= -1 and rew <= 1

    GazeboConnection().unpause_sim()


def test_compute_success_rew():
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.compute_success_rew

    achieved_goal = np.ones(3)
    desired_goal = achieved_goal
    result = fn(achieved_goal, desired_goal)
    expect = 1
    np.testing.assert_allclose(result, expect)

    achieved_goal = np.zeros(3)
    desired_goal = achieved_goal
    result = fn(achieved_goal, desired_goal)
    expect = 1
    np.testing.assert_allclose(result, expect)

    achieved_goal = 1 * np.ones(3)
    desired_goal = 2 * np.ones(3)
    result = fn(achieved_goal, desired_goal)
    expect = 1
    np.testing.assert_allclose(result, expect)

    achieved_goal = 10 * np.ones(3)
    desired_goal = -10 * np.ones(3)
    result = fn(achieved_goal, desired_goal)
    expect = 0
    np.testing.assert_allclose(result, expect)

    for _ in range(100):
        achieved_goal = np.random.uniform(-100, 100, 3)
        desired_goal = np.random.uniform(-100, 100, 3)
        rew = fn(achieved_goal, desired_goal)
        assert isinstance(rew, float)
        assert rew == 0.0 or rew == 1.0


def test_is_terminal(mocker):
    env = ENV(copy.deepcopy(env_kwargs))
    mock_fn = "blimp_env.envs.planar_navigate_env.PlanarNavigateEnv.compute_success_rew"
    dummy_obs_info = {"position": np.array([0, 0, 0])}

    env.config["duration"] = 100
    env.steps = 5
    mocker.patch(mock_fn, return_value=0.0)
    result = env._is_terminal(dummy_obs_info)
    expect = False
    assert result == expect

    env.config["duration"] = 100
    env.steps = 5
    mocker.patch(mock_fn, return_value=1.0)
    result = env._is_terminal(dummy_obs_info)
    expect = True
    assert result == expect

    env.config["duration"] = 100
    env.steps = 200
    mocker.patch(mock_fn, return_value=0.0)
    result = env._is_terminal(dummy_obs_info)
    expect = True
    assert result == expect

    env.config["duration"] = 100
    env.steps = 5
    mocker.patch(mock_fn, return_value=0.0)
    result = env._is_terminal(dummy_obs_info)
    expect = False
    assert result == expect


def test_rew(mocker):
    env = ENV(copy.deepcopy(env_kwargs))
    env.config["tracking_reward_weights"] = np.array([0.1, 0.2, 0.3, 0.4])
    env.config["reward_weights"] = np.array([1, 0.95, 0.05])
    mock_fn = "blimp_env.envs.planar_navigate_env.PlanarNavigateEnv.compute_success_rew"
    dummy_obs_info = {"position": np.array([0, 0, 0])}

    def dummy_act_rew():
        return 0

    mocker.patch(mock_fn, return_value=1.0)
    obs = np.zeros(9)
    obs[1] = -1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = 1
    np.testing.assert_allclose(result, expect)

    env.action_type.action_rew = dummy_act_rew
    mocker.patch(mock_fn, return_value=0.0)
    obs = np.zeros(9)
    obs[1] = -1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = 0
    np.testing.assert_allclose(result, expect)

    obs = np.zeros(9)
    obs[0] = 1
    obs[1] = -1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = -0.1 * 0.95
    np.testing.assert_allclose(result, expect)

    obs = np.zeros(9)
    obs[1] = 1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = -0.2 * 0.95
    np.testing.assert_allclose(result, expect)

    obs = np.zeros(9)
    obs[2] = 1
    obs[1] = -1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = -0.3 * 0.95
    np.testing.assert_allclose(result, expect)

    obs = np.zeros(9)
    obs[3] = 1
    obs[1] = -1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = -0.4 * 0.95
    np.testing.assert_allclose(result, expect)

    obs = np.zeros(9)
    obs[4] = 1
    obs[1] = -1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = 0
    np.testing.assert_allclose(result, expect)

    def dummy_act_rew():
        return -1

    env.action_type.action_rew = dummy_act_rew
    mocker.patch(mock_fn, return_value=0.0)
    obs = np.ones(9)
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = -1.0
    np.testing.assert_allclose(result, expect)

    obs = np.zeros(9)
    obs[1] = -1
    result, _ = env._reward(obs, [], dummy_obs_info)
    expect = -0.05
    np.testing.assert_allclose(result, expect)


# ============== test obs ==============#


def test_compute_psi_diff():
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.observation_type.compute_psi_diff

    goal_pos, obs_pos, obs_psi = np.array([1, 0, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = 0.0
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.ones(3), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = 0.25 * np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.array([0, 1, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = 0.5 * np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.array([-1, 1, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = 0.75 * np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.array([-1, 1e-9, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.array([1, -1, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = -0.25 * np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.array([0, -1, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = -0.5 * np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.array([-1, -1, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = -0.75 * np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.array([-1, 0, 0]), np.zeros(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = -np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.zeros(3), np.ones(3), 0
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = -0.75 * np.pi
    np.testing.assert_allclose(result, expect)

    goal_pos, obs_pos, obs_psi = np.ones(3), -np.ones(3), 0.5 * np.pi
    result = fn(goal_pos, obs_pos, obs_psi)
    expect = -0.25 * np.pi
    np.testing.assert_allclose(result, expect)

    for _ in range(100):
        goal_pos = np.random.uniform(-1, 1, 3)
        obs_pos = np.random.uniform(-1, 1, 3)
        obs_psi = np.random.uniform(-1, 1)
        result = fn(goal_pos, obs_pos, obs_psi)
        assert isinstance(result, float)
        assert result >= -np.pi and result <= np.pi


def get_test_scale_obs_dict_io():
    in_list, out_list = [], []
    in_list.append(
        {
            "z_diff": np.array(100),
            "planar_dist": np.array(200 * np.sqrt(2)),
            "psi_diff": np.array(np.pi),
            "vel_diff": np.array(11.5),
            "vel": np.array(11.5),
        }
    )
    out_list.append(
        {
            "z_diff": 1,
            "planar_dist": 1,
            "psi_diff": 1,
            "vel_diff": 1,
            "vel": 1,
        }
    )
    in_list.append(
        {
            "z_diff": np.array(0),
            "planar_dist": np.array(100 * np.sqrt(2)),
            "psi_diff": np.array(0),
            "vel_diff": np.array(0),
            "vel": np.array(11.5 / 2),
        }
    )
    out_list.append(
        {
            "z_diff": 0,
            "planar_dist": 0,
            "psi_diff": 0,
            "vel_diff": 0,
            "vel": 0,
        }
    )
    in_list.append(
        {
            "z_diff": np.array(-100),
            "planar_dist": np.array(0),
            "psi_diff": np.array(-np.pi),
            "vel_diff": np.array(-11.5),
            "vel": np.array(0),
        }
    )
    out_list.append(
        {
            "z_diff": -1,
            "planar_dist": -1,
            "psi_diff": -1,
            "vel_diff": -1,
            "vel": -1,
        }
    )
    in_list.append(
        {
            "z_diff": np.array(50),
            "planar_dist": np.array(150 * np.sqrt(2)),
            "psi_diff": np.array(0.5 * np.pi),
            "vel_diff": np.array(0.5 * 11.5),
            "vel": np.array(0.75 * 11.5),
        }
    )
    out_list.append(
        {
            "z_diff": 0.5,
            "planar_dist": 0.5,
            "psi_diff": 0.5,
            "vel_diff": 0.5,
            "vel": 0.5,
        }
    )

    return in_list, out_list


@pytest.mark.parametrize(
    "idx", [i for i in range(len(get_test_scale_obs_dict_io()[0]))]
)
def test_scale_obs_dict(idx):
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.observation_type.scale_obs_dict

    input, expect = get_test_scale_obs_dict_io()
    result = fn(input[idx], 0.0)
    for k, _ in result.items():
        np.testing.assert_allclose(result[k], expect[idx][k])


def get_test_process_obs_io():
    in_list, out_list = [], []
    obs_dict = {
        "position": np.array([10, 10, 10]),
        "velocity": np.array([3, 2, 1]),
        "angle": np.array([0, 0, np.pi]),
    }
    goal_dict = {
        "position": np.array([20, 20, 50]),
        "velocity": 2.5,
    }
    in_list.append([obs_dict, goal_dict, False])
    out_list.append(
        {
            "z_diff": -40,
            "planar_dist": 10 * np.sqrt(2),
            "psi_diff": -0.75 * np.pi,
            "vel_diff": np.sqrt(14) - 2.5,
            "vel": np.sqrt(14),
        }
    )
    obs_dict = {
        "position": np.array([40, 0, 70]),
        "velocity": np.array([3, 3, 3]),
        "angle": np.array([0, 0, 0]),
    }
    goal_dict = {
        "position": np.array([20, 20, 50]),
        "velocity": 5,
    }
    in_list.append([obs_dict, goal_dict, False])
    out_list.append(
        {
            "z_diff": 20,
            "planar_dist": 20 * np.sqrt(2),
            "psi_diff": 0.75 * np.pi,
            "vel_diff": np.sqrt(27) - 5,
            "vel": np.sqrt(27),
        }
    )
    return in_list, out_list


@pytest.mark.parametrize("idx", [i for i in range(len(get_test_process_obs_io()[0]))])
def test_process_obs(idx):
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.observation_type.process_obs

    input, expect = get_test_process_obs_io()
    result = fn(*input[idx])
    for k, _ in result.items():
        np.testing.assert_allclose(result[k], expect[idx][k])


# ============== test act ==============#


def get_test_process_action_io():
    in_list, out_list = [], []
    in_list.append([np.array([0, 0, 0, 0]), np.array([0.0, 0.0, 0.0, 0.0])])
    out_list.append(np.array([0, 0, 0, 0]))

    in_list.append([np.array([0, 0, 1, 0]), np.array([0.0, 0.0, 0.0, 0.0])])
    out_list.append(np.array([0, 0, 0, 0]))

    in_list.append([np.array([0, 0, 0, -1]), np.array([0.0, 0.0, 0.0, 0.0])])
    out_list.append(np.array([0, 0, 0, 0]))

    in_list.append([np.array([1, 1, 1, 1]), np.array([0.0, 0.0, 0.0, 0.0])])
    out_list.append(np.array([0.1, 0.1, 0.0, 0.04]))

    in_list.append([np.array([-1, -1, -1, -1]), np.array([0.0, 0.0, 0.0, 0.0])])
    out_list.append(np.array([-0.1, -0.1, -0.1, 0.0]))

    in_list.append([np.array([1, 1, 1, 1]), np.array([0.0, 0.0, -0.2, 0.0])])
    out_list.append(np.array([0.1, 0.1, -0.1, 0.04]))

    return in_list, out_list


@pytest.mark.parametrize(
    "idx", [i for i in range(len(get_test_process_action_io()[0]))]
)
def test_process_action(idx):
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.action_type.process_action

    env.action_type.forward_servo=True
    env.action_type.disable_servo=False
    input, expect = get_test_process_action_io()
    result = fn(
        *input[idx],
    )
    np.testing.assert_allclose(result, expect[idx])


def test_process_actuator_state():
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.action_type.process_actuator_state

    input = np.array([0, 0, 0, 0])
    expect = 1500 * np.ones(12)
    result = fn(input, 0)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([1.5, 1.4, 1.3, 1.2])
    expect = np.array(
        [2000, 2000, 2000, 2000, 2000, 2000, 2000, 1500, 2000, 1500, 1500, 1500]
    )
    result = fn(input, 0)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([-1.2, -1.3, -1.4, -1.5])
    expect = np.array(
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1500, 1000, 1500, 1500, 1500]
    )
    result = fn(input, 0)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([1, 0, 0, 0])
    expect = np.array(
        [2000, 1500, 1500, 2000, 2000, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
    )
    result = fn(input, 0)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([0, 1, 0, 0])
    expect = np.array(
        [1500, 2000, 2000, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
    )
    result = fn(input, 0)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([0, 0, 1, 0])
    expect = np.array(
        [1500, 1500, 1500, 1500, 1500, 2000, 1500, 1500, 1500, 1500, 1500, 1500]
    )
    result = fn(input, 0)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([0, 0, 0, 1])
    expect = np.array(
        [1500, 1500, 1500, 1500, 1500, 1500, 2000, 1500, 2000, 1500, 1500, 1500]
    )
    result = fn(input, 0)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))


def test_match_channel():
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.action_type.match_channel

    input = np.array([1500, 1500, 1500, 1500])
    expect = 1500 * np.ones(12)
    result = fn(input)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([2000, 1500, 1500, 1500])
    expect = np.array(
        [2000, 1500, 1500, 2000, 2000, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
    )
    result = fn(input)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([1500, 2000, 1500, 1500])
    expect = np.array(
        [1500, 2000, 2000, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
    )
    result = fn(input)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([1500, 1500, 2000, 1500])
    expect = np.array(
        [1500, 1500, 1500, 1500, 1500, 2000, 1500, 1500, 1500, 1500, 1500, 1500]
    )
    result = fn(input)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))

    input = np.array([1500, 1500, 1500, 2000])
    expect = np.array(
        [1500, 1500, 1500, 1500, 1500, 1500, 2000, 1500, 2000, 1500, 1500, 1500]
    )
    result = fn(input)
    np.testing.assert_allclose(result, np.expand_dims(expect, axis=1))


def test_action_rew():
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.action_type.action_rew
    scale = np.array([0.5, 1, 1])

    env.action_type.cur_act = np.array([0, 0, 0, 0])
    result = fn(scale)
    expect = 0
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([1, 0, 0, 0])
    result = fn(scale)
    expect = -0.2
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([0, 0, 0, 1])
    result = fn(scale)
    expect = -0.8
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([1, 0, 0, 1])
    result = fn(scale)
    expect = -1
    np.testing.assert_allclose(result, expect)


def test_get_cur_act():
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.action_type.get_cur_act

    env.action_type.cur_act = np.array([0, 0, 0, 0])
    result = fn()
    expect = np.array([0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1)
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([1, 0, 0, 0])
    result = fn()
    expect = np.array([1, 0, 0, 1, 1, 0, 0, 0]).reshape(-1)
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([0, 1, 0, 0])
    result = fn()
    expect = np.array([0, 1, 1, 0, 0, 0, 0, 0]).reshape(-1)
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([0, 0, 1, 0])
    result = fn()
    expect = np.array([0, 0, 0, 0, 0, 1, 0, 0]).reshape(-1)
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([0, 0, 0, 1])
    result = fn()
    expect = np.array([0, 0, 0, 0, 0, 0, 1, 1]).reshape(-1)
    np.testing.assert_allclose(result, expect)

    env.action_type.cur_act = np.array([1, 1, 1, 1])
    result = fn()
    expect = np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1)
    np.testing.assert_allclose(result, expect)


def test_check_action_publisher_connection():
    env = ENV(copy.deepcopy(env_kwargs))
    GazeboConnection().unpause_sim()
    connected = env.action_type.check_publishers_connection()
    assert connected == True


# ============== test target ==============#
def test_sample():
    env = ENV(copy.deepcopy(env_kwargs))
    fn = env.target_type.sample

    goal = fn()
    assert isinstance(goal, dict)
    assert isinstance(goal["position"], np.ndarray)
    assert isinstance(goal["velocity"], float)
    assert isinstance(goal["angle"], np.ndarray)

    env.target_type.pos_cmd_data = np.array([1, 2, 3])
    env.target_type.vel_cmd_data = 5.0
    env.target_type.ang_cmd_data = np.array([5, 6, 7])
    goal = fn()
    np.testing.assert_allclose(goal["position"], np.array([1, 2, 3]))
    np.testing.assert_allclose(goal["velocity"], 5.0)
    np.testing.assert_allclose(goal["angle"], np.array([5, 6, 7]))


def test_timeout_handle():
    env = ENV(copy.deepcopy(env_kwargs))
    result = env.target_type.timeout_handle()

    assert result == {"kill_goal_reply": 0, "spawn_goal_reply": 0}
