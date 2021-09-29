import pytest
from blimp_env.envs.common.abstract import AbstractEnv
from blimp_env.envs.common.action import action_factory
import numpy as np

continuous_action_kwargs = {"type": "ContinuousAction"}
simple_continuous_action_kwargs = {"type": "SimpleContinuousAction"}

continuous_kwargs = [
    continuous_action_kwargs,
    simple_continuous_action_kwargs,
]

discrete_action_kwargs = {"type": "DiscreteMetaAction"}
discrete_hover_action_kwargs = {"type": "DiscreteMetaHoverAction"}

discrete_kwargs = [
    discrete_action_kwargs,
    discrete_hover_action_kwargs,
]

all_kwargs = continuous_kwargs + discrete_kwargs


def get_testset_process_action(action_type, id):
    act_dim = action_type.act_dim
    test_process_act = [
        np.ones(act_dim) * 1,
        np.ones(act_dim) * 1.5,
        np.ones(act_dim) * 0,
        np.ones(act_dim) * -1,
        np.ones(act_dim) * -1.5,
    ]
    return test_process_act[id]


expect_process_act = [
    np.array(
        [2000, 2000, 2000, 2000, 2000, 2000, 2000, 1500, 2000, 1500, 1500, 1500]
    ).reshape(12, 1),
    np.array(
        [2000, 2000, 2000, 2000, 2000, 2000, 2000, 1500, 2000, 1500, 1500, 1500]
    ).reshape(12, 1),
    np.array(
        [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
    ).reshape(12, 1),
    np.array(
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1500, 1000, 1500, 1500, 1500]
    ).reshape(12, 1),
    np.array(
        [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1500, 1000, 1500, 1500, 1500]
    ).reshape(12, 1),
]


@pytest.mark.parametrize("id", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("all_kwargs", all_kwargs)
def test_process_action(all_kwargs, id):
    action_type = action_factory(AbstractEnv, all_kwargs)
    action_type.act_noise_stdv = 0

    test = get_testset_process_action(action_type, id)
    result = action_type.process_action(test)
    expect = expect_process_act[id]
    np.testing.assert_allclose(result, expect)


@pytest.mark.parametrize("all_kwargs", all_kwargs)
def test_get_cur_act(all_kwargs):
    action_type = action_factory(AbstractEnv, all_kwargs)
    act = action_type.get_cur_act()
    assert act.shape == (8,)


@pytest.mark.parametrize("all_kwargs", all_kwargs)
def test_action_rew(all_kwargs):
    action_type = action_factory(AbstractEnv, all_kwargs)
    rew = action_type.action_rew()
    assert isinstance(rew, float)
    assert rew >= 0 and rew <= 1


def expect_decode_act(kwargs, act):
    if kwargs["type"] == "DiscreteMetaAction":
        expect = {
            0: [0, 0, 0, 0, 0, -1, 0.0, 0.0],
            1: [0, 0, 0, 0, 0, -1, 0.01, 0.01],
            2: [0, 0, 0, 0, 0, -1, 0.0, 0.0],
            3: [0, 0.025, 0.025, 0, 0, -1, 0, 0],
            4: [0, -0.025, -0.025, 0, 0, -1, 0, 0],
            5: [0.025, 0, 0, 0.025, 0.025, -1, 0, 0],
            6: [-0.025, 0, 0, -0.025, -0.025, -1, 0, 0],
        }

    if kwargs["type"] == "DiscreteMetaHoverAction":
        expect = {
            0: [0, 0, 0, 0, 0, 0, 0.0, 0.0],
            1: [0, 0, 0, 0, 0, 0, 0.01, 0.01],
            2: [0, 0, 0, 0, 0, 0, -0.01, -0.01],
            3: [0, 0, 0, 0, 0, 0.025, 0, 0],
            4: [0, 0, 0, 0, 0, -0.025, 0, 0],
            5: [0.1, 0, 0, 0, 0, 0, 0, 0],
            6: [-0.1, 0, 0, 0, 0, 0, 0, 0],
        }

    return expect[act]


@pytest.mark.parametrize("act", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("kwargs", discrete_kwargs)
def test_decode_act(kwargs, act):
    action_type = action_factory(AbstractEnv, kwargs)

    defined_act = action_type.actions[act]
    assert isinstance(defined_act, str)

    decoded_act = action_type.decode_act(defined_act)
    expect = np.array(expect_decode_act(kwargs, act))
    assert isinstance(decoded_act, np.ndarray)
    np.testing.assert_allclose(decoded_act, expect)
