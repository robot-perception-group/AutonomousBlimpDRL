from sys import exec_prefix
import pytest
from blimp_env.envs.common.data_processor import DataProcessor, SimpleDataProcessor
import numpy as np
from geometry_msgs.msg import Point, Quaternion
from blimp_env.envs.common.utils import RangeObj, DataObj


def test_add_noise():
    np.random.seed(1)
    result = DataProcessor().add_noise(np.array([-1.0, 0.0, 1.0]))
    expect = np.array([-0.91878273, -0.03058782, 0.97359141])
    np.testing.assert_allclose(result, expect)


def test_scaler():
    result = DataProcessor().scaler(0, (-1, 1), (1000, 2000))
    expect = 1500.0
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().scaler(9.8, (9.5, 10.5), (-1, 1))
    expect = -0.3999999999999986
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().scaler(3.14, (-3.14, 3.14), (-1, 1))
    expect = 1.0
    np.testing.assert_allclose(result, expect)


def test_clip():
    result = DataProcessor().clip(np.array([1234.0, 999.0, 2001.0]), (1000, 2000))
    expect = np.array([1234.0, 1000.0, 2000.0])
    np.testing.assert_allclose(result, expect)


def test_scale_action():
    act_range = RangeObj((1000, 2000), (-1, 1))
    result = DataProcessor().scale_action(np.array([-1, 0, 1]), act_range)
    expect = np.array([1000.0, 1500.0, 2000.0])
    np.testing.assert_allclose(result, expect)


def test_scale_and_clip():
    val_range = RangeObj((-10, 10), (-1, 1))
    result = DataProcessor().scale_and_clip(np.array(-11), val_range)
    expect = -1.0
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().scale_and_clip(np.array([9.0, -11, 11]), val_range)
    expect = np.array([0.9, -1.0, 1.0])
    np.testing.assert_allclose(result, expect)


def test_augment_action():
    result = DataProcessor().augment_action(np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    expect = np.array(
        [
            0.0e00,
            1.0e00,
            2.0e00,
            3.0e00,
            4.0e00,
            5.0e00,
            6.0e00,
            1.5e03,
            7.0e00,
            1.5e03,
            1.5e03,
            1.5e03,
        ]
    ).reshape(12, 1)
    np.testing.assert_allclose(result, expect)


def test_scale_and_clip_data_obj():
    range_a = {
        "x": RangeObj((-10, 10), (-1, 1)),
        "y": RangeObj((-10, 10), (-1, 1)),
        "z": RangeObj((-10, 10), (-1, 1)),
    }
    value_a = Point(-100, 11, 5)
    result = DataProcessor().scale_and_clip_data_obj(value_a, range_a)
    expect = np.array([-1.0, 1.0, 0.5])
    np.testing.assert_allclose(result, expect)


def test_normalize_data_obj_dict():
    range_a = {
        "x": RangeObj((-10, 10), (-1, 1)),
        "y": RangeObj((-10, 10), (-1, 1)),
        "z": RangeObj((-10, 10), (-1, 1)),
    }
    dataObj_a = DataObj(Point(10, -20, 3), range_a)

    range_b = {
        "x": RangeObj((-1, 1), (-1, 1)),
        "y": RangeObj((-1, 1), (-1, 1)),
        "z": RangeObj((-1, 1), (-1, 1)),
    }
    dataObj_b = DataObj(Point(4, 0.5, 60), range_b)

    obj_dict = {"obj_a": dataObj_a, "obj_b": dataObj_b}

    result = DataProcessor().normalize_data_obj_dict(obj_dict)
    expect = {"obj_a": np.array([1.0, -1.0, 0.3]), "obj_b": np.array([1.0, 0.5, 1.0])}

    np.testing.assert_allclose(result["obj_a"], expect["obj_a"])
    np.testing.assert_allclose(result["obj_b"], expect["obj_b"])


def test_simple_augment_action():
    result = SimpleDataProcessor().augment_action(np.array([0, 1, 2, 3]))
    expect = np.array(
        [
            [0.0e00],
            [1.0e00],
            [1.0e00],
            [0.0e00],
            [0.0e00],
            [2.0e00],
            [3.0e00],
            [1.5e03],
            [3.0e00],
            [1.5e03],
            [1.5e03],
            [1.5e03],
        ]
    )
    np.testing.assert_allclose(result, expect)


def test_quaternion_from_euler():
    result = DataProcessor().quaternion_from_euler(0, 0, 0)
    expect = [0.0, 0.0, 0.0, 1.0]
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().quaternion_from_euler(-1.5707963268, 0, 0)
    expect = [-0.70710678118, 0.0, 0.0, 0.70710678118]
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().quaternion_from_euler(0, 1.5707963268, 0)
    expect = [0.0, 0.70710678118, 0.0, 0.70710678118]
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().quaternion_from_euler(0, 0, 1.5707963268)
    expect = [0.0, 0.0, 0.70710678118, 0.70710678118]
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().quaternion_from_euler(0, 0, -1.5707963268)
    expect = [0.0, 0.0, -0.70710678118, 0.70710678118]
    np.testing.assert_allclose(result, expect)


def test_euler_from_quaternion():
    result = DataProcessor().euler_from_quaternion(0, 0, 0, 1)
    expect = [0.0, 0.0, 0.0]
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().euler_from_quaternion(
        -0.70710678118, 0.0, 0.0, 0.70710678118
    )
    expect = [-1.5707963268, 0, 0]
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().euler_from_quaternion(
        0.0, 0.0, 0.70710678118, 0.70710678118
    )
    expect = [0, 0, 1.5707963268]
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().euler_from_quaternion(
        0.0, 0.0, -0.70710678118, 0.70710678118
    )
    expect = [0, 0, -1.5707963268]
    np.testing.assert_allclose(result, expect)


def test_angle_diff():
    result = DataProcessor().angle_diff(
        np.array([0.9, 1.0, 1.1]), np.array([0.0, 0.0, 0.0])
    )
    expect = np.array([0.9, 1.0, -0.9])
    np.testing.assert_allclose(result, expect)

    result = DataProcessor().angle_diff(
        np.array([-0.9, -1.0, -1.1]), np.array([0.0, 0.0, 0.0])
    )
    expect = np.array([-0.9, -1.0, 0.9])
    np.testing.assert_allclose(result, expect)


def test_quat_diff():
    q_a = np.array([0, 0, 0, 1])
    q_b = np.array([0, 0, 0.70710678118, 0.70710678118])
    result = DataProcessor().quat_diff(q_a, q_b)
    expect = 0.5
    np.testing.assert_allclose(result, expect)

    q_b = np.array([0, 0, -0.70710678118, -0.70710678118])
    result = DataProcessor().quat_diff(q_a, q_b)
    expect = 0.5
    np.testing.assert_allclose(result, expect)

    q_b = np.array([0, 0, 1.0, 0.0])
    result = DataProcessor().quat_diff(q_a, q_b)
    expect = 1
    np.testing.assert_allclose(result, expect)

    q_b = np.array([0, 0, -1.0, 0.0])
    result = DataProcessor().quat_diff(q_a, q_b)
    expect = 1
    np.testing.assert_allclose(result, expect)
