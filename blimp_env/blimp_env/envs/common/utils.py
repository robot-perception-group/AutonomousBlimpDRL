""" env util """
import os
from types import LambdaType
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import rospy
from functools import wraps


def with_retry(
    retries_limit: int = 3,
    allowed_exceptions: Optional[Sequence[Exception]] = None,
):
    """a decorator retry operation a few times and allow parameterize

    Args:
        retries_limit (int, optional):
            [maximum retry number].Defaults to 3.
        allowed_exceptions (Optional[Sequence[Exception]], optional):
            [if specified, retry the operation]. Defaults to None.
    """

    def retry(operation):
        """a decorator retry operation a few times"""

        @wraps(operation)
        def wrapped(*args, **kwargs):
            last_raised = None
            for _ in range(retries_limit):
                try:
                    return operation(*args, **kwargs)
                except allowed_exceptions as exc:
                    rospy.loginfo("retrying %s due to %s", operation.__qualname__, exc)
                    last_raised = exc
            raise last_raised

        return wrapped

    return retry


def update_dict(dic: Dict, key: str, new_v: int) -> Dict:
    """update nested dictionary key value with new_value

    Args:
        dic ([dict]): a nested dictionary with variant length
        key ([str]): key at any depth
        new_v ([int]): new value for same key of dic

    Returns:
        [dict]: dictionary with new value
    """
    for k, val in dic.items():
        if isinstance(val, dict):
            update_dict(val, key, new_v)

        if k == key:
            dic[k] = new_v
    return dic


def my_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,  # pylint: disable=unused-argument
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.
    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs

    def make_env(rank):
        def _init():
            robot_id = str(rank)
            env_kwargs.update({"robotID": robot_id})

            # if isinstance(env_id, str):
            #     env = gym.make(env_id, env_kwargs)
            # else:
            env = env_id(env_kwargs)

            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = (
                os.path.join(monitor_dir, str(rank))
                if monitor_dir is not None
                else None
            )
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    # return vec_env_cls(
    #     [make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs
    # )
    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)])


def parallel_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,  # pylint: disable=unused-argument
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.
    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs

    def make_env(rank):
        def _init():
            robot_id = str(rank)
            env_kwargs.update(
                {
                    "robotID": robot_id,
                    "ros_port": 11311 + int(robot_id),
                    "gaz_port": 11351 + int(robot_id),
                }
            )

            # if isinstance(env_id, str):
            #     env = gym.make(env_id, env_kwargs)
            # else:
            env = env_id(env_kwargs)

            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = (
                os.path.join(monitor_dir, str(rank))
                if monitor_dir is not None
                else None
            )
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    # return vec_env_cls(
    #     [make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs
    # )
    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)])


class RangeObj:
    """obj store min and max value"""

    def __init__(self, attr_min_max, attr_scaled_min_max):
        self.min = attr_min_max[0]
        self.max = attr_min_max[1]
        self.scaled_min = attr_scaled_min_max[0]
        self.scaled_max = attr_scaled_min_max[1]

    def get_range(self):
        """get range

        Returns:
            [list]: [min, max] of type float
        """
        return [self.min, self.max]

    def get_scale_range(self):
        """get scaled range

        Returns:
            [list]: scaled [min, max] of type float
        """

        return [self.scaled_min, self.scaled_max]

    def get_all(self):
        """get range and scaled range

        Returns:
            [list]: [min, max] and scaled [min, max] of type float
        """

        return [self.min, self.max, self.scaled_min, self.scaled_max]


class DataObj:
    """data object has value and range"""

    def __init__(self, value=None, vrange: Dict[str, RangeObj] = None):
        self.value = value
        self.vrange = vrange

    def get_value(self):
        """set value as private and use method to
        get value of DataObj
        """
        raise NotImplementedError

    def get_scaled_value(self):
        """scale the value and get scaled value"""
        raise NotImplementedError


def create_range_obj() -> Dict:
    scale_min_max = (-1, 1)
    position_range = {
        "x": RangeObj((-100, 100), scale_min_max),
        "y": RangeObj((-100, 100), scale_min_max),
        "z": RangeObj((-200, 0), scale_min_max),
    }  # NED frame
    vel_range = {
        "x": RangeObj((-8, 8), scale_min_max),
        "y": RangeObj((-8, 8), scale_min_max),
        "z": RangeObj((-2, 2), scale_min_max),
    }  # NED frame
    acc_range = {
        "x": RangeObj((-5, 5), scale_min_max),
        "y": RangeObj((-5, 5), scale_min_max),
        "z": RangeObj((-2, 2), scale_min_max),
    }
    ori_range = {
        "x": RangeObj((-1, 1), scale_min_max),
        "y": RangeObj((-1, 1), scale_min_max),
        "z": RangeObj((-1, 1), scale_min_max),
        "w": RangeObj((-1, 1), scale_min_max),
    }
    ang_range = {
        "x": RangeObj((-3.14, 3.14), scale_min_max),
        "y": RangeObj((-3.14, 3.14), scale_min_max),
        "z": RangeObj((-3.14, 3.14), scale_min_max),
    }
    ang_vel_range = {
        "x": RangeObj((-40, 40), scale_min_max),
        "y": RangeObj((-40, 40), scale_min_max),
        "z": RangeObj((-30, 30), scale_min_max),
    }
    return {
        "position_range": position_range,
        "vel_range": vel_range,
        "acc_range": acc_range,
        "ori_range": ori_range,
        "ang_range": ang_range,
        "ang_vel_range": ang_vel_range,
    }


def lmap(v, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
