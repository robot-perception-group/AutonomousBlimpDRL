#!/usr/bin/env python
""" environment abstract test """
from __future__ import absolute_import, division, print_function

import os
import socket
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import rosgraph
import rospy
from blimp_env.envs.common.action import (
    Action,
    ActionType,
    action_factory,
)
from blimp_env.envs.common.controllers_connection import ControllersConnection
from blimp_env.envs.common.data_processor import DataProcessor
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.common.observation import ObservationType, observation_factory
from blimp_env.envs.common.target import TargetType, target_factory
from blimp_env.envs.common.utils import update_dict
from blimp_env.envs.script.blimp_script import (
    spawn_simulation_on_different_port,
    spawn_simulation_on_marvin,
)
from geometry_msgs.msg import Point
from gym.utils import seeding

Observation = Union[np.ndarray, float]
Target = Union[np.ndarray, float]


class AbstractEnv(gym.Env):
    """generic blimp env"""

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        self.config = self.default_config()
        self.configure(config)

        self.np_random = None
        self.seed(self.config["seed"])

        self.goal: Tuple[np.ndarray, Dict[str, np.ndarray]]

        self.target_type: TargetType
        self.observation_type: ObservationType
        self.action_type: ActionType
        self.target_space = None
        self.action_space = None
        self.observation_space = None
        # self.define_spaces()

        self.time = 0
        self.steps = 0
        self.done = False

        self.viewer = None

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.
        :return: a configuration dict
        """
        return {
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "seed": None,
        }

    def configure(self, config: Optional[Dict[Any, Any]]) -> None:
        """update self.config with external environment config

        Args:
            config (dict): env config
        """
        if config:
            self.config.update(config)

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def define_spaces(self) -> None:
        """gym space"""
        self.target_type = target_factory(self, self.config["target"])
        self.goal = self.target_type.sample()

        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])

        self.target_space = self.target_type.space()
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        raise NotImplementedError

    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information
        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "observation": obs,
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass
        return info

    def _cost(self, action: Action) -> float:
        """
        A constraint metric, for budgeted MDP.
        If a constraint is defined, it must be used
        with an alternate reward that doesn't contain it as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def reset(self) -> Observation:
        self.time = self.steps = 0
        self.done = False
        self._reset()
        return self.observation_type.observe()

    def _reset(self) -> None:
        raise NotImplementedError

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(
            int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        ):
            if (
                action is not None
                and self.time
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):
                self.action_type.act(action)

    def render(self):  # pylint: disable=arguments-differ
        """render env"""
        raise NotImplementedError

    def close(self):
        """close env"""
        raise NotImplementedError

    def get_available_actions(self):
        """get available actions"""
        raise NotImplementedError


class ROSAbstractEnv(AbstractEnv):
    """ros abstract environment with publisher and subscriber"""

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.
        :return: a configuration dict
        """
        config = super().default_config()
        config.update(
            {
                "robot_id": "0",
                "name_space": "machine_",
                "real_experiment": False,
                "DBG": False,
                "simulation": {
                    "robot_id": "0",
                    "ros_ip": "localhost",
                    "ros_port": 11311,
                    "gaz_port": 11351,
                    "gui": False,
                    "enable_meshes": False,
                    "world": "basic",
                    "task": "navigate_goal",
                    "auto_start_simulation": False,
                    "update_robotID_on_workerID": True,
                },
                "observation": {
                    "type": "Kinematics",
                    "name_space": "machine_",
                    "orientation_type": "euler",
                    "real_experiment": False,
                    "DBG_ROS": False,
                    "DBG_OBS": False,
                },
                "action": {
                    "type": "ContinuousAction",
                    "robot_id": "0",
                    "name_space": "machine_",
                    "flightmode": 3,
                    "DBG_ACT": False,
                },
                "target": {
                    "type": "PATH",
                    "target_name_space": "target_",
                    "orientation_type": "euler",
                    "real_experiment": False,
                    "DBG_ROS": False,
                },
                "seed": 123,
                "simulation_frequency": 10,
                "policy_frequency": 2,
                "duration": 2400,
            }
        )

        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        # if rllib parallelization, use worker index as robot_id
        if hasattr(config, "worker_index"):
            config["robot_id"] = str(config.worker_index - 1)

        super().__init__(config=config)

        if self.config["simulation"]["auto_start_simulation"]:
            self.setup_env(int(self.config["robot_id"]))

        print(self.config)

        rospy.loginfo(
            "[ RL Node " + str(self.config["robot_id"]) + " ] Initialising..."
        )
        rospy.init_node(
            "RL_node_" + str(self.config["robot_id"]),
            anonymous=True,
            disable_signals=True,
        )

        self.gaz = GazeboConnection(
            start_init_physics_parameters=True, reset_world_or_sim="WORLD"
        )
        self.controllers_object = ControllersConnection(
            namespace=self.config["name_space"]
        )
        self.rate = rospy.Rate(self.config["simulation_frequency"])
        self.data_processor = DataProcessor()

        self.define_spaces()

        self.step_info: Dict = {}
        self.dbg = self.config["DBG"]

        self._create_pubs_subs()
        self.gaz.unpause_sim()
        rospy.loginfo("[ RL Node " + str(self.config["robot_id"]) + " ] Initialized")

    def configure(self, config: Optional[Dict[Any, Any]]) -> None:
        if config:
            config_tmp0 = deepcopy(config)
            self_config_tmp = deepcopy(self.config)

            self.config.update(config_tmp0)
            for key in ["simulation", "observation", "action", "target"]:
                self.config[key].update(self_config_tmp[key])
                if key in config:
                    self.config[key].update(config[key])

        self._update_config_id()

    def _update_config_id(self):
        """synchronize parameters across config in all depth"""
        robot_id = str(self.config["robot_id"])
        name_space = self.config["name_space"] + robot_id
        target_name_space = self.config["target"]["target_name_space"] + robot_id
        real_experiment = self.config["real_experiment"]

        self.config = update_dict(self.config, "robot_id", robot_id)
        self.config = update_dict(self.config, "name_space", name_space)
        self.config = update_dict(self.config, "target_name_space", target_name_space)
        self.config = update_dict(self.config, "real_experiment", real_experiment)

    def setup_env(self, worker_index: int = 0):
        """setup gazebo env. Worker index is used to modify env config allowing
        training on different env in parallel. It is important to specify ros and gaz
        master channel to spawn env in parallel.

        Args:
            worker_index (int, optional): [the index of worker or subprocess]. Defaults to 1.
        """
        if worker_index >= 10:
            # spawn env on another pc
            marvin = True
            ros_ip = "frg07"
        else:
            marvin = False
            ros_ip = socket.gethostbyname(socket.gethostname())
        time.sleep(worker_index)
        ros_port = self.config["simulation"]["ros_port"] + worker_index
        gaz_port = self.config["simulation"]["gaz_port"] + worker_index
        host_addr = "http://" + str(ros_ip) + ":"

        os.environ["ROS_MASTER_URI"] = host_addr + str(ros_port) + "/"
        os.environ["GAZEBO_MASTER_URI"] = host_addr + str(gaz_port) + "/"

        while rosgraph.is_master_online():
            time.sleep(1)
            ros_port += 1
            gaz_port += 1
            os.environ["ROS_MASTER_URI"] = host_addr + str(ros_port) + "/"
            os.environ["GAZEBO_MASTER_URI"] = host_addr + str(gaz_port) + "/"
            print("current channel occupied, move to next channel:", ros_port)

        self.config["simulation"].update(
            {"ros_ip": ros_ip, "ros_port": ros_port, "gaz_port": gaz_port}
        )

        self._spawn_env(marvin)

    def _spawn_env(self, marvin=False):
        if marvin:
            spawn_simulation_on_marvin(**self.config["simulation"])
        else:
            spawn_simulation_on_different_port(**self.config["simulation"])

    def _create_pubs_subs(self):
        self.reward_publisher = rospy.Publisher(
            self.config["name_space"] + "/reward", Point, queue_size=1
        )

    def reset(self) -> Observation:
        self.time = self.steps = 0
        self.done = False
        self._reset()
        obs, _ = self.observation_type.observe()
        return obs

    def _reset(self):
        self._reset_gazebo()
        self._update_goal()

    def _reset_gazebo(self):
        self.gaz.unpause_sim()
        self._check_system_ready()
        self.gaz.pause_sim()

        self.gaz.reset_sim()

        self.gaz.unpause_sim()
        self._check_system_ready()
        self.gaz.pause_sim()

    def _check_system_ready(self):
        self.controllers_object.reset_blimp_joint_controllers()
        self.action_type.set_init_pose()
        self.observation_type.check_connection()
        self.target_type.check_connection()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """step

        Args:
            action (Action): [action taken from agent]

        Returns:
            Tuple[Observation, float, bool, dict]: [environment information]
        """

        self.steps += 1

        self.gaz.unpause_sim()
        obs, reward, terminal, info = self.one_step(action)
        self.gaz.pause_sim()

        assert isinstance(
            reward, (float, int)
        ), "The reward returned by `step()` must be a float"
        assert isinstance(terminal, bool), "terminal must be a boolean"
        assert isinstance(info, dict), "info must be a dict"

        return obs, reward, terminal, info

    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """perform a step action and observe result"""
        self.step_info.update({"step": self.steps})

        self._simulate(action)
        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)
        self.step_info.update(
            {"obs": obs, "reward": reward, "terminal": terminal, "info": info}
        )

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(
            int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        ):
            if (
                action is not None
                and self.time
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):
                self.action_type.act(action)
                self.step_info.update({"action": action})

            self.rate.sleep()

    def _update_goal(self):
        raise NotImplementedError

    def render(self) -> None:
        pass

    def _cost(self, action: Action) -> float:
        pass

    def close(self) -> None:
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")
