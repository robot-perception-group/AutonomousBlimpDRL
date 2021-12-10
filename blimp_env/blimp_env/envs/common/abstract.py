#!/usr/bin/env python
""" environment abstract """
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
from blimp_env.envs.common.action import Action, ActionType, action_factory
from blimp_env.envs.common.controllers_connection import ControllersConnection
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.common.observation import ObservationType, observation_factory
from blimp_env.envs.common.target import TargetType, target_factory
from blimp_env.envs.common.utils import update_dict
from blimp_env.envs.script.blimp_script import (
    spawn_simulation_on_different_port,
    spawn_simulation_on_marvin,
    kill_all_screen,
)
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

        self.steps = 0
        self.done = False

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

    def _reward(self, action: Action, observation: Observation) -> float:
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        raise NotImplementedError

    def reset(self) -> Observation:
        self.steps = 0
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
        info = {"obs": obs, "act": action}

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        n_steps = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for _ in range(n_steps):
            if (action is not None) and (self.steps % n_steps == 0):
                self.action_type.act(action)

    def close(self):
        """close env"""
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
                "ros_port": 11311,
                "gaz_port": 11351,
                "name_space": "machine_",
                "DBG": False,
                "simulation": {
                    "robot_id": "0",
                    "ros_ip": "localhost",
                    "ros_port": 11311,
                    "gaz_port": 11351,
                    "gui": False,
                    "enable_meshes": False,
                    "enable_wind": False,
                    "wind_speed": 2.0,
                    "world": "basic",
                    "auto_start_simulation": True,
                    "remote_host_name": "frg07",
                    "maximum_local_worker": 10,
                },
                "observation": {
                    "type": "PlanarKinematics",
                    "name_space": "machine_",
                    "real_experiment": False,
                    "DBG_ROS": False,
                    "DBG_OBS": False,
                },
                "action": {
                    "type": "SimpleContinuousDifferentialAction",
                    "robot_id": "0",
                    "name_space": "machine_",
                    "flightmode": 3,
                    "DBG_ACT": False,
                },
                "target": {
                    "type": "InteractiveGoal",
                    "target_name_space": "goal_",
                    "robot_id": "0",
                    "ros_port": 11311,
                    "gaz_port": 11351,
                    "DBG_ROS": False,
                },
                "seed": 123,
                "simulation_frequency": 30,
                "policy_frequency": 6,
                "duration": 1200,
            }
        )

        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        # if rllib parallelization, use worker index as robot_id
        if hasattr(config, "worker_index"):
            config["robot_id"] = str(config.worker_index - 1)
            config["seed"] = int(config.worker_index) + 123
        super().__init__(config=config)

        if self.config["simulation"]["auto_start_simulation"]:
            self.setup_env(int(self.config["robot_id"]))

        print(self.config)
        self.dbg = self.config["DBG"]

        rospy.loginfo("[ RL Node " + str(self.config["robot_id"]) + " ] Initialize...")
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

        self.define_spaces()
        self._create_pubs_subs()

        self.gaz.unpause_sim()
        rospy.loginfo("[ RL Node " + str(self.config["robot_id"]) + " ] Initialized")

    def configure(self, config: Optional[Dict[Any, Any]]) -> None:
        if config:
            tmp_config = deepcopy(config)
            tmp_config_ori = deepcopy(self.config)

            self.config.update(tmp_config)
            for key in ["simulation", "observation", "action", "target"]:
                self.config[key].update(tmp_config_ori[key])
                if key in config:
                    self.config[key].update(config[key])

        self._update_shared_param_in_config()

    def _update_shared_param_in_config(self):
        """synchronize parameters across config in all depth"""
        name_space = self.config["name_space"] + str(self.config["robot_id"])
        target_name_space = self.config["target"]["target_name_space"] + str(
            self.config["robot_id"]
        )
        self.config = update_dict(self.config, "robot_id", str(self.config["robot_id"]))
        self.config = update_dict(self.config, "ros_port", int(self.config["ros_port"]))
        self.config = update_dict(self.config, "gaz_port", int(self.config["gaz_port"]))
        self.config = update_dict(self.config, "name_space", name_space)
        self.config = update_dict(self.config, "target_name_space", target_name_space)

    def setup_env(self, worker_index: int = 0):
        """setup gazebo env. Worker index is used to modify env config allowing
        training on different env in parallel. It is important to specify ros and gaz
        master channel to spawn env in parallel.

        Args:
            worker_index (int, optional): [the index of worker or subprocess]. Defaults to 0.
        """
        assert (
            worker_index >= 0
        ), f"worker index has to be larger than 0, index: {worker_index}"

        # spawn env on other pc if exceed maximum local worker
        marvin = worker_index >= self.config["simulation"]["maximum_local_worker"]
        ros_ip = (
            self.config["simulation"]["remote_host_name"]
            if marvin
            else socket.gethostbyname(socket.gethostname())
        )

        time.sleep(
            10 * int(worker_index)
        )  # spawn at different time increase spawn stability
        ros_port = self.config["simulation"]["ros_port"] + worker_index
        gaz_port = self.config["simulation"]["gaz_port"] + worker_index
        host_addr = "http://" + str(ros_ip) + ":"
        os.environ["ROS_MASTER_URI"] = host_addr + str(ros_port) + "/"
        os.environ["GAZEBO_MASTER_URI"] = host_addr + str(gaz_port) + "/"

        while rosgraph.is_master_online():
            time.sleep(2)
            ros_port += 7
            gaz_port += 7
            os.environ["ROS_MASTER_URI"] = host_addr + str(ros_port) + "/"
            os.environ["GAZEBO_MASTER_URI"] = host_addr + str(gaz_port) + "/"
            print("current channel occupied, move to next channel:", ros_port)

        self.config["simulation"].update(
            {"ros_ip": ros_ip, "ros_port": ros_port, "gaz_port": gaz_port}
        )

        self._spawn_sim(marvin)

    def _spawn_sim(self, marvin=False):
        if marvin:
            spawn_simulation_on_marvin(**self.config["simulation"])
        else:
            spawn_simulation_on_different_port(**self.config["simulation"])

    def _create_pubs_subs(self):
        pass

    def reset(self) -> Observation:
        self.steps = 0
        self.done = False
        self._reset()
        obs, _ = self.observation_type.observe()
        return obs

    def _reset(self):
        self._reset_gazebo()
        self._update_goal()

    def _reset_gazebo(self):
        self.gaz.unpause_sim()
        self._reset_joint_and_check_sys()
        self.gaz.pause_sim()

        self.gaz.reset_sim()

        self.gaz.unpause_sim()
        self._reset_joint_and_check_sys()
        self.gaz.pause_sim()

    def _reset_joint_and_check_sys(self):
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
        self.gaz.unpause_sim()
        obs, reward, terminal, info = self.one_step(action)
        self.gaz.pause_sim()

        self.steps += 1

        assert isinstance(reward, (float, int)), "The reward must be a float"
        assert isinstance(terminal, bool), "terminal must be a boolean"
        assert isinstance(info, dict), "info must be a dict"

        return obs, reward, terminal, info

    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """perform a step action and observe result"""
        self._simulate(action)
        obs = self.observation_type.observe()
        reward = self._reward(obs, action)
        terminal = self._is_terminal()

        info = {
            "step": self.steps,
            "obs": obs,
            "act": action,
            "reward": reward,
            "terminal": terminal,
        }

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        n_steps = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for step in range(n_steps):
            if (action is not None) and (step == 0):
                self.action_type.act(action)

            self.rate.sleep()

    def _update_goal(self):
        raise NotImplementedError

    def close(self) -> None:
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")
        kill_reply = kill_all_screen(int(self.config["robot_id"]))
        print("kill screen reply:", kill_reply)
