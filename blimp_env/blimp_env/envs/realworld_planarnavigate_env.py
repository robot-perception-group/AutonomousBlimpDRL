""" navigate env with velocity target """
#!/usr/bin/env python

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
from blimp_env.envs.common.abstract import AbstractEnv
from blimp_env.envs.common.action import Action
from blimp_env.envs.common.controllers_connection import ControllersConnection
from blimp_env.envs.common.data_processor import DataProcessor
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.common.utils import update_dict
from blimp_env.envs.script.blimp_script import (
    spawn_simulation_on_different_port,
    spawn_simulation_on_marvin,
)
import std_msgs.msg
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Pose
from gym.utils import seeding
import tf

Observation = Union[np.ndarray, float]
Target = Union[np.ndarray, float]


class RealWorldPlanarNavigateEnv(AbstractEnv):
    """Navigate blimp by path following decomposed to altitude and planar control"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "robot_id": "0",
                "name_space": "machine_",
                "real_experiment": True,
                "DBG": True,
                "simulation": {
                    "robot_id": "0",
                    "ros_ip": "localhost",
                    "ros_port": 11311,
                    "gaz_port": 11351,
                    "gui": False,
                    "enable_meshes": False,
                    "world": "basic",
                    "task": "square",
                    "auto_start_simulation": False,
                    "update_robotID_on_workerID": True,
                },
                "observation": {
                    "type": "RealPlanarKinematics",
                    "name_space": "machine_",
                    "orientation_type": "euler",
                    "real_experiment": True,
                    "action_feedback": True,
                    "DBG_ROS": False,
                    "DBG_OBS": False,
                },
                "action": {
                    "type": "RealDiscreteMetaAction",
                    "robot_id": "0",
                    "name_space": "machine_",
                    "flightmode": 3,
                    "DBG_ACT": False,
                    "act_noise_stdv": 0.0,
                },
                "target": {
                    "type": "PlanarGoal",
                    "name_space": "machine_",
                    "target_name_space": "goal_",
                    "real_experiment": True,
                    "orientation_type": "euler",
                    "DBG_ROS": False,
                },
                "seed": 123,
                "simulation_frequency": 10,
                "policy_frequency": 2,
                "duration": None,
            }
        )
        config.update(
            {
                "reward_weights": np.array(
                    [1, 0.95, 0.05]
                ),  # success, tracking, action
                "tracking_reward_weights": np.array(
                    [0.1, 0.7, 0.2]
                ),  # z_diff, planar_dist, psi_diff
                "reward_scale": (1, 1),
                "success_threshhold": 0.15,
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
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

        self.rate = rospy.Rate(self.config["simulation_frequency"])
        self.data_processor = DataProcessor()

        self.define_spaces()

        print(self.observation_space)
        print(self.target_space)
        print(self.action_space)

        self.step_info: Dict = {}
        self.dbg = self.config["DBG"]

        self._create_pubs_subs()

        if self.config["real_experiment"] is False:
            self.gaz = GazeboConnection(
                start_init_physics_parameters=True, reset_world_or_sim="WORLD"
            )
            self.controllers_object = ControllersConnection(
                namespace=self.config["name_space"]
            )
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
        if worker_index >= 8:
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
        self.planar_angle_cmd_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/planar_ang_cmd", Point, queue_size=1
        )

    def reset(self) -> Observation:
        self.time = self.steps = 0
        self.done = False
        self._reset()
        obs = self.process_obs_and_goal()
        return obs

    def _reset(self):
        self._update_goal()
        if self.config["real_experiment"] is False:
            self._reset_gazebo()
        else:
            self.action_type.realworld_set_init_pose()

    def _reset_gazebo(self):
        self.gaz.pause_sim()
        self.gaz.reset_sim()
        self.gaz.unpause_sim()
        self._check_system_ready()

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
        obs, reward, terminal, info = self.one_step(action)

        return obs, reward, terminal, info

    def one_step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        """one agent step

        Args:
            action (Action): action chosen by the agent

        Returns:
            Tuple[Observation, float, bool, dict]: a tuple of step informations
        """
        self.step_info.update({"step": self.steps})

        self._simulate(action)
        obs = self.process_obs_and_goal()
        reward = self._reward(obs)
        terminal = False
        info = self._info(obs, action)
        self.step_info.update({"terminal": terminal, "info": info})

        self._update_goal()

        if self.dbg:
            print(
                f"================= [ navigate_env ] step {self.steps} ================="
            )
            print("STEP INFO:", self.step_info)
            print("\r")

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

    def process_obs_and_goal(self) -> np.ndarray:
        """process the difference between observation and goal

        Returns:
            [np.array]: [process observation]
        """
        _, goal_info = self.goal
        _, obs_info = self.observation_type.observe()

        obs_pos, goal_pos = obs_info["position"], goal_info["position"]
        z_diff = obs_pos[2] - goal_pos[2]
        z_vel = obs_info["velocity"][2]
        velocity = np.linalg.norm(obs_info["velocity"]) / np.sqrt(3)
        planar_dist = np.linalg.norm((obs_pos[0:2] - goal_pos[0:2])) / np.sqrt(2)
        psi_diff = self.compute_psi_diff(goal_pos, obs_pos, obs_info["angle"][2])

        act = obs_info["action"]

        feature_list = [z_diff, planar_dist, psi_diff, z_vel, velocity, act]
        processed = np.empty(0)
        for feature in feature_list:
            processed = np.append(processed, feature)
        processed = np.clip(processed, -1, 1)

        self.step_info.update(
            {
                "process_obs": processed,
                "obs_info": obs_info,
                "goal_info": goal_info,
            }
        )
        self.planar_angle_cmd_rviz_publisher.publish(Point(0, 0, psi_diff))

        return processed

    def _update_goal(self):
        """sample new goal"""
        self.goal = self.target_type.sample()

    def _reward(self, obs: np.array) -> float:  # pylint: disable=arguments-renamed
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - psi angle difference
        action_reward: penalty for motor use

        Args:
            action (np.array): []

        Returns:
            float: [reward]
        """
        goal_info: dict
        goal_info = self.step_info["goal_info"]
        obs_info = self.step_info["obs_info"]

        success_reward = self.compute_success_rew(
            obs_info["position"], goal_info["position"]
        )
        track_weights = self.config["tracking_reward_weights"].copy()
        if success_reward == 1:
            track_weights[0] += track_weights[2]
            track_weights[2] = 0

        tracking_reward = self.compute_tracking_rew(
            -np.abs(obs[0:3]), self.config["tracking_reward_weights"]
        )
        action_reward = self.action_type.action_rew(self.config["reward_scale"][1])

        reward = np.dot(
            self.config["reward_weights"],
            (success_reward, tracking_reward, action_reward),
        )
        reward = np.clip(reward, -1, 1)

        reward_info = (success_reward, tracking_reward, action_reward)
        self.step_info.update({"reward": reward, "reward_info": reward_info})
        self.reward_publisher.publish(Point(*self.step_info["reward_info"]))
        return reward

    @classmethod
    def compute_success_rew(
        cls, pos: np.array, goal_pos: np.array, k: float = 1
    ) -> float:
        """task success if distance to goal is less than sucess_threshhold

        Args:
            pos ([np.array]): [position of machine]
            goal_pos ([np.array]): [position of planar goal]

        Returns:
            [float]: [1 if success, otherwise 0]
        """
        config = cls.default_config()
        return (
            1.0
            if np.linalg.norm(pos - goal_pos) / np.sqrt(3)
            < k * config["success_threshhold"]
            else 0.0
        )

    @classmethod
    def compute_tracking_rew(cls, obs_diff: tuple, weights: tuple) -> float:
        """compute tracking reward

        Args:
            obs_diff (tuple): [observed differences to goal]
            weights (tuple): [weight to each observation difference]

        Returns:
            float: [tracking reward close to 0 if obs_diff is small]
        """
        return np.dot(weights, obs_diff)

    @classmethod
    def compute_psi_diff(
        cls, goal_pos: np.array, obs_pos: np.array, obs_psi: float
    ) -> float:
        """compute psi angle of the vector machine position to goal position
        then compute the difference of this angle to machine psi angle
        last, make sure this angle lies within (-pi, pi) and scale to (-1,1)

        Args:
            goal_pos (np.array): [machine position]
            obs_pos (np.array): [goal postiion]
            obs_psi (float): [machine psi angle]

        Returns:
            float: [psi angle differences]
        """
        pos_diff = obs_pos - goal_pos
        scaled_goal_psi = np.arctan2(pos_diff[1], pos_diff[0]) / np.pi - 1
        ang_diff = scaled_goal_psi - obs_psi

        if ang_diff > 1:
            ang_diff -= 2
        elif ang_diff < -1:
            ang_diff += 2

        return ang_diff

    def close(self):
        return NotImplementedError

    def _cost(self, action: Action):
        return NotImplementedError

    def _is_terminal(self):
        return NotImplementedError

    def get_available_actions(self):
        return NotImplementedError

    def render(self):
        return NotImplementedError
