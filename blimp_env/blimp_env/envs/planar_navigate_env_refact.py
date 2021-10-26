""" navigate env with velocity target """
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from blimp_env.envs.common.abstract import ROSAbstractEnv
from blimp_env.envs.common.action import Action
from geometry_msgs.msg import Point, Quaternion

Observation = Union[np.ndarray, float]


class PlanarNavigateEnv(ROSAbstractEnv):
    """Navigate blimp by path following decomposed to altitude and planar control"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update({"task": "navigate_goal"})
        config["observation"].update(
            {
                "type": "PlanarKinematics",
                "noise_stdv": 0.02,
                "action_feedback": True,
                "enable_velocity_goal": True,
            }
        )
        config["action"].update(
            {
                "type": "SimpleContinuousDifferentialAction",
                "act_noise_stdv": 0.05,
            }
        )
        config["target"].update(
            {
                "type": "PlanarGoal",
                "name_space": "machine_",
                "target_name_space": "goal_",
                "enable_velocity_goal": True,
            }
        )
        config.update(
            {
                "duration": 400,
                "simulation_frequency": 10,  # [hz]
                "policy_frequency": 2,
                "reward_weights": np.array(
                    [1, 0.95, 0.05]
                ),  # success, tracking, action
                "tracking_reward_weights": np.array(
                    [0.20, 0.20, 0.30, 0.30]
                ),  # z_diff, planar_dist, psi_diff, vel_diff
                "reward_scale": np.array([1, 1]),
                "success_threshhold": 0.05,  # scaled distance, i.e. 200*threshold meters
            }
        )
        return config

    def _create_pubs_subs(self):
        super()._create_pubs_subs()

        self.planar_angle_cmd_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/planar_ang_cmd", Point, queue_size=1
        )
        self.reward_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward", Quaternion, queue_size=1
        )
        self.diff_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_diff", Quaternion, queue_size=1
        )
        self.goal_pos_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_goal_pos", Point, queue_size=1
        )
        self.vel_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel", Point, queue_size=1
        )
        self.u_velocity_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel_u", Point, queue_size=1
        )
        self.act_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_action", Quaternion, queue_size=1
        )

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

        self._update_goal()

        return obs, reward, terminal, info

    def reset(self) -> Observation:
        self.time = self.steps = 0
        self.done = False
        self._reset()
        obs = self.process_obs_and_goal()
        return obs

    def process_obs_and_goal(self):
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

        tracking_reward = self.compute_tracking_rew(-np.abs(obs[0:3]), track_weights)
        action_reward = self.action_type.action_rew(self.config["reward_scale"][1])

        reward = np.dot(
            self.config["reward_weights"],
            (success_reward, tracking_reward, action_reward),
        )
        reward = np.clip(reward, -1, 1)

        reward_info = (success_reward, tracking_reward, action_reward)
        self.step_info.update({"reward": reward, "reward_info": reward_info})

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

    def _is_terminal(self) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"])

        return time

    def get_available_actions(self) -> None:
        pass

    def close(self) -> None:
        pass


class PlanarNavigateEnv2(PlanarNavigateEnv):
    """
    similar to v1: state space, goal space

    different from v1:
    - include velocity command
    - refined reward function
    - continuous action space
    - only navigation task (no hover)

    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update({"task": "navigate_goal"})
        config["observation"].update(
            {
                "type": "PlanarKinematics",
                "noise_stdv": 0.02,
                "action_feedback": True,
                "enable_velocity_goal": True,
            }
        )
        config["action"].update(
            {
                "type": "SimpleContinuousDifferentialAction",
                "act_noise_stdv": 0.05,
            }
        )
        config["target"].update(
            {
                "type": "PlanarGoal",
                "name_space": "machine_",
                "target_name_space": "goal_",
                "enable_velocity_goal": True,
            }
        )
        config.update(
            {
                "duration": 400,  # [time steps]
                "simulation_frequency": 10,  # [hz]
                "policy_frequency": 2,
                "reward_weights": np.array(
                    [1.0, 0.9, 0.1]
                ),  # success, tracking, action
                "tracking_reward_weights": np.array(
                    [0.20, 0.20, 0.30, 0.30]
                ),  # z_diff, planar_dist, psi_diff, vel_diff
                "reward_scale": np.array([1, 1]),
                "success_threshhold": 0.05,  # scaled distance, i.e. 200*threshold meters
            }
        )
        return config

    def _create_pubs_subs(self):
        super()._create_pubs_subs()
        self.reward_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward", Quaternion, queue_size=1
        )
        self.diff_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_diff", Quaternion, queue_size=1
        )
        self.goal_pos_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_goal_pos", Point, queue_size=1
        )
        self.vel_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel", Point, queue_size=1
        )
        self.u_velocity_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel_u", Point, queue_size=1
        )
        self.act_rviz_publisher = rospy.Publisher(
            self.config["name_space"] + "/rviz_action", Quaternion, queue_size=1
        )

    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        self.step_info.update({"step": self.steps})

        self._simulate(action)
        obs = self.process_obs_and_goal()
        reward = self._reward(obs)
        terminal = self._is_terminal()
        info = self._info(obs, action)
        self.step_info.update({"terminal": terminal, "info": info})

        self._update_goal()
        self.reward_rviz_publisher.publish(Quaternion(*self.step_info["reward_info"]))
        self.act_rviz_publisher.publish(Quaternion(*action))

        if self.dbg:
            print(
                f"================= [ navigate_env ] step {self.steps} ================="
            )
            print("STEP INFO:", self.step_info)
            print("\r")

        return obs, reward, terminal, info

    def process_obs_and_goal(self):
        """process the difference between observation and goal

        Returns:
            [np.array]: [process observation]
        """
        _, goal_info = self.goal
        _, obs_info = self.observation_type.observe()

        obs_pos, goal_pos = obs_info["position"], goal_info["position"]
        z_diff = obs_pos[2] - goal_pos[2]
        planar_dist = np.linalg.norm((obs_pos[0:2] - goal_pos[0:2])) / np.sqrt(2)

        xy_vel, z_vel, goal_xy_vel = (
            obs_info["velocity"][0:2],
            obs_info["velocity"][2],
            goal_info["velocity"][0:2],
        )
        # convert to body frame with assumption, u~xy_vel, and omit velocity direction
        # since we only care about positional direction
        u_vel = np.linalg.norm(xy_vel) / np.sqrt(2)
        goal_u_vel = np.linalg.norm(goal_xy_vel) / np.sqrt(2)
        u_diff = u_vel - goal_u_vel

        psi_diff = self.compute_psi_diff(goal_pos, obs_pos, obs_info["angle"][2])

        act = obs_info["action"]
        obs = np.array([z_diff, planar_dist, psi_diff, u_diff, u_vel])
        processed = np.concatenate([obs, act])
        processed = np.clip(processed, -1, 1)

        self.step_info.update(
            {
                "process_obs": processed,
                "obs_info": obs_info,
                "goal_info": goal_info,
            }
        )

        self.diff_rviz_publisher.publish(
            Quaternion(planar_dist, psi_diff, z_diff, u_diff)
        )
        self.vel_rviz_publisher.publish(Point(*obs_info["velocity"]))
        self.u_velocity_rviz_publisher.publish(Point(u_vel, goal_u_vel, u_diff))
        return processed

    def _reward(self, obs: np.array) -> float:  # pylint: disable=arguments-differ
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - psi angle difference
        action_reward: penalty for motor use

        Args:
            obs (np.array): [z_diff, planar_dist, psi_diff, u_diff, z_vel, u_vel, act]

        Returns:
            float: [reward]
        """
        goal_info: dict
        goal_info = self.step_info["goal_info"]
        obs_info = self.step_info["obs_info"]

        track_weights = self.config["tracking_reward_weights"].copy()
        reward_weights = self.config["reward_weights"].copy()
        reward_scale = self.config["reward_scale"].copy()

        success_reward = self.compute_success_rew(
            obs_info["position"], goal_info["position"], k=2
        )
        tracking_reward = self.compute_tracking_rew(-np.abs(obs[0:4]), track_weights)
        action_reward = self.action_type.action_rew(reward_scale[1])

        reward = np.dot(
            reward_weights,
            (success_reward, tracking_reward, action_reward),
        )
        reward = np.clip(reward, -1, 1)

        reward_info = (reward, success_reward, tracking_reward, action_reward)
        self.step_info.update({"reward": reward, "reward_info": reward_info})

        return reward

    def _is_terminal(self) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"])

        goal_info = self.step_info["goal_info"]
        obs_info = self.step_info["obs_info"]
        success_reward = self.compute_success_rew(
            obs_info["position"], goal_info["position"], k=2
        )
        success = success_reward >= 1

        return time or success
