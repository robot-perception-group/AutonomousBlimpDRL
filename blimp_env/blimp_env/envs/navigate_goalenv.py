#!/usr/bin/env python
""" blimp goal env navigation task"""
from __future__ import absolute_import, division, print_function
from typing import Tuple, Union

import numpy as np
from blimp_env.envs.common.abstract import ROSAbstractEnv
from blimp_env.envs.common.action import Action
from gym import GoalEnv
from geometry_msgs.msg import Point

Observation = Union[np.ndarray, float]


class NavigateGoalEnv(ROSAbstractEnv, GoalEnv):
    """navigate blimp to a position goal"""

    @classmethod
    def default_config(cls) -> dict:  # pylint: disable=duplicate-code
        config = super().default_config()
        config["simulation"].update({"task": "navigate_goal"})
        config["observation"].update(
            {
                "type": "KinematicsGoal",
                "orientation_type": "euler",
                "action_feedback": True,
                "goal_obs_diff_feedback": True,
            }
        )
        config["action"].update(
            {
                "type": "DiscreteMetaAction",
            }
        )
        config["target"].update(
            {
                "type": "GOAL",
                "target_name_space": "goal_",
                "orientation_type": "euler",
            }
        )
        config.update(
            {
                "reward_weights": (1, 0),
                "reward_scale": (10, 1.5),  # track, act
                "track_reward_weights": {
                    "euler": np.array([0.27, 0.27, 0.26, 0.06, 0.06, 0.08]),
                    "quaternion": np.array([0.27, 0.27, 0.26, 0.2, 0, 0, 0]),
                },
                "success_goal_reward": 0.85,
            }
        )
        return config

    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        self._simulate(action)
        obs, obs_info = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)
        self._update_goal()

        self.step_info.update(
            {
                "step": self.steps,
                "obs": obs,
                "obs_info": obs_info,
                "goal": self.goal[0],
                "goal_info": self.goal[1],
                "reward": reward,
                "terminal": terminal,
                "info": info,
            }
        )

        self.reward_publisher.publish(
            Point(
                self.step_info["total_rew"],
                self.step_info["track_rew"],
                self.step_info["act_rew"],
            )
        )

        if self.dbg:

            print(
                f"================= [ navigate_goalenv ] step {self.steps} ================="
            )
            print("STEP INFO:", self.step_info)
            print("\r")

        return obs, reward, terminal, info

    def _update_goal(self) -> None:
        self.goal = self.target_type.sample()

    def compute_goal_diff(self, desired_goal, achieved_goal):
        """compute differences between achieved goal and desired goal

        Args:
            achieved_goal (np.ndarray): [robots position and orientation]
            desired_goal (np.ndarray): [goal position and orientation]

        Returns:
            float: [difference between goal]
        """

        goal_diff = (desired_goal - achieved_goal) / 2

        ori_type = self.config["target"]["orientation_type"]
        if ori_type == "euler":
            goal_diff[3:6] = self.data_processor.angle_diff(
                desired_goal[3:6], achieved_goal[3:6]
            )
        elif ori_type == "quaternion":
            goal_diff[3] = self.data_processor.quat_diff(
                desired_goal[3:7], achieved_goal[3:7]
            )
            goal_diff[4:7] = 0
        else:
            raise ValueError("unkonwn target orientation type")

        return goal_diff

    def compute_reward(  # pylint: disable=arguments-differ
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        # info: dict,
    ) -> float:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - psi angle difference
        action_reward: penalty for motor use

        Args:
            achieved_goal (np.ndarray): achieved goal or current state
            desired_goal (np.ndarray): target state
            info (dict): addition parameters

        Returns:
            float: [reward]
        """
        goal_diff = self.compute_goal_diff(desired_goal, achieved_goal)

        ori_type = self.config["target"]["orientation_type"]
        track_rew = np.exp(
            -self.config["reward_scale"][0]
            * np.dot(np.abs(goal_diff), self.config["track_reward_weights"][ori_type])
        )
        act_rew = 0
        total_rew = (
            self.config["reward_weights"][0] * track_rew
            + self.config["reward_weights"][1] * act_rew
        )

        self.step_info.update(
            {"total_rew": total_rew, "track_rew": track_rew, "act_rew": act_rew}
        )
        return total_rew

    def _reward(self, action: np.ndarray) -> float:
        obs, _ = self.observation_type.observe()

        total_rew = self.compute_reward(
            obs["achieved_goal"],
            obs["desired_goal"],
            {"action": action},
        )
        return total_rew

    def _is_success(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> bool:
        rew = self.compute_reward(achieved_goal, desired_goal, info)
        return rew > self.config["success_goal_reward"]

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        success = self._is_success(obs["achieved_goal"], obs["desired_goal"], info)
        info.update({"is_success": success})
        return info

    def _is_terminal(self) -> bool:
        time = success = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"])

        obs, _ = self.observation_type.observe()
        position = self.far_from_goal(
            obs["achieved_goal"][0:3], obs["desired_goal"][0:3]
        )
        success = self._is_success(obs["achieved_goal"], obs["desired_goal"], {})

        return bool(time or success or position)

    @classmethod
    def far_from_goal(cls, position: np.ndarray, goal_position: np.ndarray) -> bool:
        """detect if current position too far from the goal

        Args:
            position ([np.ndarray]): [current position (-1,1)]
            goal_position ([np.ndarray]): [position of the goal ranges (-1,1)]

        Returns:
            [bool]: [whether current position too far from goal]
        """
        return np.any(np.abs(position - goal_position) > 1)

    def close(self) -> None:
        pass

    def get_available_actions(self):
        pass
