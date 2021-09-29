#!/usr/bin/env python
""" blimp goal env navigation task"""
from __future__ import absolute_import, division, print_function
from typing import Tuple, Union

import numpy as np
from blimp_env.envs.navigate_goalenv import NavigateGoalEnv

Observation = Union[np.ndarray, float]


def compute_goal_diff(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
) -> float:
    """compute differences between achieved goal and desired goal in position z-axis

    Args:
        achieved_goal (np.ndarray): [robots position and orientation]
        desired_goal (np.ndarray): [goal position and orientation]

    Returns:
        float: [difference between goal, and scaled by 0.5 to stay in range (-1,1)]
    """
    assert achieved_goal.shape == desired_goal.shape

    ndim = desired_goal.ndim
    if ndim == 1:
        desired_z = desired_goal[2]
        achieved_z = achieved_goal[2]
    elif ndim == 2:
        desired_z = desired_goal[:, 2]
        achieved_z = achieved_goal[:, 2]

    return (desired_z - achieved_z) / 2


class VerticalHoverGoalEnv(NavigateGoalEnv):
    """a simple one dimensional z control task"""

    @classmethod
    def default_config(cls) -> dict:  # pylint: disable=duplicate-code
        config = super().default_config()
        config["simulation"].update({"task": "vertical_hover_goal"})
        config["observation"].update(
            {
                "type": "KinematicsGoal",
                "action_feedback": True,
                "goal_obs_diff_feedback": True,
            }
        )
        config["action"].update(
            {
                "type": "DiscreteMetaHoverAction",
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
                "duration": 200,
                "reward_type": None,  # "sparse",
                "reward_weights": (1, 0),
                "reward_scale": (10, 1.5),
                "success_goal_reward": 0.9,
            }
        )

        return config

    def compute_reward(  # pylint: disable=arguments-differ
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        # info: dict,
    ) -> float:

        goal_diff = compute_goal_diff(achieved_goal, desired_goal)

        track_rew = np.exp(-self.config["reward_scale"][0] * np.abs(goal_diff))
        act_rew = 0

        total_rew = (
            self.config["reward_weights"][0] * track_rew
            + self.config["reward_weights"][1] * act_rew
        )

        if self.config["reward_type"] == "sparse":
            total_rew = (total_rew > self.config["success_goal_reward"]).astype(
                np.float32
            ) + 0.0

        self.step_info.update(
            {
                "total_rew": total_rew,
                "track_rew": track_rew,
                "act_rew": act_rew,
            }
        )
        return total_rew
