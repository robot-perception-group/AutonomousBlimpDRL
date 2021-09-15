#!/usr/bin/env python
""" blimp goal env navigation task"""
from __future__ import absolute_import, division, print_function
from typing import Tuple, Union

import numpy as np
from blimp_env.envs.vertical_hover_goalenv import VerticalHoverGoalEnv

Observation = Union[np.ndarray, float]


class VerticalHoverGoal2ActEnv(VerticalHoverGoalEnv):
    """a simple one dimensional z control task but only allowed to
    move only up and down"""

    @classmethod
    def default_config(cls) -> dict:  # pylint: disable=duplicate-code
        config = super().default_config()

        config["simulation"].update({"task": "vertical_upward"})
        config["observation"].update(
            {
                "type": "KinematicsGoal",
                "action_feedback": True,
                "goal_obs_diff_feedback": True,
            }
        )
        config["action"].update(
            {
                "type": "SimpleDiscreteMetaHoverAction",
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
