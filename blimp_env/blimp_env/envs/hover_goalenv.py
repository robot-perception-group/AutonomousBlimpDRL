#!/usr/bin/env python
""" blimp goal env navigation task"""
from __future__ import absolute_import, division, print_function
from typing import Tuple, Union

import numpy as np
from .navigate_goalenv import NavigateGoalEnv

Observation = Union[np.ndarray, float]


class HoverGoalEnv(NavigateGoalEnv):
    """hover blimp to a nearby position goal"""

    @classmethod
    def default_config(cls) -> dict:  # pylint: disable=duplicate-code
        config = super().default_config()
        config["simulation"].update({"task": "hover_goal"})
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
                "reward_weights": (1, 0),
                "reward_scale": (10, 1.5),  # track, act
                "track_reward_weights": {
                    "euler": np.array([0.27, 0.27, 0.26, 0.06, 0.06, 0.08]),
                    "quaternion": np.array([0.27, 0.27, 0.26, 0.2]),
                },
                "success_goal_reward": 0.85,
            }
        )

        return config
