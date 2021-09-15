""" navigate env with velocity target """
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
from typing import Dict, Tuple, Union

import numpy as np
from blimp_env.envs.common.abstract import ROSAbstractEnv
from blimp_env.envs.common.action import Action
from geometry_msgs.msg import Point

Observation = Union[np.ndarray, float]


class NavigateEnv(ROSAbstractEnv):
    """Navigate blimp by velocity and orientaion from a path planner"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update({"task": "navigate"})
        config["observation"].update(
            {
                "type": "Kinematics",
            }
        )
        config["action"].update(
            {
                "type": "ContinuousAction",
            }
        )
        config["target"].update(
            {
                "type": "PATH",
                "target_name_space": "target_",
                "orientation_type": "euler",
            }
        )

        config.update(
            {
                "duration": 400,
                "reward_weights": np.array([0.9, 0.1]),
                "reward_scale": (1, 1),
            }
        )

        return config

    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        self.step_info.update({"step": self.steps})

        self._simulate(action)
        obs = self.process_obs_and_goal()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)
        self._update_goal()

        self.step_info.update({"terminal": terminal, "info": info})
        self.reward_publisher.publish(Point(*self.step_info["reward_info"]))

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

        processed = []
        for obs_k, obs_v in obs_info.items():
            if obs_k in goal_info:
                diff_v = self._compute_diff_feature(obs_k, goal_info, obs_info)
                processed.extend(diff_v)
            else:
                processed.extend(obs_v)

        processed = np.array(processed)

        self.step_info.update(
            {
                "process_obs": processed,
                "obs_info": obs_info,
                "goal_info": goal_info,
            }
        )

        return processed

    def _update_goal(self):
        """sample new goal"""
        self.goal = self.target_type.sample()

    def _reward(self, action: np.array) -> float:
        """calculate reward
        total_reward = tracking_reward + action_reward

        Args:
            action (np.array): [action taken by agent]

        Returns:
            float: [reward]
        """
        goal_info: dict
        goal_info = self.step_info["goal_info"]
        obs_info = self.step_info["obs_info"]
        track_scale, act_scale = self.config["reward_scale"]

        tracking_reward = 0
        for feature, _ in goal_info.items():
            diff = self._compute_diff_feature(feature, goal_info, obs_info)
            diff = np.linalg.norm(diff) / np.sqrt(len(diff))
            tracking_reward += 1 / len(goal_info) * np.exp(-track_scale * diff)

        action_reward = self.action_type.action_rew(act_scale)

        reward = np.dot(self.config["reward_weights"], (tracking_reward, action_reward))

        reward_info = (reward, tracking_reward, action_reward)
        self.step_info.update({"reward": reward, "reward_info": reward_info})

        return reward

    def _compute_diff_feature(
        self, feature: str, goal_dic: dict, obs_dic: dict
    ) -> np.ndarray:
        """compute differences in features and scaled to (0,1)
        Usually each feature has range (-1,1). Differences of two vector has range(-2,2)
        Therefore need to be devided by two.

        Args:
            feature ([str]): [feature shared between goal and observations]
            goal_dic ([dict]): [include goal information]
            obs_dic ([dict]): [include observation information]

        Returns:
            [np.ndarray]: [vec differences for each component ranges(-1,1)],
        """
        if feature == "angle":
            vec_diff = self.data_processor.angle_diff(
                goal_dic[feature], obs_dic[feature]
            )
        else:
            vec_diff = (goal_dic[feature] - obs_dic[feature]) / 2

        return vec_diff

    def _is_terminal(self) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["robotID"] == "0":  # only call terminal in first environment
            if self.config["duration"] is not None:
                time = self.steps >= int(self.config["duration"])

        return time

    def get_available_actions(self) -> None:
        pass

    def close(self) -> None:
        pass
