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
                "scale_obs": True,
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
                "target_name_space": "goal_",
            }
        )
        config.update(
            {
                "duration": 2000,
                "simulation_frequency": 50,  # [hz]
                "policy_frequency": 10,
                "reward_weights": np.array(
                    [1, 0.95, 0.05]
                ),  # success, tracking, action
                "tracking_reward_weights": np.array(
                    [0.20, 0.20, 0.4, 0.20]
                ),  # z_diff, planar_dist, psi_diff, vel_diff
                "success_threshhold": 10,  # [meters]
            }
        )
        return config

    def _create_pubs_subs(self):
        super()._create_pubs_subs()
        self.rew_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward", Quaternion, queue_size=1
        )
        self.state_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_state", Quaternion, queue_size=1
        )
        self.vel_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel", Point, queue_size=1
        )
        self.vel_diff_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel_diff", Point, queue_size=1
        )
        self.ang_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang", Point, queue_size=1
        )
        self.ang_diff_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang_diff", Point, queue_size=1
        )
        self.act_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_act", Quaternion, queue_size=1
        )
        self.pos_cmd_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_pos_cmd",
            Point,
            queue_size=1,
        )

    # @profile
    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """[perform a step action and observe result]

        Args:
            action (Action): action from the agent [-1,1] with size (4,)

        Returns:
            Tuple[Observation, float, bool, dict]:
                obs: np.array [-1,1] with size (9,),
                reward: scalar,
                terminal: bool,
                info: dictionary of all the step info,
        """
        self._simulate(action)
        obs, obs_info = self.observation_type.observe()
        reward, reward_info = self._reward(obs, action, obs_info)
        terminal = self._is_terminal(obs_info)
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "reward": reward,
            "reward_info": reward_info,
            "terminal": terminal,
        }

        self._update_goal()
        self._step_info(info)

        return obs, reward, terminal, info

    def _step_info(self, info: dict):
        """publish all the step information to rviz

        Args:
            info ([dict]): [dict contain all step information]
        """
        obs_info = info["obs_info"]
        proc_info = obs_info["proc_dict"]

        self.rew_rviz_pub.publish(Quaternion(*info["reward_info"]))
        self.state_rviz_pub.publish(
            Quaternion(
                proc_info["planar_dist"],
                proc_info["psi_diff"],
                proc_info["z_diff"],
                proc_info["vel_diff"],
            )
        )
        self.vel_rviz_pub.publish(Point(*obs_info["velocity"]))
        self.vel_diff_rviz_pub.publish(
            Point(
                proc_info["vel"],
                self.goal["velocity"],
                proc_info["vel_diff"],
            )
        )
        self.ang_rviz_pub.publish(Point(*obs_info["angle"]))
        self.ang_diff_rviz_pub.publish(Point(0, 0, proc_info["psi_diff"]))
        self.act_rviz_pub.publish(Quaternion(*info["act"]))

        self.pos_cmd_pub.publish(Point(*self.goal["position"]))

        if self.dbg:
            print(
                f"================= [ PlanarNavigateEnv ] step {self.steps} ================="
            )
            print("STEP INFO:", info)
            print("\r")

    def reset(self) -> Observation:
        self.steps = 0
        self.done = False
        self._reset()
        obs, _ = self.observation_type.observe()
        return obs

    def _update_goal(self):
        """sample new goal dictionary"""
        self.goal = self.target_type.sample()

    def _reward(
        self, obs: np.array, act: np.array, obs_info: dict
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - psi angle difference
        action_reward: penalty for motor use

        Args:
            obs (np.array): ("z_diff", "planar_dist", "psi_diff", "vel_diff", "vel", "action")
            act (np.array): agent action [-1,1] with size (4,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        track_weights = self.config["tracking_reward_weights"].copy()
        reward_weights = self.config["reward_weights"].copy()

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )
        tracking_reward = np.dot(track_weights, -np.abs(obs[0:4]))
        action_reward = self.action_type.action_rew()

        reward = np.dot(
            reward_weights,
            (success_reward, tracking_reward, action_reward),
        )
        reward = np.clip(reward, -1, 1)
        reward_info = (reward, success_reward, tracking_reward, action_reward)

        return reward, reward_info

    def compute_success_rew(self, pos: np.array, goal_pos: np.array) -> float:
        """task success if distance to goal is less than sucess_threshhold

        Args:
            pos ([np.array]): [position of machine]
            goal_pos ([np.array]): [position of planar goal]
            k (float): scaler for success

        Returns:
            [float]: [1 if success, otherwise 0]
        """
        return (
            1.0
            if np.linalg.norm(pos - goal_pos) < self.config["success_threshhold"]
            else 0.0
        )

    def _is_terminal(self, obs_info: dict) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"])

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )
        success = success_reward >= 1

        return time or success


if __name__ == "__main__":
    import copy
    from blimp_env.envs.common.gazebo_connection import GazeboConnection

    # ============== profile ==============#
    # 1. pip install line_profiler
    # 2. in terminal run the command:
    # kernprof -l -v blimp_env/envs/planar_navigate_env.py
    ENV = PlanarNavigateEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "auto_start_simulation": False,
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": True,
        },
        "action": {
            "DBG_ACT": True,
        },
        "target": {"DBG_ROS": True},
    }

    @profile
    def env_step():
        env = ENV(copy.deepcopy(env_kwargs))
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminal, info = env.step(action)

        GazeboConnection().unpause_sim()

    env_step()
