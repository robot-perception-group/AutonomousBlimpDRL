""" aerobatic env  """
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from blimp_env.envs.common.abstract import ROSAbstractEnv
from blimp_env.envs.common.action import Action
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Float32MultiArray
from rotors_comm.msg import WindSpeed
from blimp_env.envs.script import close_simulation, change_buoynacy
import line_profiler
import copy

profile = line_profiler.LineProfiler()

Observation = Union[np.ndarray, float]


class AerobaticEnv(ROSAbstractEnv):
    """aerobatic movement"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update(
            {
                "enable_wind": False,
                "enable_wind_sampling": False,
                "wind_speed": 2.0,
                "enable_buoyancy_sampling": False,
                "buoyancy_range": [0.9, 1.1],
            }
        )
        config["observation"].update(
            {
                "type": "AerobaticObservation",
                "noise_stdv": 0.0,
                "scale_obs": True,
                "enable_airspeed_sensor": False,
            }
        )
        config["action"].update(
            {
                "type": "ContinuousDifferentialAction",
                "act_noise_stdv": 0.0,
                "disable_servo": True,
                "max_servo": -0.5,
                "max_thrust": 0.5,
            }
        )
        config["target"].update(
            {
                "type": "AerobaticGoal",
                "target_name_space": "goal_",
                "aerobatic_name": "vertical_roll",
            }
        )
        config.update(
            {
                "duration": 1200,
                "simulation_frequency": 30,  # [hz]
                "policy_frequency": 10,  # [hz] has to be greater than 5 to overwrite backup controller
                "reward_weights": np.array(
                    [100, 0.8, 0.2]
                ),  # success, tracking, action
                "tracking_reward_weights": np.array(
                    [0.25, 0.25, 0.25, 0.25]
                ),  # z_diff, planar_dist, yaw_diff, vel_diff
                "success_threshhold": 5,  # [meters]
            }
        )
        return config

    def _create_pub_and_sub(self):
        super()._create_pub_and_sub()
        self.rew_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward", Float32MultiArray, queue_size=1
        )

    @profile
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
        reward, reward_info = self._reward(obs.copy(), action, copy.deepcopy(obs_info))
        terminal = self._is_terminal(copy.deepcopy(obs_info))
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "reward": reward,
            "reward_info": reward_info,
            "terminal": terminal,
        }

        self._update_goal_and_env()
        self._step_info(info)

        return obs, reward, terminal, info

    def _step_info(self, info: dict):
        """publish all the step information to rviz

        Args:
            info ([dict]): [dict contain all step information]
        """
        obs_info = info["obs_info"]
        proc_info = obs_info["proc_dict"]

        self.rew_rviz_pub.publish(
            Float32MultiArray(data=np.array(info["reward_info"]["rew_info"]))
        )
        self.state_rviz_pub.publish(
            Quaternion(
                proc_info["planar_dist"],
                proc_info["yaw_diff"],
                proc_info["z_diff"],
                proc_info["vel_diff"],
            )
        )
        self.vel_rviz_pub.publish(Point(*obs_info["velocity"]))
        self.vel_diff_rviz_pub.publish(
            Point(
                obs_info["velocity_norm"],
                self.goal["velocity"],
                proc_info["vel_diff"],
            )
        )
        if self.config["observation"]["enable_airspeed_sensor"]:
            self.airspeed_rviz_pub.publish(Point(obs_info["airspeed"], 0, 0))
        self.ang_rviz_pub.publish(Point(*obs_info["angle"]))
        self.ang_diff_rviz_pub.publish(Point(0, 0, proc_info["yaw_diff"]))
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

        if self.config["target"]["type"] == "MultiGoal" and self.config["target"].get(
            "enable_random_goal", True
        ):
            n_waypoints = np.random.randint(4, 8)
            self.target_type.sample_new_wplist(n_waypoints=n_waypoints)
        if self.config["simulation"]["enable_wind_sampling"]:
            self._sample_wind_state()
        if self.config["simulation"]["enable_buoyancy_sampling"]:
            self._sample_buoyancy(
                buoyancy_range=self.config["simulation"]["buoyancy_range"]
            )

        obs, _ = self.observation_type.observe()
        return obs

    def _update_goal_and_env(self):
        """update goal and env state"""
        self.goal = self.target_type.sample()

        if (
            self.config["simulation"]["enable_wind"]
            and self.config["simulation"]["enable_wind_sampling"]
        ):
            self.wind_state_pub.publish(self.wind_state)

    def _sample_wind_state(self):
        self.wind_state = WindSpeed()
        wind_speed = self.config["simulation"]["wind_speed"]
        self.wind_state.velocity.x = np.random.uniform(-wind_speed, wind_speed)
        self.wind_state.velocity.y = np.random.uniform(-wind_speed, wind_speed)
        self.wind_state.velocity.z = np.random.uniform(
            -wind_speed / 10, wind_speed / 10
        )

    def _sample_buoyancy(
        self,
        deflation_range=[0.0, 1.5],
        freeflop_angle_range=[0.0, 1.5],
        collapse_range=[0.0, 0.02],
        buoyancy_range=[0.9, 1.1],
    ):
        change_buoynacy(
            robot_id=self.config["robot_id"],
            ros_port=self.config["ros_port"],
            gaz_port=self.config["gaz_port"],
            deflation=np.random.uniform(*deflation_range),
            freeflop_angle=np.random.uniform(*freeflop_angle_range),
            collapse=np.random.uniform(*collapse_range),
            buoyancy=np.random.uniform(*buoyancy_range),
        )

    def _reward(
        self, obs: np.array, act: np.array, obs_info: dict
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - yaw angle difference - z diff - vel diff
        action_reward: penalty for motor use

        Args:
            obs (np.array): ("z_diff", "planar_dist", "yaw_diff", "vel_diff", "vel", "yaw_vel", "action")
            act (np.array): agent action [-1,1] with size (4,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        track_weights = self.config["tracking_reward_weights"].copy()
        reward_weights = self.config["reward_weights"].copy()

        success_reward = self.compute_success_rew(
            obs_info["position"], obs_info["goal_dict"]["position"]
        )
        obs[1] = (obs[1] + 1) / 2  # dist -1 should have max reward
        tracking_reward = np.dot(track_weights, -np.abs(obs[0:4]))
        action_reward = self.action_type.action_rew()

        reward = np.dot(
            reward_weights,
            (
                success_reward,
                tracking_reward,
                action_reward,
            ),
        )
        reward = np.clip(reward, -1, 1)
        rew_info = (reward, success_reward, tracking_reward, action_reward)
        reward_info = {"rew_info": rew_info}

        return float(reward), reward_info

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
            if np.linalg.norm(pos[0:2] - goal_pos[0:2])
            <= self.config["success_threshhold"]
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
            time = self.steps >= int(self.config["duration"]) - 1

        success = False
        if self.config["target"]["type"] == "MultiGoal":
            success = self.target_type.wp_index == self.target_type.wp_max_index
        else:
            success_reward = self.compute_success_rew(
                obs_info["position"], obs_info["goal_dict"]["position"]
            )
            success = success_reward >= 0.9

        return time or success

    def close(self) -> None:
        return super().close()
