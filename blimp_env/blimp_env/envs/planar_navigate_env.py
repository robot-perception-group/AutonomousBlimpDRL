""" navigate env with velocity target """
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from blimp_env.envs.common.abstract import ROSAbstractEnv
from blimp_env.envs.common.action import Action
from geometry_msgs.msg import Point, Quaternion
from rotors_comm.msg import WindSpeed
from blimp_env.envs.script import close_simulation, change_buoynacy
import line_profiler

profile = line_profiler.LineProfiler()

Observation = Union[np.ndarray, float]


class PlanarNavigateEnv(ROSAbstractEnv):
    """Navigate blimp by path following decomposed to altitude and planar control"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update(
            {
                "enable_wind": False,
                "enable_wind_sampling": False,
                "wind_speed": 2.0,
                "enable_buoyancy_sampling": False,
            }
        )
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
                "disable_servo": True,
                "max_servo": -0.5,
                "max_thrust": 0.5,
            }
        )
        config["target"].update(
            {
                "type": "RandomGoal",
                "target_name_space": "goal_",
                "new_target_every_ts": 1200,
            }
        )
        config.update(
            {
                "duration": 1200,
                "simulation_frequency": 30,  # [hz]
                "policy_frequency": 10,  # [hz] has to be greater than 5 to overwrite backup controller
                "reward_weights": np.array([1, 0.8, 0.2]),  # success, tracking, action
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
        if self.config["simulation"]["enable_wind"]:
            self.wind_state_pub = rospy.Publisher(
                self.config["name_space"] + "/wind_state", WindSpeed, queue_size=1
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

        self.rew_rviz_pub.publish(Quaternion(*info["reward_info"]["rew_info"]))
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

        if self.config["simulation"]["enable_wind_sampling"]:
            self._sample_wind_state()
        if self.config["simulation"]["enable_buoyancy_sampling"]:
            self._sample_buoyancy()

        obs, _ = self.observation_type.observe()
        return obs

    def _update_goal_and_env(self):
        """sample new goal and update env state"""
        self.goal = self.target_type.sample()

        if self.config["simulation"]["enable_wind_sampling"]:
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
        deflation_range=[0.0, 2.0],
        freeflop_angle_range=[0.0, 2.0],
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
            obs_info["position"], self.goal["position"]
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
            if np.round(np.linalg.norm(pos[0:2] - goal_pos[0:2]), 5)
            < self.config["success_threshhold"]
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

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )
        success = success_reward >= 1

        return time or success

    def close(self) -> None:
        return super().close()


class ResidualPlanarNavigateEnv(PlanarNavigateEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update(
            {
                "enable_wind": False,
                "enable_wind_sampling": False,
                "wind_speed": 2.0,
                "enable_buoyancy_sampling": True,
            }
        )
        config["observation"].update(
            {
                "type": "PlanarKinematics",
                "noise_stdv": 0.015,
                "scale_obs": True,
                "enable_rsdact_feedback": True,
            }
        )
        config["action"].update(
            {
                "type": "SimpleContinuousDifferentialAction",
                "act_noise_stdv": 0.05,
                "disable_servo": True,
                "max_servo": -0.5,
                "max_thrust": 0.5,
            }
        )
        config["target"].update(
            {
                "type": "RandomGoal",
                "target_name_space": "goal_",
                "new_target_every_ts": 1200,
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
                    [0.3, 0.40, 0.20, 0.10]
                ),  # z_diff, planar_dist, yaw_diff, vel_diff
                "success_threshhold": 10,  # [meters]
                "reward_scale": 0.1,
                "clip_reward": False,
                "enable_residual_ctrl": True,
                "mixer_type": "relative",
                "beta": 0.5,
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config=config)

        self.pid_act = np.zeros(4)

        self.delta_t = 1 / self.config["policy_frequency"] * 10
        self.yaw_err_i, self.prev_yaw = 0, 0
        self.alt_err_i, self.prev_alt = 0, 0
        self.vel_err_i, self.prev_vel = 0, 0

    def _create_pub_and_sub(self):
        self.ang_vel_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang_vel", Point, queue_size=1
        )
        self.pid_act_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_pid_act", Quaternion, queue_size=1
        )
        return super()._create_pub_and_sub()

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
        joint_act = self.mixer(action, self.pid_act, self.config["beta"])
        self._simulate(joint_act)

        self.pid_act = (
            self.residual_ctrl() if self.config["enable_residual_ctrl"] else np.zeros(4)
        )
        obs, obs_info = self.observation_type.observe(self.pid_act.copy())
        reward, reward_info = self._reward(obs, joint_act, obs_info)
        terminal = self._is_terminal(obs_info)
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "pid_act": self.pid_act,
            "joint_act": joint_act,
            "reward": reward,
            "reward_info": reward_info,
            "terminal": terminal,
        }

        self._update_goal_and_env()
        self._step_info(info)

        return obs, reward, terminal, info

    def _step_info(self, info: dict):
        obs_info = info["obs_info"]
        proc_info = obs_info["proc_dict"]

        self.ang_vel_rviz_pub.publish(Point(0, 0, proc_info["yaw_vel"]))
        self.pid_act_rviz_pub.publish(Quaternion(*info["pid_act"]))
        return super()._step_info(info)

    def mixer(self, action, pid_act, beta=0.5):
        if self.config["enable_residual_ctrl"] == False:
            return action

        if self.config["mixer_type"] == "absolute":
            joint_act = beta * action + (1 - beta) * pid_act
        elif self.config["mixer_type"] == "relative":
            joint_act = pid_act * (1 + beta * action)
        else:
            raise NotImplementedError

        joint_act[2] = action[2]
        return np.clip(joint_act, -1, 1)

    def residual_ctrl(self):
        """
        use PID controller to generate control signal
        """
        obs, obs_dict = self.observation_type.observe()

        yaw_ctrl, self.yaw_err_i, _ = self.pid_ctrl(
            -obs[2],
            self.yaw_err_i,
            obs_dict["angular_velocity"][2],
            pid_coeff=np.array([1.0, 0.1, 0.05]),
            d_from_sensor=True,
        )
        alt_ctrl, self.alt_err_i, self.prev_alt = self.pid_ctrl(
            obs[0],
            self.alt_err_i,
            self.prev_alt,
        )
        vel_ctrl, self.vel_err_i, self.prev_vel = self.pid_ctrl(
            -obs[3], self.vel_err_i, self.prev_vel
        )
        ser_ctrl = 0

        return np.clip(np.array([yaw_ctrl, alt_ctrl, ser_ctrl, vel_ctrl]), -1, 1)

    def clear_pid_param(self):
        self.yaw_err_i, self.alt_err_i, self.vel_err_i = 0, 0, 0
        self.prev_yaw, self.prev_alt, self.prev_vel = 0, 0, 0

    def pid_ctrl(
        self,
        err,
        err_i,
        err_d,
        offset=0.0,
        pid_coeff=np.array([1.0, 0.2, 0.05]),
        i_from_sensor=False,
        d_from_sensor=False,
    ):
        if not i_from_sensor:
            err_i += err * self.delta_t
            err_i = np.clip(err_i, -1, 1)

        if not d_from_sensor:
            err_d = (err - err_d) / self.delta_t

        ctrl = np.dot(pid_coeff, np.array([err, err_i, err_d])) + offset
        return ctrl, err_i, err

    def reset(self) -> Observation:
        self.clear_pid_param()
        return super().reset()

    def _reward(
        self, obs: np.array, act: np.array, obs_info: dict
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - yaw angle difference - z diff - vel diff
        action_reward: penalty for motor use

        Args:
            obs (np.array): ("z_diff", "planar_dist", "yaw_diff", "vel_diff", "vel", "yaw_vel" "action")
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

        obs[1] = (obs[1] + 1) / 2  # dist -1 should have max reward
        tracking_reward = np.dot(track_weights, -np.abs(obs[0:4]))

        action_reward = self.action_type.action_rew()

        reward = self.config["reward_scale"] * np.dot(
            reward_weights,
            (success_reward, tracking_reward, action_reward),
        )
        if self.config["clip_reward"]:
            reward = np.clip(reward, -1, 1)

        rew_info = (reward, success_reward, tracking_reward, action_reward)

        return float(reward), {"rew_info": rew_info}

    def _is_terminal(self, obs_info: dict) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"]) - 1

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )
        success = success_reward >= 1

        return time or success


class TestYawEnv(ResidualPlanarNavigateEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["observation"].update(
            {
                "type": "DummyYaw",
                "noise_stdv": 0.015,
                "scale_obs": True,
                "enable_rsdact_feedback": True,
            }
        )
        config["action"].update(
            {
                "type": "DummyYawAction",
                "act_noise_stdv": 0.05,
                "disable_servo": True,
            }
        )
        config["target"].update(
            {
                "type": "RandomGoal",
                "target_name_space": "goal_",
                "new_target_every_ts": 1200,
            }
        )
        config.update(
            {
                "duration": 1200,
                "simulation_frequency": 30,  # [hz]
                "policy_frequency": 10,  # [hz] has to be greater than 5 to overwrite backup controller
                "reward_weights": np.array([1.0, 1.0, 0]),  # success, tracking, action
                "tracking_reward_weights": np.array([1.0]),  # yaw_diff
                "success_seconds": 5,  # seconds within threshold as success
                "reward_scale": 0.1,
                "clip_reward": False,
                "enable_residual_ctrl": True,
                "mixer_type": "absolute",  # absolute, relative, hybrid
                "pid_param": np.array([1.0, 0.0, 0.05]),
                "beta": 0.5,
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config=config)
        self.success_cnt = 0
        self.pid_act = 0

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
        joint_act = self.mixer(action, self.pid_act, self.config["beta"])
        self._simulate(joint_act)
        self.pid_act = (
            self.residual_ctrl() if self.config["enable_residual_ctrl"] else np.zeros(1)
        )
        obs, obs_info = self.observation_type.observe(self.pid_act.copy())
        reward, reward_info = self._reward(obs, joint_act, obs_info)
        terminal = self._is_terminal(obs_info)
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "pid_act": self.pid_act,
            "joint_act": joint_act,
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

        self.rew_rviz_pub.publish(Quaternion(*info["reward_info"]["rew_info"]))
        self.state_rviz_pub.publish(Quaternion(0, proc_info["yaw_diff"], 0, 0))
        self.ang_diff_rviz_pub.publish(Point(0, 0, proc_info["yaw_diff"]))
        self.ang_vel_rviz_pub.publish(Point(0, 0, proc_info["yaw_vel"]))

        self.act_rviz_pub.publish(Quaternion(*info["act"], 0, 0, 0))
        self.pid_act_rviz_pub.publish(
            Quaternion(*info["pid_act"], *info["joint_act"], 0, 0)
        )

        self.pos_cmd_pub.publish(Point(*self.goal["position"]))

        if self.dbg:
            print(
                f"================= [ PlanarNavigateEnv ] step {self.steps} ================="
            )
            print("STEP INFO:", info)
            print("\r")

    def mixer(self, action, pid_act, beta=0.5):
        if self.config["enable_residual_ctrl"] == False:
            return action

        if self.config["mixer_type"] == "absolute":
            joint_act = beta * action + (1 - beta) * pid_act
        elif self.config["mixer_type"] == "relative":
            joint_act = pid_act * (1 + beta * action)
        else:
            raise NotImplementedError

        return np.clip(joint_act, -1, 1)

    def residual_ctrl(self):
        """
        use PID controller to generate a residual signal
        """
        obs, obs_dict = self.observation_type.observe()

        yaw_ctrl, self.yaw_err_i, _ = self.pid_ctrl(
            -obs[0],
            self.yaw_err_i,
            obs_dict["angular_velocity"][2],
            self.config["pid_param"],
        )
        return np.clip(np.array([yaw_ctrl]), -1, 1)

    def pid_ctrl(self, err, err_i, err_d, pid_coeff=np.array([1, 0.5, 0.3])):
        err_i += err * self.delta_t
        err_i = np.clip(err_i, -1, 1)
        ctrl = np.dot(pid_coeff, np.array([err, err_i, err_d]))
        return ctrl, err_i, err

    def _reward(
        self, obs: np.array, act: np.array, obs_info: dict
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: 0
        tracking_reward: - yaw angle difference
        action_reward: penalty for motor use

        Args:
            obs (np.array): ("yaw_diff", "yaw_vel", "action")
            act (np.array): agent action [-1,1] with size (1,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        track_weights = self.config["tracking_reward_weights"].copy()
        reward_weights = self.config["reward_weights"].copy()

        yaw_diff = obs_info["proc_dict"]["yaw_diff"]
        success_reward = self.compute_success_rew(yaw_diff)

        tracking_reward = np.dot(track_weights, -np.abs(obs[0]))

        action_reward = self.action_type.action_rew()

        reward = self.config["reward_scale"] * np.dot(
            reward_weights,
            (success_reward, tracking_reward, action_reward),
        )

        if self.config["clip_reward"]:
            reward = np.clip(reward, -1, 1)

        rew_info = (reward, success_reward, tracking_reward, action_reward)

        return float(reward), {"rew_info": rew_info}

    def compute_success_rew(
        self,
        yaw_diff: np.array,
        epsilon: float = 0.1,
    ) -> float:
        """yaw_diff less than 0.1 for 5 seconds

        Args:
            yaw_diff (np.array): [scaled yaw diff]
            epsilon (float, optional): [tolerence]. Defaults to 0.1.

        Returns:
            float: [0.0 or 1.0]
        """
        if np.abs(yaw_diff) < epsilon:
            self.success_cnt += 1
        else:
            self.success_cnt = 0

        return float(
            self.success_cnt
            > self.config["success_seconds"] * self.config["simulation_frequency"]
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

        return time

    def close(self) -> None:
        close_simulation()


if __name__ == "__main__":
    import copy
    from blimp_env.envs.common.gazebo_connection import GazeboConnection
    from blimp_env.envs.script import close_simulation

    # ============== profile ==============#
    # 1. pip install line_profiler
    # 2. in terminal:
    # kernprof -l -v blimp_env/envs/planar_navigate_env.py

    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    ENV = ResidualPlanarNavigateEnv  # PlanarNavigateEnv, ResidualPlanarNavigateEnv, TestYawEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
            "enable_wind": False,
            "enable_wind_sampling": False,
            "enable_buoyancy_sampling": True,
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": False,
            "noise_stdv": 0.0,
        },
        "action": {
            "DBG_ACT": False,
            "act_noise_stdv": 0.0,
        },
        "target": {"DBG_ROS": False},
    }

    @profile
    def env_step():
        env = ENV(copy.deepcopy(env_kwargs))
        env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            action = np.zeros_like(action)  # [yaw, pitch, servo, thrust]
            obs, reward, terminal, info = env.step(action)

        GazeboConnection().unpause_sim()

    env_step()
