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
                "wind_speed": 0.0,
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
                "max_servo": 1.0,
                "max_thrust": 1.0,
            }
        )
        config["target"].update(
            {
                "type": "AerobaticGoal",
                "target_name_space": "goal_",
                "task_name": "stand",  # stand, backward, upside_down, loop, roll, or int(0,4), or random
            }
        )
        config.update(
            {
                "duration": 1200,
                "simulation_frequency": 30,  # [hz]
                "policy_frequency": 10,  # [hz] has to be greater than 5 to overwrite backup controller
                "reward_weights": np.array(
                    [100, 1.0, 0.0]
                ),  # success, tracking, action
                "success_threshhold": (0.1, 10, 0.1, 0.1, 0.1),  # (%, m, %, %, %)
                "success_seconds": 5,  # above success threshhold for [sec] to reach terminal
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config)
        self.success = False
        self.success_cnt = 0

    def _create_pub_and_sub(self):
        super()._create_pub_and_sub()
        self.rew_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward", Float32MultiArray, queue_size=1
        )
        self.state_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_state", Float32MultiArray, queue_size=1
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
        self.angvel_diff_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_angvel_diff", Point, queue_size=1
        )
        self.act_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_act", Float32MultiArray, queue_size=1
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
        if self.config["observation"]["enable_airspeed_sensor"]:
            self.airspeed_rviz_pub = rospy.Publisher(
                self.config["name_space"] + "/rviz_airspeed", Point, queue_size=1
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
            Float32MultiArray(
                data=(
                    proc_info["roll_diff"],
                    proc_info["pitch_diff"],
                    proc_info["yaw_diff"],
                    proc_info["roll_vel_diff"],
                    proc_info["pitch_vel_diff"],
                    proc_info["yaw_vel_diff"],
                    proc_info["z_diff"],
                    proc_info["planar_dist"],
                    proc_info["planar_yaw_diff"],
                    obs_info["task"],
                )
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
        self.ang_diff_rviz_pub.publish(
            Point(
                proc_info["roll_diff"], proc_info["pitch_diff"], proc_info["yaw_diff"]
            )
        )
        self.angvel_diff_rviz_pub.publish(
            Point(
                proc_info["roll_vel_diff"],
                proc_info["pitch_vel_diff"],
                proc_info["yaw_vel_diff"],
            )
        )
        self.act_rviz_pub.publish(Float32MultiArray(data=info["act"]))

        self.pos_cmd_pub.publish(Point(*self.goal["position"]))

        if self.dbg:
            print(
                f"================= [ PlanarNavigateEnv ] step {self.steps} ================="
            )
            print("STEP INFO:", info)
            print("\r")

    def reset(self) -> Observation:
        self.steps = 0
        self.success = False
        self.success_cnt = 0
        self.done = False
        self._reset()

        if self.config["target"]["task_name"] == "random":
            self.target_type.sample_task()

        if self.target_type.task == int(1):
            self.target_type.sample_planar_goal()

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
            obs (np.array): ("roll_diff", "pitch_diff", "yaw_diff", "vel_diff",
                             "roll_vel_diff", "pitch_vel_diff", "pitch_vel_diff",
                             "z_diff", "planar_dist", "planar_yaw_diff") in [-1, 1]
            act (np.array): agent action [-1,1] with size (8,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        reward_weights = self.config["reward_weights"].copy()

        success_reward = self.compute_success_rew(copy.deepcopy(obs_info))
        tracking_reward = self.compute_tacking_rew(obs, copy.deepcopy(obs_info))
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

    def compute_tacking_rew(self, obs, obs_info: dict) -> float:
        """task success if distance to goal is less than sucess_threshhold

        Args:
            obs (np.array): ("roll_diff", "pitch_diff", "yaw_diff", "vel_diff",
                    "roll_vel_diff", "pitch_vel_diff", "pitch_vel_diff",
                    "z_diff", "planar_dist", "planar_yaw_diff") in [-1, 1]
            obs_info (dict): contain all information of a step

        Returns:
            [float]: tracking reward in [0,1]
        """
        proc_dict = obs_info["proc_dict"]
        goal_dict = obs_info["goal_dict"]
        task = obs_info["task"]

        dist_to_goal = np.linalg.norm(obs_info["position"] - goal_dict["position"])
        scaled_dist = dist_to_goal / (100 * np.sqrt(3))  # penalize distance to goal
        penalty = -10  # if distance is too far, add extra penalty

        if task == int(0):
            rew = -np.abs(proc_dict["pitch_diff"]) - scaled_dist
        elif task == int(1):
            rew = -scaled_dist
        elif task == int(2):
            rew = (
                -np.abs(proc_dict["pitch_diff"])
                - np.abs(proc_dict["roll_diff"])
                - scaled_dist
            )
        elif task == int(3):
            rew = -np.abs(proc_dict["pitch_vel_diff"]) - scaled_dist
        elif task == int(4):
            rew = -np.abs(proc_dict["roll_vel_diff"]) - scaled_dist

        if scaled_dist >= 1:
            rew += penalty

        return rew

    def compute_success_rew(self, obs_info: dict) -> float:
        """define success condition of each task

        Args:
            [dict]: obs_info contains all step information

        Returns:
            [float]: [1 if success, otherwise 0]
        """
        proc_dict = obs_info["proc_dict"]
        goal_dict = obs_info["goal_dict"]
        task = obs_info["task"]
        threshhold = self.config["success_threshhold"]
        if task == int(0):
            success = np.abs(proc_dict["pitch_diff"]) <= threshhold[task]
        elif task == int(1):
            success = (
                np.linalg.norm(obs_info["position"] - goal_dict["position"])
                <= threshhold[task]
            )
        elif task == int(2):
            success = (
                np.abs(proc_dict["pitch_diff"]) <= threshhold[task]
                or np.abs(proc_dict["roll_diff"]) <= threshhold[task]
            )

        elif task == int(3):
            success = np.abs(proc_dict["pitch_vel_diff"]) <= threshhold[task]

        elif task == int(4):
            success = np.abs(proc_dict["roll_vel_diff"]) <= threshhold[task]

        self.success = bool(success)
        return float(success)

    def _is_terminal(self, obs_info: dict) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"]) - 1

        terminal_success = False
        if self.success is True:
            self.success_cnt += 1
        else:
            self.success_cnt = 0

        terminal_success = (
            self.success_cnt
            > self.config["success_seconds"] * self.config["simulation_frequency"]
        )

        return time or terminal_success

    def close(self) -> None:
        return super().close()


if __name__ == "__main__":
    import copy
    from blimp_env.envs.common.gazebo_connection import GazeboConnection
    from blimp_env.envs.script import close_simulation

    # ============== profile ==============#
    # 1. pip install line-profiler
    # 2. in terminal:
    # kernprof -l -v blimp_env/envs/planar_navigate_env.py

    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    ENV = AerobaticEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
            "enable_wind": False,
            "enable_wind_sampling": False,
            "enable_buoyancy_sampling": False,
            "wind_speed": 0,
            "wind_direction": (1, 0),
            "position": (0, 0, 100),  # initial spawned position
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": False,
            "noise_stdv": 0.0,
        },
        "action": {
            "max_thrust": 1.0,
            "DBG_ACT": False,
            "act_noise_stdv": 0.0,
            "disable_servo": False,
        },
        "target": {
            "task_name": "random",  # stand, backward, upside_down, loop, roll, or int(0,4), or random
            "DBG_ROS": False,
        },
    }

    @profile
    def env_step():
        env = ENV(copy.deepcopy(env_kwargs))
        env.reset()
        for _ in range(100000):
            # action: [m2, lfin, rfin, tfin, bfin, stick, m1, m0]
            action = env.action_space.sample()
            action = np.zeros_like(action)

            obs, reward, terminal, info = env.step(action)

        GazeboConnection().unpause_sim()

    env_step()
