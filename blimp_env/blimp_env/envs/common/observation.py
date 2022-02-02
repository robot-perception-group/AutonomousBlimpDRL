""" observation type """
#!/usr/bin/env python

from math import e
import subprocess
from typing import TYPE_CHECKING, Any, Dict, Tuple
import time
import numpy as np
import pandas as pd
import rospy
from blimp_env.envs.common import utils
from blimp_env.envs.script.blimp_script import respawn_model, resume_simulation
from gym import spaces
from sensor_msgs.msg import Imu
from transforms3d.euler import quat2euler
from uav_msgs.msg import uav_pose

if TYPE_CHECKING:
    from blimp_env.envs.common.abstract import AbstractEnv

GRAVITY = 9.81


class ObservationType:
    """abstract observation type"""

    def __init__(
        self, env: "AbstractEnv", **kwargs  # pylint: disable=unused-argument
    ) -> None:
        self.env = env

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()


class ROSObservation(ObservationType):
    """kinematirc obervation from sensors"""

    def __init__(
        self,
        env: "AbstractEnv",
        name_space="machine_0",
        DBG_ROS=False,
        DBG_OBS=False,
        real_experiment=False,
        **kwargs  # pylint: disable=unused-argument
    ):
        super().__init__(env)
        self.name_space = name_space
        self.dbg_ros = DBG_ROS
        self.dbg_obs = DBG_OBS
        self.real_exp = real_experiment
        self.imu_namespace = (
            self.name_space + "/Imu" if self.real_exp else self.name_space + "/tail/imu"
        )

        self.obs_dim = 15

        self.pos_data = np.array([0, 0, 0])
        self.vel_data = np.array([0, 0, 0])
        self.acc_data = np.array([0, 0, 0])
        self.ori_data = np.array([0, 0, 0, 0])
        self.ang_data = np.array([0, 0, 0])
        self.ang_vel_data = np.array([0, 0, 0])
        self.airspeed_data = np.array([0])

        self.ros_cnt = 0

        self._create_pub_and_sub()

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.full((self.obs_dim), -1),
            high=np.full((self.obs_dim), 1),
            dtype=np.float32,
        )

    def _create_pub_and_sub(self):
        rospy.Subscriber(
            self.imu_namespace,
            Imu,
            self._imu_callback,
        )

        rospy.Subscriber(self.name_space + "/pose", uav_pose, self._pose_callback)
        time.sleep(1)

    def _imu_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        acc = utils.obj2array(msg.linear_acceleration)
        if self.real_exp:
            acc[2] += GRAVITY
        else:
            acc[2] -= GRAVITY

        self.acc_data = acc

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 100 == 0:
                print(
                    "[ KinematicObservation ] imu_callback: linear_acceleration",
                    self.acc_data,
                )

    def _pose_callback(self, msg):
        """pose msg callback

        Args:
            msg ([uav_pose]): gcs processed sensor data
        """
        self.pos_data = utils.obj2array(msg.position)
        self.vel_data = utils.obj2array(msg.velocity)
        self.ori_data = utils.obj2array(msg.orientation)
        self.ang_data = quat2euler(self.ori_data)
        self.ang_vel_data = utils.obj2array(msg.angVelocity)
        self.airspeed_data = np.array(msg.POI.x)
        if self.airspeed_data < 0.25:
            self.airspeed_data = np.zeros(1)

        if self.dbg_ros:
            print(
                "[ KinematicObservation ] pose_callback: position",
                self.pos_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: velocity",
                self.vel_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: orientation",
                self.ori_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: angle",
                self.ang_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: ang_vel",
                self.ang_vel_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: airspeed",
                self.airspeed_data,
            )

    def check_connection(self):
        """check ros connection"""
        while (self.pos_data == np.zeros(3)).all():
            rospy.loginfo("[ observation ] waiting for pose subscriber...")
            try:
                pose_data = rospy.wait_for_message(
                    self.name_space + "/pose",
                    uav_pose,
                    timeout=50,
                )
                imu_data = rospy.wait_for_message(
                    self.imu_namespace,
                    Imu,
                    timeout=50,
                )
            except:
                rospy.loginfo("[ observation ] cannot find pose subscriber")
                self.obs_err_handle()
                pose_data = rospy.wait_for_message(
                    self.name_space + "/pose",
                    uav_pose,
                    timeout=50,
                )

            self.pos_data = utils.obj2array(pose_data.position)

        rospy.loginfo("[ observation ] pose ready")

    def observe(self) -> np.ndarray:
        raise NotImplementedError

    def obs_err_handle(self):
        try:
            rospy.loginfo("[ observation ] respawn model...")
            reply = respawn_model(**self.env.config["simulation"])
            rospy.loginfo("[ observation ] respawn model status ", reply)
        except:
            rospy.loginfo("[ observation ] resume simulation...")
            reply = resume_simulation(**self.env.config["simulation"])
            rospy.loginfo("[ observation ] resume simulation status ", reply)
        return reply


class PlanarKinematicsObservation(ROSObservation):
    """Planar kinematics observation with actuator feedback"""

    OBS = ["z_diff", "planar_dist", "yaw_diff", "vel_diff", "vel", "yaw_vel"]
    OBS_range = {
        "z_diff": [-100, 100],
        "planar_dist": [0, 200 * np.sqrt(2)],
        "yaw_diff": [-np.pi, np.pi],
        "vel_diff": [-11.5, 11.5],
        "vel": [0, 11.5],
        "yaw_vel": [-15, 15],
    }

    def __init__(
        self,
        env: "AbstractEnv",
        noise_stdv=0.02,
        scale_obs=True,
        enable_rsdact_feedback=True,  # observe other control command source
        enable_airspeed_sensor=False,  # add airspeed sensor
        enable_next_goal=False,  # add next goal, only used with multigoal
        **kwargs: dict
    ) -> None:
        super().__init__(env, **kwargs)
        self.noise_stdv = noise_stdv
        self.scale_obs = scale_obs
        self.enable_rsdact_feedback = enable_rsdact_feedback
        self.enable_airspeed_sensor = enable_airspeed_sensor
        self.enable_next_goal = enable_next_goal

        self.obs_name = self.OBS.copy()
        self.obs_dim = len(self.OBS)
        self.range_dict = self.OBS_range

        if self.enable_rsdact_feedback:
            self.obs_dim += 4
            self.obs_name.append("residual_act")

        if self.enable_airspeed_sensor:
            self.obs_dim += 1
            self.obs_name.append("airspeed")
            self.range_dict.update({"airspeed": [0, 7]})

        if self.enable_next_goal:
            self.obs_dim += 1
            self.obs_name.append("next_yaw_diff")
            self.range_dict.update(
                {
                    "next_yaw_diff": [-np.pi, np.pi],
                }
            )

        self.obs_dim += 4
        self.obs_name.append("actuator")

    def observe(self, rsdact=np.zeros(4)) -> np.ndarray:
        obs, obs_dict = self._observe(rsdact)
        while np.isnan(obs).any():
            rospy.loginfo("[ observation ] obs corrupted by NA")
            self.obs_err_handle()
            obs, obs_dict = self._observe(rsdact)
        return obs, obs_dict

    def _observe(self, rsdact=np.zeros(4)) -> np.ndarray:
        obs_dict = {
            "position": self.pos_data,
            "velocity": self.vel_data,
            "velocity_norm": np.linalg.norm(self.vel_data),
            "linear_acceleration": self.acc_data,
            "acceleration_norm": np.linalg.norm(self.acc_data),
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
            "airspeed": self.airspeed_data,
        }

        goal_dict = self.env.goal
        processed_dict = self.process_obs(obs_dict, goal_dict, self.scale_obs)

        if self.enable_rsdact_feedback:
            processed_dict.update({"residual_act": rsdact})

        actuator = self.env.action_type.get_cur_act()[[0, 1, 5, 6]]
        processed_dict.update({"actuator": actuator})

        proc_df = pd.DataFrame.from_records([processed_dict])
        processed = np.hstack(proc_df[self.obs_name].values[0])

        obs_dict.update({"proc_dict": processed_dict})
        obs_dict.update({"goal_dict": goal_dict})

        if self.dbg_obs:
            print("[ observation ] state", processed)
            print("[ observation ] obs dict", obs_dict)

        return processed, obs_dict

    def process_obs(
        self, obs_dict: dict, goal_dict: dict, scale_obs: bool = True
    ) -> dict:
        obs_pos, goal_pos, next_goal_pos = (
            obs_dict["position"],
            goal_dict["position"],
            goal_dict["next_position"],
        )
        vel = np.linalg.norm(obs_dict["velocity"])
        goal_vel = goal_dict["velocity"]

        planar_dist = np.linalg.norm(obs_pos[0:2] - goal_pos[0:2])
        yaw_diff = self.compute_yaw_diff(goal_pos, obs_pos, obs_dict["angle"][2])

        state_dict = {
            "z_diff": obs_pos[2] - goal_pos[2],
            "planar_dist": planar_dist,
            "yaw_diff": yaw_diff,
            "vel_diff": vel - goal_vel,
            "vel": vel,
            "yaw_vel": obs_dict["angular_velocity"][2],
        }

        if self.enable_airspeed_sensor:
            state_dict.update({"airspeed": obs_dict["airspeed"]})

        if self.enable_next_goal:
            next_yaw_diff = self.compute_yaw_diff(
                next_goal_pos, obs_pos, obs_dict["angle"][2]
            )
            state_dict.update({"next_yaw_diff": next_yaw_diff})

        if scale_obs:
            state_dict = self.scale_obs_dict(state_dict, self.noise_stdv)

        return state_dict

    def scale_obs_dict(self, state_dict: dict, noise_level: float = 0.0) -> dict:
        for key, val in state_dict.items():
            proc = utils.lmap(val, self.range_dict[key], [-1, 1])
            proc += np.random.normal(0, noise_level, proc.shape)
            proc = np.clip(proc, -1, 1)
            state_dict[key] = proc
        return state_dict

    @classmethod
    def compute_yaw_diff(
        cls, goal_pos: np.array, obs_pos: np.array, obs_yaw: float
    ) -> float:
        """compute yaw angle of the vector machine position to goal position
        then compute the difference of this angle to machine yaw angle
        last, make sure this angle lies within (-pi, pi)

        Args:
            goal_pos (np.array): [machine position]
            obs_pos (np.array): [goal postiion]
            obs_yaw (float): [machine yaw angle]

        Returns:
            float: [yaw angle differences]
        """
        pos_diff = obs_pos - goal_pos
        goal_yaw = np.arctan2(pos_diff[1], pos_diff[0]) - np.pi
        ang_diff = goal_yaw - obs_yaw

        if ang_diff > np.pi:
            ang_diff -= 2 * np.pi
        elif ang_diff < -np.pi:
            ang_diff += 2 * np.pi

        return ang_diff


class DummyYawObservation(PlanarKinematicsObservation):
    """Planar kinematics observation with actuator feedback"""

    OBS = ["yaw_diff", "yaw_vel"]
    OBS_range = {
        "yaw_diff": [-np.pi, np.pi],
        "yaw_vel": [-15, 15],
    }

    def __init__(
        self,
        env: "AbstractEnv",
        noise_stdv=0.02,
        scale_obs=True,
        enable_rsdact_feedback=True,
        **kwargs: dict
    ) -> None:
        super().__init__(env, noise_stdv=noise_stdv, scale_obs=scale_obs, **kwargs)
        self.enable_rsdact_feedback = enable_rsdact_feedback

        self.obs_name = self.OBS.copy()
        self.obs_dim = len(self.OBS)
        self.range_dict = self.OBS_range

        if self.enable_rsdact_feedback:
            self.obs_dim += 1
            self.obs_name.append("residual_act")

        self.obs_dim += 1
        self.obs_name.append("actuator")

    def observe(self, rsdact=np.array([0.0])) -> np.ndarray:
        obs, obs_dict = self._observe(rsdact)
        while np.isnan(obs).any():
            rospy.loginfo("[ observation ] obs corrupted by NA")
            self.obs_err_handle()
            obs, obs_dict = self._observe(rsdact)
        return obs, obs_dict

    def _observe(self, rsdact=np.array([0.0])) -> np.ndarray:
        obs_dict = {
            "position": self.pos_data,
            "velocity": self.vel_data,
            "velocity_norm": np.linalg.norm(self.vel_data),
            "linear_acceleration": self.acc_data,
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
        }

        goal_dict = self.env.goal
        processed_dict = self.process_obs(obs_dict, goal_dict, self.scale_obs)

        if self.enable_rsdact_feedback:
            processed_dict.update({"residual_act": rsdact})

        actuator = self.env.action_type.get_cur_act()[[0]]
        processed_dict.update({"actuator": actuator})

        proc_df = pd.DataFrame.from_records([processed_dict])
        processed = np.hstack(proc_df[self.obs_name].values[0])

        obs_dict.update({"proc_dict": processed_dict})
        obs_dict.update({"goal_dict": goal_dict})

        if self.dbg_obs:
            print("[ observation ] state", processed)
            print("[ observation ] obs dict", obs_dict)

        return processed, obs_dict

    def process_obs(
        self, obs_dict: dict, goal_dict: dict, scale_obs: bool = True
    ) -> dict:
        obs_pos, goal_pos = obs_dict["position"], goal_dict["position"]
        yaw_diff = self.compute_yaw_diff(goal_pos, obs_pos, obs_dict["angle"][2])

        state_dict = {"yaw_diff": yaw_diff, "yaw_vel": obs_dict["angular_velocity"][2]}

        if scale_obs:
            state_dict = self.scale_obs_dict(state_dict, self.noise_stdv)

        return state_dict


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    """observation factory for different observation type"""
    if config["type"] == "PlanarKinematics":
        return PlanarKinematicsObservation(env, **config)
    elif config["type"] == "DummyYaw":
        return DummyYawObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
