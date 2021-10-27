""" observation type """
#!/usr/bin/env python

from blimp_env.envs.script.blimp_script import respawn_model
from typing import Any, TYPE_CHECKING, Dict, Tuple
import pandas as pd
import numpy as np
import rospy
from blimp_env.envs.common import utils
from geometry_msgs.msg import Point
from gym import spaces
from sensor_msgs.msg import Imu
from uav_msgs.msg import uav_pose
from transforms3d.euler import quat2euler

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

        self.obs_dim = 15

        self.pos_data = np.array([0, 0, 0])
        self.vel_data = np.array([0, 0, 0])
        self.acc_data = np.array([0, 0, 0])
        self.ori_data = np.array([0, 0, 0, 0])
        self.ang_data = np.array([0, 0, 0])
        self.ang_vel_data = np.array([0, 0, 0])

        self._create_pub_and_sub()
        self.ros_cnt = 0

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.full((self.obs_dim), -1),
            high=np.full((self.obs_dim), 1),
            dtype=np.float32,
        )

    def _create_pub_and_sub(self):
        if self.real_exp:
            imu_namespace = self.name_space + "/Imu"
        else:
            imu_namespace = self.name_space + "/tail/imu"
        rospy.Subscriber(
            imu_namespace,
            Imu,
            self._imu_callback,
        )

        rospy.Subscriber(self.name_space + "/pose", uav_pose, self._pose_callback)

    def _imu_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        acc = self.obj2array(msg.linear_acceleration)
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
        self.pos_data = self.obj2array(msg.position)
        self.vel_data = self.obj2array(msg.velocity)
        self.ori_data = self.obj2array(msg.orientation)
        self.ang_data = quat2euler(self.ori_data)
        self.ang_vel_data = self.obj2array(msg.angVelocity)

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

    def obj2array(
        rosobj,
        attr_list=["w", "x", "y", "z"],
    ):
        val_list = []
        for attr in attr_list:
            try:
                val_list.append(getattr(rosobj, attr))
            except:
                pass

        return np.array(val_list)

    def check_connection(self):
        """check ros connection"""
        while self.pos_data == np.array([0, 0, 0]):
            rospy.loginfo("[ observation ] waiting for pose subscriber")
            try:
                pose_data = rospy.wait_for_message(
                    self.name_space + "/pose",
                    uav_pose,
                    timeout=50,
                )
            except TimeoutError:
                rospy.loginfo("[ observation ] Simulation Crashed...Respawn")
                reply = respawn_model(**self.env.config["simulation"])
                rospy.loginfo("[ observation ] Simulation Respawned:", reply)

            self.pos_data = self.obj2array(pose_data.position)

        rospy.logdebug("pose ready")

    def observe(self) -> np.ndarray:
        raise NotImplementedError


class PlanarKinematicsObservation(ROSObservation):
    """Planar kinematics observation with action feedback"""

    OBS = ["z_diff", "planar_dist", "psi_diff", "vel_diff", "vel", "action"]
    OBS_range = {
        "z_diff": [0, 200],
        "planar_dist": [0, 200 * np.sqrt(2)],
        "psi_diff": [-np.pi, np.pi],
        "vel_diff": [-11.5, 11.5],
        "vel": [0, 11.49],
    }

    def __init__(
        self,
        env: "AbstractEnv",
        noise_stdv=0.02,
        scale_obs=True,
        action_feedback=True,
        **kwargs: dict
    ) -> None:
        super().__init__(env, **kwargs)
        self.noise_stdv = noise_stdv
        self.scale_obs = scale_obs

        self.obs_name = self.OBS
        self.obs_dim = 5
        self.range_dict = self.OBS_range

        self.action_feedback = action_feedback
        if self.action_feedback:
            act_dim = 4
            self.obs_dim += act_dim
            self.obs_name.append("action")

    def observe(self) -> np.ndarray:
        obs_dict = {
            "position": self.pos_data,
            "velocity": self.vel_data,
            "linear_acceleration": self.acc_data,
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
        }
        goal_dict = self.env.goal
        processed_dict = self.process_obs(obs_dict, goal_dict)

        if self.scale_obs:
            for key, val in processed_dict.items():
                proc = utils.lmap(val, self.range_dict[key], [-1, 1])
                proc = np.clip(proc, -1, 1)
                processed_dict[key] += np.random.normal(0, self.noise_stdv, proc.shape)

        if self.action_feedback:
            action = self.env.action_type.get_cur_act()[[0, 1, 5, 6]]
            processed_dict.update({"action": action})

        proc_df = pd.DataFrame.from_records([processed_dict])
        processed = np.concatenate(proc_df[self.obs_name].values[0])

        obs_dict.update({"proc_dict": processed_dict})
        obs_dict.update({"goal_dict": goal_dict})

        if self.dbg_obs:
            print("[ observation ] obs dict", obs_dict)
            print("[ observation ] processed array", processed)

        return processed, obs_dict

    def process_obs(self, obs_dict, goal_dict):
        obs_pos, goal_pos = obs_dict["position"], goal_dict["position"]
        vel = np.linalg.norm(obs_dict["velocity"])
        goal_vel = np.linalg.norm(goal_dict["velocity"])

        planar_dist = np.linalg.norm(obs_pos[0:2] - goal_pos[0:2])
        psi_diff = self.compute_psi_diff(goal_pos, obs_pos, obs_dict["angle"][2])

        return {
            "z_diff": obs_pos[2] - goal_pos[2],
            "planar_dist": planar_dist,
            "psi_diff": psi_diff,
            "vel_diff": vel - goal_vel,
            "vel": np.abs(vel),
        }

    @classmethod
    def compute_psi_diff(
        cls, goal_pos: np.array, obs_pos: np.array, obs_psi: float
    ) -> float:
        """compute psi angle of the vector machine position to goal position
        then compute the difference of this angle to machine psi angle
        last, make sure this angle lies within (-pi, pi) and scale to (-1,1)

        Args:
            goal_pos (np.array): [machine position]
            obs_pos (np.array): [goal postiion]
            obs_psi (float): [machine psi angle]

        Returns:
            float: [psi angle differences]
        """
        pos_diff = obs_pos - goal_pos
        goal_psi = np.arctan2(pos_diff[1], pos_diff[0]) - np.pi
        ang_diff = goal_psi - obs_psi

        if ang_diff > np.pi:
            ang_diff -= 2 * np.pi
        elif ang_diff < -np.pi:
            ang_diff += 2 * np.pi

        return ang_diff


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    """observation factory for different observation type"""
    if config["type"] == "PlanarKinematics":
        return PlanarKinematicsObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
