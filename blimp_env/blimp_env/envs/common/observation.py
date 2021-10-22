""" observation type """
#!/usr/bin/env python

from blimp_env.envs.script.blimp_script import respawn_model
from typing import Any, TYPE_CHECKING, Dict, Tuple

import numpy as np
import rospy
from blimp_env.envs.common.data_processor import DataProcessor
from blimp_env.envs.common.utils import DataObj, RangeObj
from geometry_msgs.msg import Point, Quaternion
from gym import spaces
from sensor_msgs.msg import Imu
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


class KinematicObservation(ObservationType):
    """kinematirc obervation from sensors"""

    OBS = [
        "position",
        "velocity",
        "linear_acceleration",
        "angle",
        # "orientation",
        "angular_velocity",
    ]

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
        self.data_processor = DataProcessor()
        self.name_space = name_space
        self.dbg_ros = DBG_ROS
        self.dbg_obs = DBG_OBS
        self.real_exp = real_experiment

        self.obs_name = self.OBS
        self.obs_dim = 15

        range_dict = self._create_range_obj()
        self.position_data = DataObj(Point(), range_dict["position_range"])
        self.vel_data = DataObj(Point(), range_dict["vel_range"])
        self.acc_data = DataObj(Point(), range_dict["acc_range"])
        self.ori_data = DataObj(Quaternion(), range_dict["ori_range"])
        self.ang_data = DataObj(Point(), range_dict["ang_range"])
        self.ang_vel_data = DataObj(Point(), range_dict["ang_vel_range"])

        self._create_pub_and_sub()
        self.ros_cnt = 0

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.full((self.obs_dim), -1),
            high=np.full((self.obs_dim), 1),
            dtype=np.float32,
        )

    @classmethod
    def _create_range_obj(cls) -> Dict:
        scale_min_max = (-1, 1)
        position_range = {
            "x": RangeObj((-100, 100), scale_min_max),
            "y": RangeObj((-100, 100), scale_min_max),
            "z": RangeObj((-200, 0), scale_min_max),
        }  # NED frame
        vel_range = {
            "x": RangeObj((-8, 8), scale_min_max),
            "y": RangeObj((-8, 8), scale_min_max),
            "z": RangeObj((-2, 2), scale_min_max),
        }  # NED frame
        acc_range = {
            "x": RangeObj((-5, 5), scale_min_max),
            "y": RangeObj((-5, 5), scale_min_max),
            "z": RangeObj((-2, 2), scale_min_max),
        }
        ori_range = {
            "x": RangeObj((-1, 1), scale_min_max),
            "y": RangeObj((-1, 1), scale_min_max),
            "z": RangeObj((-1, 1), scale_min_max),
            "w": RangeObj((-1, 1), scale_min_max),
        }
        ang_range = {
            "x": RangeObj((-3.14, 3.14), scale_min_max),
            "y": RangeObj((-3.14, 3.14), scale_min_max),
            "z": RangeObj((-3.14, 3.14), scale_min_max),
        }
        ang_vel_range = {
            "x": RangeObj((-40, 40), scale_min_max),
            "y": RangeObj((-40, 40), scale_min_max),
            "z": RangeObj((-30, 30), scale_min_max),
        }

        return {
            "position_range": position_range,
            "vel_range": vel_range,
            "acc_range": acc_range,
            "ori_range": ori_range,
            "ang_range": ang_range,
            "ang_vel_range": ang_vel_range,
        }

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

        self.rviz_angle_pub = rospy.Publisher(
            self.name_space + "/rqt_plot_euler", Point, queue_size=1
        )

    def _imu_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        if self.real_exp:
            self.acc_data.value = msg.linear_acceleration
            self.acc_data.value.z += GRAVITY
        else:
            self.acc_data.value = msg.linear_acceleration
            self.acc_data.value.z -= GRAVITY

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 100 == 0:
                print(
                    "[ KinematicObservation ] imu_callback: linear_acceleration",
                    self.acc_data.value,
                )

    def _pose_callback(self, msg):
        """pose msg callback

        Args:
            msg ([uav_pose]): gcs EKF processed sensor data
        """

        self.position_data.value = msg.position
        self.vel_data.value = msg.velocity
        self.ori_data.value = msg.orientation
        self.ang_vel_data.value = msg.angVelocity

        quat = msg.orientation
        euler = self.data_processor.euler_from_quaternion(
            quat.x, quat.y, quat.z, quat.w
        )
        self.ang_data.value = Point(*euler)
        self.rviz_angle_pub.publish(self.ang_data.value)

        if self.dbg_ros:
            print(
                "[ KinematicObservation ] pose_callback: position",
                self.position_data.value,
            )
            print(
                "[ KinematicObservation ] pose_callback: velocity", self.vel_data.value
            )
            print(
                "[ KinematicObservation ] pose_callback: orientation",
                self.ori_data.value,
            )
            print(
                "[ KinematicObservation ] pose_callback: angle",
                self.ang_data.value,
            )
            print(
                "[ KinematicObservation ] pose_callback: ang_vel",
                self.ang_vel_data.value,
            )

    def check_connection(self):
        """check ros connection"""
        while (
            self.position_data.value.x == 0.0
            and self.position_data.value.y == 0.0
            and self.position_data.value.z == 0.0
        ):
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

            self.position_data.value = pose_data.position

        rospy.logdebug("pose ready")

    def observe(self) -> np.ndarray:
        observation_dict = {
            "position": self.position_data,
            "velocity": self.vel_data,
            "linear_acceleration": self.acc_data,
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
        }
        scaled_observation_dict = self.data_processor.normalize_data_obj_dict(
            observation_dict
        )

        observation, obs_info = [], {}
        for key in self.obs_name:
            observation.extend(scaled_observation_dict[str(key)])
            obs_info.update({str(key): scaled_observation_dict[str(key)]})

        if self.dbg_obs:
            print(
                "[ KinematicObservation ] observe: observation_dict", observation_dict
            )
            print(
                "[ KinematicObservation ] observe: obs_info",
                obs_info,
            )

        return np.array(observation), obs_info


class KinematicsGoalObservation(KinematicObservation):
    """Goal environment observation"""

    Quat_OBS = [
        "position",
        "velocity",
        "linear_acceleration",
        "orientation",
        "angular_velocity",
    ]
    Euler_OBS = [
        "position",
        "velocity",
        "linear_acceleration",
        "angle",
        "angular_velocity",
    ]

    def __init__(
        self,
        env: "AbstractEnv",
        orientation_type="euler",
        action_feedback=False,
        goal_obs_diff_feedback=False,
        **kwargs: dict
    ) -> None:
        super().__init__(env, **kwargs)
        if orientation_type == "euler":
            self.obs_name = self.Euler_OBS
            self.obs_dim = 15
        elif orientation_type == "quaternion":
            self.obs_name = self.Quat_OBS
            self.obs_dim = 16
        else:
            raise ValueError(
                "unknown orientation type, should be one of [euler, quaternion]"
            )

        self.action_feedback = action_feedback
        if self.action_feedback:
            act_dim = 8
            self.obs_dim += act_dim
            self.obs_name.append("action")

        self.goal_obs_diff_feedback = goal_obs_diff_feedback
        if self.goal_obs_diff_feedback:
            goal_dim = 7 if orientation_type == "quaternion" else 6
            self.obs_dim += goal_dim
            self.obs_name.append("goal_obs_diff")

    def space(self) -> spaces.Space:
        obs, _ = self.observe()
        return spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["desired_goal"].shape,
                    dtype=np.float32,
                ),
                achieved_goal=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["achieved_goal"].shape,
                    dtype=np.float32,
                ),
                observation=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["observation"].shape,
                    dtype=np.float32,
                ),
            )
        )

    def observe(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        observation_dict = {
            "position": self.position_data,
            "velocity": self.vel_data,
            "linear_acceleration": self.acc_data,
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
        }
        scaled_observation_dict = self.data_processor.normalize_data_obj_dict(
            observation_dict
        )

        goal, goal_info = self.env.goal
        achieved_goal = []
        for k, _ in goal_info.items():
            achieved_goal.extend(scaled_observation_dict[k])
        achieved_goal = np.array(achieved_goal)

        if self.action_feedback:
            scaled_observation_dict.update(
                {"action": self.env.action_type.get_cur_act()}
            )
        if self.goal_obs_diff_feedback:
            scaled_observation_dict.update(
                {"goal_obs_diff": (goal - achieved_goal) / 2}
            )

        observation, observation_info = [], {}
        for key in self.obs_name:
            val = scaled_observation_dict[str(key)]
            observation.extend(val)
            observation_info.update({str(key): val})
        observation = np.array(observation)

        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": goal,
        }
        obs_info = {
            "observation": observation_info,
            "desired_goal": goal_info,
        }

        if self.dbg_obs:
            print(
                "[ KinematicsGoalObservation ] observe: observation_dict",
                observation_dict,
            )
            print(
                "[ KinematicsGoalObservation ] observe: observation_info",
                observation_info,
            )
            print("[ KinematicsGoalObservation ] observe: goal", goal)

        return obs, obs_info


class PlanarKinematicsObservation(KinematicObservation):
    """Planar kinematics observation with action feedback"""

    OBS = [
        "position",
        "velocity",
        "linear_acceleration",
        "angle",
        "angular_velocity",
    ]

    def __init__(
        self,
        env: "AbstractEnv",
        noise_stdv=0.02,
        action_feedback=True,
        enable_velocity_goal=False,
        **kwargs: dict
    ) -> None:
        super().__init__(env, **kwargs)
        self.noise_stdv = noise_stdv

        self.obs_name = self.OBS
        self.obs_dim = 5

        self.action_feedback = action_feedback
        if self.action_feedback:
            act_dim = 4
            self.obs_dim += act_dim
            self.obs_name.append("action")

        if enable_velocity_goal:
            self.obs_dim += 1

    def observe(self) -> np.ndarray:
        observation_dict = {
            "position": self.position_data,
            "velocity": self.vel_data,
            "linear_acceleration": self.acc_data,
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
        }
        scaled_observation_dict = self.data_processor.normalize_data_obj_dict(
            observation_dict
        )
        if self.action_feedback:
            action = self.env.action_type.get_cur_act()[[0, 1, 5, 6]]
            scaled_observation_dict.update({"action": action})

        observation, obs_info = [], {}
        for key in self.obs_name:
            val = self.data_processor.add_noise(
                scaled_observation_dict[str(key)], self.noise_stdv
            )
            observation.extend(val)
            obs_info.update({str(key): val})

        if self.dbg_obs:
            print(
                "[ KinematicObservation ] observe: observation_dict", observation_dict
            )
            print(
                "[ KinematicObservation ] observe: obs_info",
                obs_info,
            )

        return np.array(observation), obs_info


class RealPlanarKinematicsObservation(PlanarKinematicsObservation):
    """Planar kinematics observation with action feedback"""

    def observe(self) -> np.ndarray:
        observation_dict = {
            "position": self.position_data,
            "velocity": self.vel_data,
            "linear_acceleration": self.acc_data,
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
        }
        scaled_observation_dict = self.data_processor.normalize_data_obj_dict(
            observation_dict
        )
        if self.action_feedback:
            action = self.env.action_type.get_cur_act()[[0, 1, 5, 6]]
            action[2] = -1  ## an hack to deceive agent and it always see -1 servo
            scaled_observation_dict.update({"action": action})

        observation, obs_info = [], {}
        for key in self.obs_name:
            val = self.data_processor.add_noise(
                scaled_observation_dict[str(key)], self.noise_stdv
            )
            observation.extend(val)
            obs_info.update({str(key): val})

        if self.dbg_obs:
            print(
                "[ KinematicObservation ] observe: observation_dict", observation_dict
            )
            print(
                "[ KinematicObservation ] observe: obs_info",
                obs_info,
            )

        return np.array(observation), obs_info


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    """observation factory for different observation type"""
    if config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "PlanarKinematics":
        return PlanarKinematicsObservation(env, **config)
    elif config["type"] == "RealPlanarKinematics":
        return RealPlanarKinematicsObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
