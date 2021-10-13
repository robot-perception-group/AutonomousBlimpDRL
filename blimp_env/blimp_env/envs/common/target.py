from typing import List, TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import rospy
from blimp_env.envs.common.data_processor import DataProcessor
from blimp_env.envs.common.observation import KinematicObservation
from blimp_env.envs.common.utils import DataObj
from geometry_msgs.msg import Point, Quaternion
from gym import spaces
from librepilot.msg import AutopilotInfo
from visualization_msgs.msg import InteractiveMarkerInit
from uav_msgs.msg import uav_pose
import copy
from blimp_env.envs.script.blimp_script import respawn_target


if TYPE_CHECKING:
    from blimp_env.envs.common.abstract import AbstractEnv


class TargetType:
    """abstract target type"""

    def __init__(
        self,
        env: "AbstractEnv",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.env = env

    def space(self) -> spaces.Space:
        """get target space"""
        raise NotImplementedError

    def sample(self):
        """sample a goal"""
        raise NotImplementedError()


class PathTarget(TargetType):
    """Target of the task"""

    PATH_TARGETS = ["velocity", "angle"]
    # PATH_TARGETS = ["velocity", "orientation"]

    def __init__(
        self,
        env: "AbstractEnv",
        target_name_space="target_0",
        DBG_ROS=False,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env)

        self.data_processor = DataProcessor()
        self.target_name_space = target_name_space
        self.dbg_ros = DBG_ROS

        self.path_progress = 0
        self.path_error = 0

        self.target_dim = 6
        self.target_name = self.PATH_TARGETS

        range_dict = KinematicObservation._create_range_obj()
        self.position_cmd_data = DataObj(Point(), range_dict["position_range"])
        self.vel_cmd_data = DataObj(Point(), range_dict["vel_range"])
        self.orientation_cmd_data = DataObj(Quaternion(), range_dict["ori_range"])
        self.angle_cmd_data = DataObj(Point(), range_dict["ang_range"])

        self._pub_and_sub = False
        self._create_pub_and_sub()

    def space(self) -> spaces.Space:
        """gym space, only for testing purpose"""
        return spaces.Box(
            low=np.full((self.target_dim), -1),
            high=np.full((self.target_dim), 1),
            dtype=np.float32,
        )

    def _create_pub_and_sub(self) -> None:
        """create publicator and subscriber"""

        rospy.Subscriber(
            self.target_name_space + "/AutopilotInfo",
            AutopilotInfo,
            self._autopilot_info_callback,
        )
        self._pub_and_sub = True

    def _autopilot_info_callback(self, msg: AutopilotInfo) -> None:
        """autopilot info msg callback

        Args:
            msg ([AutopilotInfo]): autopilot command from path planner or task manager
        """
        self.vel_cmd_data.value = msg.VelocityDesired

        vec = (
            self.vel_cmd_data.value.x,
            self.vel_cmd_data.value.y,
            self.vel_cmd_data.value.z,
        )

        self.angle_cmd_data.value = Point(*self.euler_from_vec(vec))
        self.orientation_cmd_data.value = Quaternion(*self.quat_from_vec(vec))

        self.path_progress = msg.fractional_progress
        self.path_error = msg.error

        if self.dbg_ros:
            print(
                "[ target ] autopilot_info_callback: velocity_cmd",
                self.vel_cmd_data.value,
            )
            print(
                "[ target ] autopilot_info_callback: orientation_cmd",
                self.orientation_cmd_data.value,
            )
            print(
                "[ target ] autopilot_info_callback: path_progress",
                self.path_progress,
            )
            print(
                "[ target ] autopilot_info_callback: path_error",
                self.path_error,
            )

    def check_connection(self) -> None:
        """check ros connection"""
        rate = rospy.Rate(100)

        while self._pub_and_sub is not True:
            try:
                rospy.loginfo("[ target ] check_connection: waiting for target startup")
                rate.sleep()
            except rospy.ROSInterruptException as err:
                rospy.logdebug("unable to establish ros connection:", err)
                break
        rospy.logdebug("target publisher and subscriber connected")

        while self.vel_cmd_data.value.x == 0.0 and self.vel_cmd_data.value.y == 0.0:
            try:
                rospy.loginfo(
                    "[ target ] check_connection: waiting for target subscriber"
                )
                vel_cmd_data = rospy.wait_for_message(
                    self.target_name_space + "/AutopilotInfo",
                    AutopilotInfo,
                    timeout=50,
                )
                self.vel_cmd_data.value = vel_cmd_data.VelocityDesired
            except rospy.ROSInterruptException as err:
                rospy.logdebug("unable to establish ros connection:", err)
                break
        rospy.logdebug("target ready")

    def sample(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """sample target state

        Returns:
            np.ndarray: target array
            dict: target info dictionary with key specified by self.target_name
        """
        target_dict = {
            "position": self.position_cmd_data,
            "velocity": self.vel_cmd_data,
            "orientation": self.orientation_cmd_data,
            "angle": self.angle_cmd_data,
        }
        scaled_target_dict = self.data_processor.normalize_data_obj_dict(target_dict)

        target, target_info = [], {}
        for key in self.target_name:
            target.extend(scaled_target_dict[str(key)])
            target_info[key] = scaled_target_dict[str(key)]

        return np.array(target), target_info

    def quat_from_vec(
        self,
        vec: tuple,
    ) -> tuple:
        """calculate quaternion from a vector

        Args:
            vec (tuple): [a 3-D vector in NED frame]

        Returns:
            tuple: [a quaternion]
        """

        angle = self.euler_from_vec(vec)
        return self.data_processor.quaternion_from_euler(*angle)

    @classmethod
    def euler_from_vec(
        cls,
        vec: tuple,
        row_cmd: Optional[bool] = False,
        cor_u: Optional[tuple] = (0, 0, -1),
    ) -> tuple:
        """calculate euler angle from a vector

        Args:
            vec (tuple): [a 3-D vector in NED frame]
            row_cmd (Optional[bool], optional): [whether to activate row_cmd].
                Defaults to False.
            cor_u (Optional[tuple], optional): [a 3-D vector define upward
                direction of a coordination]. Defaults to (0, 0, -1).

        Returns:
            tuple: [euler angle in the order (row, yaw, pitch))]
        """
        vec_d = vec / np.linalg.norm(vec)

        # angle_p = np.arcsin(vec_d[2])
        angle_p = -np.arcsin(vec_d[2])
        angle_y = np.arctan2(vec_d[1], vec_d[0])

        if row_cmd:
            vec_w = (-vec_d[1], vec_d[0], 0)
            vec_u = np.cross(vec_w, vec_d)
            angle_r = np.arctan2(
                np.dot(vec_w, cor_u) / np.linalg.norm(vec_w),
                np.dot(vec_u, cor_u) / np.linalg.norm(vec_u),
            )
        else:
            angle_r = 0

        return (angle_r, angle_p, angle_y)


class GoalTarget(PathTarget):
    """Target of the task"""

    GOAL_TARGETS = ["position", "orientation"]
    GOAL_ANG_TARGETS = ["position", "angle"]

    def __init__(
        self,
        env: "AbstractEnv",
        orientation_type="euler",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env, **kwargs)

        if orientation_type == "euler":
            self.target_dim = 6
            self.target_name = self.GOAL_ANG_TARGETS
        elif orientation_type == "quaternion":
            self.target_dim = 7
            self.target_name = self.GOAL_TARGETS
        else:
            raise ValueError(
                "unknown orientation type, should be one of (euler, quaternion)"
            )

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.full((self.target_dim), -1),
            high=np.full((self.target_dim), 1),
            dtype=np.float32,
        )

    def _create_pub_and_sub(self) -> None:
        """create publicator and subscriber"""
        self._pub_and_sub = False
        rospy.Subscriber(
            self.target_name_space + "/update_full",
            InteractiveMarkerInit,
            self._goal_cmd_callback,
        )

        self.rviz_position_pub = rospy.Publisher(
            self.target_name_space + "/rqt_plot_point", Point, queue_size=1
        )
        self.rviz_quat_pub = rospy.Publisher(
            self.target_name_space + "/rqt_plot_quat", Quaternion, queue_size=1
        )
        self.rviz_angle_pub = rospy.Publisher(
            self.target_name_space + "/rqt_plot_euler", Point, queue_size=1
        )

        self._pub_and_sub = True

    def _goal_cmd_callback(self, msg: InteractiveMarkerInit) -> None:
        """goal command msg callback including position command and orientation command
        Command is in ENU frame. Need to convert to NED frame

        Args:
            msg ([InteractiveMarkerInit]): positon and orientation command from task manager
        """
        if self._pub_and_sub is True and msg.markers is not None:
            pose = msg.markers[0].pose

            position = pose.position
            self.position_cmd_data.value = Point(position.y, position.x, -position.z)

            quat = pose.orientation
            euler = self.data_processor.euler_from_quaternion(
                quat.x, quat.y, quat.z, quat.w
            )
            euler[2] = -euler[2]
            quat2 = self.data_processor.quaternion_from_euler(*euler)
            self.orientation_cmd_data.value = Quaternion(*quat2)
            self.angle_cmd_data.value = Point(*euler)

            self.publish_to_rviz()

        if self.dbg_ros:
            print(
                "[ GoalTarget ] _goal_cmd_callback: position",
                self.position_cmd_data.value,
            )
            print(
                "[ GoalTarget ] _goal_cmd_callback: orientation",
                self.orientation_cmd_data.value,
            )
            print(
                "[ GoalTarget ] _goal_cmd_callback: angle",
                self.angle_cmd_data.value,
            )

    def publish_to_rviz(self):
        """publish msg to rviz visualization"""
        self.rviz_position_pub.publish(self.position_cmd_data.value)
        self.rviz_quat_pub.publish(self.orientation_cmd_data.value)
        self.rviz_angle_pub.publish(self.angle_cmd_data.value)

    def check_connection(self):
        """check for ros connection"""
        rate = rospy.Rate(100)

        while self._pub_and_sub is not True:
            try:
                rospy.loginfo("[ target ] check_connection: waiting for target startup")
                rate.sleep()
            except rospy.ROSInterruptException as err:
                rospy.logdebug("unable to establish ros connection:", err)
                break
        rospy.logdebug("target publisher and subscriber connected")

        while self.position_cmd_data.value == Point(0.0, 0.0, 0.0):
            try:
                rospy.loginfo("[ target ] waiting for position subscriber")
                msg = rospy.wait_for_message(
                    self.target_name_space + "/update_full",
                    InteractiveMarkerInit,
                    timeout=100,
                )
                position = msg.markers[0].pose.position
                self.position_cmd_data.value = Point(
                    position.y, position.x, -position.z
                )
            except TimeoutError:
                rospy.loginfo(
                    "[ target ] unable to establish ros connection, respawn..."
                )
                reply = respawn_target(**self.env.config["simulation"])
                rospy.loginfo("target respawn:", reply)

        rospy.logdebug("target ready")


class PlanarGoal(GoalTarget):
    """planar goal is designed for planar navigate environment
    goal z position is now modified depend on the path vector
    between current goal and previous goal
    this is done by projecting the current machine position vector
    to path vector and compute the desired z position
    """

    def __init__(
        self,
        env: "AbstractEnv",
        name_space="machine_0",
        enable_velocity_goal=False,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        if enable_velocity_goal:
            self.target_dim += 1
            self.target_name.append("velocity")

        self.planar_pos_cmd_rviz_publisher = rospy.Publisher(
            name_space + "/rviz_goal_position", Point, queue_size=1
        )

        if enable_velocity_goal:
            self.target_dim += 1
            self.target_name.append("velocity")
            rospy.Subscriber(
                self.target_name_space + "/AutopilotInfo",
                AutopilotInfo,
                self._autopilot_info_callback,
            )

    def sample(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """sample target state

        Returns:
            np.ndarray: target array
            dict: target info dictionary with key specified by self.target_name
        """
        target_dict = {
            "position": self.position_cmd_data,
            "velocity": self.vel_cmd_data,
            "orientation": self.orientation_cmd_data,
            "angle": self.angle_cmd_data,
        }
        scaled_target_dict = self.data_processor.normalize_data_obj_dict(target_dict)

        target, target_info = [], {}
        for key in self.target_name:
            target.extend(scaled_target_dict[str(key)])
            target_info[key] = scaled_target_dict[str(key)]

        self.planar_pos_cmd_rviz_publisher.publish(
            Point(*self.dataobj_to_array(self.position_cmd_data))
        )

        return np.array(target), target_info

    @classmethod
    def dataobj_to_array(cls, dataobj):
        """convert a data object to numpy array

        Args:
            dataobj ([DataObj]): [a data object has a value attribute]

        Returns:
            [numpy array]: [value of data object]
        """
        return np.array([dataobj.value.x, dataobj.value.y, dataobj.value.z])


def target_factory(env: "AbstractEnv", config: dict) -> TargetType:
    """generate different types of target

    Args:
        config (dict): [config should specify target type]

    Returns:
        TargetType: [a target will generate goal for RL agent]
    """
    if config["type"] == "PATH":
        return PathTarget(env, **config)
    elif config["type"] == "GOAL":
        return GoalTarget(env, **config)
    elif config["type"] == "PlanarGoal":
        return PlanarGoal(env, **config)
    else:
        raise ValueError("Unknown target type")
