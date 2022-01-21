from math import pi
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from blimp_env.envs.common import utils
from blimp_env.envs.script.blimp_script import respawn_target, spawn_target
from gym import spaces
from librepilot.msg import AutopilotInfo
from transforms3d.euler import euler2quat, quat2euler
from visualization_msgs.msg import InteractiveMarkerInit, Marker, MarkerArray
import time

if TYPE_CHECKING:
    from blimp_env.envs.common.abstract import AbstractEnv


class WayPoint:
    def __init__(self, position=np.zeros(3), velocity=0.0, angle=np.zeros(3)):
        self.position = np.array(position)
        self.velocity = float(velocity)
        self.angle = np.array(angle)


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


class RandomGoal(TargetType):
    """a random generated goal during training"""

    def __init__(
        self,
        env: "AbstractEnv",
        target_name_space="target_0",
        new_target_every_ts: int = 1200,
        DBG_ROS=False,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env)

        self.target_name_space = target_name_space
        self.dbg_ros = DBG_ROS
        self.target_dim = 9

        self.pos_cmd_data = np.array([0, 0, 0])
        self.vel_cmd_data = 0.0
        self.ang_cmd_data = np.array([0, 0, 0])

        self.new_target_every_ts = new_target_every_ts
        self.x_range = [-105, 105]
        self.y_range = [-105, 105]
        self.z_range = [-5, -210]
        self.v_range = [3, 8]

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
        self.wp_viz_publisher = rospy.Publisher(
            self.target_name_space + "/rviz_pos_cmd", Marker, queue_size=1
        )
        self._pub_and_sub = True

    def publish_waypoint_toRviz(self, waypoint):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.id = 0
        marker.scale.x, marker.scale.y, marker.scale.z = 2, 2, 2
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1, 1, 1, 0
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            waypoint[1],
            waypoint[0],
            -waypoint[2],
        )  ## NED --> rviz(ENU)
        marker.pose.orientation.w = 1
        self.wp_viz_publisher.publish(marker)

    def generate_goal(self):
        x = np.random.uniform(*self.x_range)
        y = np.random.uniform(*self.y_range)
        z = np.random.uniform(*self.z_range)
        pos_cmd = np.array([x, y, z])

        v_cmd = np.random.uniform(*self.v_range)

        phi, the = 0, 0
        psi = np.random.uniform(-pi, pi)
        ang_cmd = np.array([phi, the, psi])
        q_cmd = euler2quat(0, 0, psi)
        return pos_cmd, v_cmd, ang_cmd, q_cmd

    def check_planar_distance_to_origin(
        self, position, origin=np.array([0, 0, 100]), min_dist=30
    ):
        dist = np.linalg.norm(position[0:2] - origin[0:2])
        return dist > min_dist

    def sample_new_goal(self):
        far_enough = False
        while far_enough == False:
            pos_cmd, v_cmd, ang_cmd, _ = self.generate_goal()
            far_enough = self.check_planar_distance_to_origin(pos_cmd)

        self.pos_cmd_data = pos_cmd
        self.vel_cmd_data = v_cmd
        self.ang_cmd_data = ang_cmd

    def sample(self) -> Dict[str, np.ndarray]:
        """sample target state

        Returns:
            dict: target info dictionary with key specified by self.target_name
        """
        if self.env.steps % self.new_target_every_ts == 0:
            self.sample_new_goal()
        self.publish_waypoint_toRviz(self.pos_cmd_data)
        return {
            "position": self.pos_cmd_data,
            "velocity": self.vel_cmd_data,
            "angle": self.ang_cmd_data,
        }

    def check_connection(self):
        pass


class MultiGoal(TargetType):
    """a specified goal sequences."""

    def __init__(
        self,
        env: "AbstractEnv",
        target_name_space="goal_0",
        trigger_dist=5,  # meter
        wp_list=[
            (50, 50, -30, 5),
            (50, -50, -30, 5),
            (-50, -50, -30, 5),
            (-50, 50, -30, 5),
        ],
        DBG_ROS=False,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env)

        self.target_name_space = target_name_space
        self.dbg_ros = DBG_ROS
        self.target_dim = 9

        self.wp_list = []
        for wp in wp_list:
            self.wp_list.append(WayPoint(wp[0:3], wp[3]))

        self.trigger_dist = trigger_dist
        self.wp_max_index = len(self.wp_list) - 1
        self.wp_index = 0

        self.pos_cmd_data = self.wp_list[0].position
        self.vel_cmd_data = self.wp_list[0].velocity
        self.ang_cmd_data = self.wp_list[0].angle

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
        self.wp_viz_publisher = rospy.Publisher(
            self.target_name_space + "/rviz_pos_cmd", Marker, queue_size=1
        )
        self._pub_and_sub = True

    def publish_waypoint_toRviz(self, waypoint):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.id = 0
        marker.scale.x, marker.scale.y, marker.scale.z = 2, 2, 2
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1, 1, 1, 0
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            waypoint[1],
            waypoint[0],
            -waypoint[2],
        )  ## NED --> rviz(ENU)
        marker.pose.orientation.w = 1
        self.wp_viz_publisher.publish(marker)

    def sample_new_goal(self):
        self.wp_index += 1

        if self.wp_index >= self.wp_max_index:
            self.wp_index = 0

        pos_cmd = self.wp_list[self.wp_index].position
        v_cmd = self.wp_list[self.wp_index].velocity
        ang_cmd = self.wp_list[self.wp_index].angle
        q_cmd = euler2quat(*ang_cmd)
        return pos_cmd, v_cmd, ang_cmd, q_cmd

    def check_planar_distance(self, goal_pos, machine_pos, trigger_dist=5):
        return np.linalg.norm(goal_pos[0:2] - machine_pos[0:2]) < trigger_dist

    def sample(self) -> Dict[str, np.ndarray]:
        """sample target state

        Returns:
            dict: target info dictionary with key specified by self.target_name
        """

        if self.env._pub_and_sub and self.check_planar_distance(
            self.pos_cmd_data, self.env.observation_type.pos_data, self.trigger_dist
        ):
            (
                self.pos_cmd_data,
                self.vel_cmd_data,
                self.ang_cmd_data,
                _,
            ) = self.sample_new_goal()

        self.publish_waypoint_toRviz(self.pos_cmd_data)
        return {
            "position": self.pos_cmd_data,
            "velocity": self.vel_cmd_data,
            "angle": self.ang_cmd_data,
        }

    def check_connection(self):
        pass


class ROSTarget(TargetType):
    """ROS Abstract Target"""

    def __init__(
        self,
        env: "AbstractEnv",
        target_name_space="target_0",
        DBG_ROS=False,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env)
        respawn_target(**kwargs)

        self.target_name_space = target_name_space
        self.dbg_ros = DBG_ROS
        self.target_dim = 9

        self.pos_cmd_data = np.array([0, 0, 0])
        self.vel_cmd_data = 0.0
        self.ang_cmd_data = np.array([0, 0, 0])

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
        rospy.Subscriber(
            self.target_name_space + "/update_full",
            InteractiveMarkerInit,
            self._goal_cmd_callback,
        )
        self._pub_and_sub = True

    def _autopilot_info_callback(self, msg: AutopilotInfo) -> None:
        """autopilot info msg callback

        Args:
            msg ([AutopilotInfo]): autopilot command from path planner or task manager
        """
        self.vel_cmd_data = msg.VelocityDesired.x

        if self.dbg_ros:
            print(
                "[ Target ] velocity_cmd: ",
                self.vel_cmd_data,
            )

    def _goal_cmd_callback(self, msg: InteractiveMarkerInit) -> None:
        """goal command msg callback including position command and orientation command
        Command is in ENU frame. Need to convert to NED frame

        Args:
            msg ([InteractiveMarkerInit]): positon and orientation command from task manager
        """
        if self._pub_and_sub is True and msg.markers is not None:
            pose = msg.markers[0].pose

            pos = pose.position
            self.pos_cmd_data = np.array([pos.y, pos.x, -pos.z])

            quat = utils.obj2array(pose.orientation)
            self.ang_cmd_data = quat2euler(quat)

        if self.dbg_ros:
            print(
                "[ Target ] position: ",
                self.pos_cmd_data,
            )
            print(
                "[ Target ] angle: ",
                self.ang_cmd_data,
            )

    def check_connection(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class InteractiveGoal(ROSTarget):
    """a waypoint in 3D space defined by desired position, velocity and yaw angle"""

    def sample(self) -> Dict[str, np.ndarray]:
        """sample target state

        Returns:
            dict: target info dictionary with key specified by self.target_name
        """
        return {
            "position": self.pos_cmd_data,
            "velocity": self.vel_cmd_data,
            "angle": self.ang_cmd_data,
        }

    def check_connection(self) -> None:
        """check ros connection"""
        while self._pub_and_sub is not True:
            try:
                rospy.logdebug("[ target ] waiting for target startup")
                time.sleep(1)
            except rospy.ROSInterruptException as err:
                rospy.logdebug("unable to establish ros connection:", err)
                break
        rospy.logdebug("target publisher and subscriber started")

        while self.vel_cmd_data == 0.0:
            try:
                rospy.logdebug("[ target ] waiting for velocity subscriber")
                vel_cmd_data = rospy.wait_for_message(
                    self.target_name_space + "/AutopilotInfo",
                    AutopilotInfo,
                    timeout=100,
                )
                self.vel_cmd_data = vel_cmd_data.VelocityDesired.x
            except TimeoutError:
                self.timeout_handle()

        while (self.pos_cmd_data == np.array([0.0, 0.0, 0.0])).all():
            try:
                rospy.logdebug("[ target ] waiting for position subscriber")
                msg = rospy.wait_for_message(
                    self.target_name_space + "/update_full",
                    InteractiveMarkerInit,
                    timeout=100,
                )
                position = msg.markers[0].pose.position
                self.pos_cmd_data = np.array([position.y, position.x, -position.z])
            except TimeoutError:
                self.timeout_handle()
        rospy.logdebug("target ready")

    def timeout_handle(self):
        rospy.logdebug("[ target ] unable to establish ros connection, respawn...")
        reply = respawn_target(**self.env.config["target"])
        rospy.logdebug("target respawn:", reply)
        return reply


def target_factory(env: "AbstractEnv", config: dict) -> TargetType:
    """generate different types of target

    Args:
        config (dict): [config should specify target type]

    Returns:
        TargetType: [a target will generate goal for RL agent]
    """
    if config["type"] == "RandomGoal":
        return RandomGoal(env, **config)
    elif config["type"] == "MultiGoal":
        return MultiGoal(env, **config)
    elif config["type"] == "InteractiveGoal":
        return InteractiveGoal(env, **config)
    else:
        raise ValueError("Unknown target type")
