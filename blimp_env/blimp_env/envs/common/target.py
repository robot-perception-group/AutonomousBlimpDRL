from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from blimp_env.envs.script.blimp_script import respawn_target
from blimp_env.envs.common import utils
from gym import spaces
from librepilot.msg import AutopilotInfo
from transforms3d.euler import quat2euler
from visualization_msgs.msg import InteractiveMarkerInit

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


class PlanarGoal(ROSTarget):
    """planar goal is designed for planar navigate environment
    goal z position is now modified depend on the path vector
    between current goal and previous goal
    this is done by projecting the current machine position vector
    to path vector and compute the desired z position
    """

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
        rate = rospy.Rate(100)

        while self._pub_and_sub is not True:
            try:
                rospy.logdebug("[ target ] waiting for target startup")
                rate.sleep()
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
        reply = respawn_target(**self.env.config["simulation"])
        rospy.logdebug("target respawn:", reply)


def target_factory(env: "AbstractEnv", config: dict) -> TargetType:
    """generate different types of target

    Args:
        config (dict): [config should specify target type]

    Returns:
        TargetType: [a target will generate goal for RL agent]
    """
    if config["type"] == "PlanarGoal":
        return PlanarGoal(env, **config)
    else:
        raise ValueError("Unknown target type")
