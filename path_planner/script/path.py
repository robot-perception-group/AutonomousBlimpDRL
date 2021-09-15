#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from math import pi

import sys
import numpy as np
import rospy
from geometry_msgs.msg import Point
from librepilot.msg import AutopilotInfo, LibrepilotActuators
from uav_msgs.msg import uav_pose
from visualization_msgs.msg import Marker, MarkerArray

NAME_SPACE = "machine_"
PATH_NAME = "target_"


class Status:
    def __init__(self):
        self.path_index = 0
        self.path_vector = np.zeros(3)
        self.correction_vector = np.zeros(3)
        self.fractional_progress = 0
        self.error = 0
        self.desired_velocity = np.zeros(3)


class WayPoint:
    def __init__(self, position):
        self.North = position[0]
        self.East = position[1]
        self.Down = position[2]


class Path:
    def __init__(self, start_position, start_velocity, end_position, end_velocity):
        self.Start = WayPoint(start_position)
        self.End = WayPoint(end_position)
        self.StartingVelocity = start_velocity
        self.EndingVelocity = end_velocity


class PathFollower:
    def __init__(
        self,
        robotID="0",
        SLEEP_RATE=2,
        Dbg=True,
    ):
        rospy.loginfo("[ PathFollower Node ] Initialising...")
        rospy.init_node("PathFollower_node_" + robotID, anonymous=False)

        self.RATE = rospy.Rate(SLEEP_RATE)
        self.robotID = robotID
        self.name_space = NAME_SPACE + robotID
        self.publish_name_space = PATH_NAME + robotID
        self.Dbg = Dbg

        self.path_progress_threshold = 0.80

        self.cur_position = np.zeros(3)
        self.desiredVector_upper_bnd = np.array([5, 5, 0.5])
        self.desiredVector_lower_bnd = np.array([-5, -5, -0.5])

        self.gps_init = False
        self.pub_and_sub_created = False
        self._create_pubs_subs()

        rospy.loginfo("[ PathFollower Node ] Initialization Finished ")

    #####################################
    ######## PUB and SUB METHODS ########
    #####################################
    def _create_pubs_subs(self):
        rospy.loginfo("[ PathFollower Node ] Create Subscribers and Publishers...")

        """ create subscribers """
        rospy.Subscriber(self.name_space + "/pose", uav_pose, self._pose_callback)

        """ create publishers """
        self.status_publisher = rospy.Publisher(
            self.publish_name_space + "/AutopilotInfo", AutopilotInfo, queue_size=1
        )

        self.path_viz_publisher = rospy.Publisher(
            self.publish_name_space + "/path", Marker, queue_size=1
        )

        self.wp_viz_publisher = rospy.Publisher(
            self.publish_name_space + "/waypoint", MarkerArray, queue_size=10
        )

        self.velCmd_viz_publisher = rospy.Publisher(
            self.publish_name_space + "/velocityCmd", Marker, queue_size=1
        )

        self.velCmd_rqt_publisher = rospy.Publisher(
            self.publish_name_space + "/velocityCmd_rqt", Point, queue_size=1
        )

        rospy.loginfo("[ PathFollower Node ] Subscribers and Publishers Created")
        self.pub_and_sub_created = True

    def _pose_callback(self, msg):
        if (self.pub_and_sub_created) and (msg is not None):
            self.cur_position = np.array(
                [msg.position.x, msg.position.y, msg.position.z]
            )
            self.gps_init = True

    # -----------------------------------

    def publish_status(self, path, cur_position, status):
        status_msg = AutopilotInfo()
        status_msg.fractional_progress = status.fractional_progress
        status_msg.error = status.error
        (
            status_msg.pathDirection.x,
            status_msg.pathDirection.y,
            status_msg.pathDirection.z,
        ) = status.path_vector
        (
            status_msg.pathCorrection.x,
            status_msg.pathCorrection.y,
            status_msg.pathCorrection.z,
        ) = status.correction_vector
        status_msg.Start.x, status_msg.Start.y, status_msg.Start.z = (
            path.Start.North,
            path.Start.East,
            path.Start.Down,
        )
        status_msg.End.x, status_msg.End.y, status_msg.End.z = (
            path.End.North,
            path.End.East,
            path.End.Down,
        )
        status_msg.StartingVelocity = path.StartingVelocity
        status_msg.EndingVelocity = path.EndingVelocity
        (
            status_msg.VelocityDesired.x,
            status_msg.VelocityDesired.y,
            status_msg.VelocityDesired.z,
        ) = status.desired_velocity
        self.status_publisher.publish(status_msg)

    def publish_path_toRviz(self, path):
        tail = Point(
            path.Start.East, path.Start.North, -path.Start.Down
        )  ## NED --> rviz(ENU)
        tip = Point(path.End.East, path.End.North, -path.End.Down)  ## NED --> rviz(ENU)

        marker = Marker()
        marker.action = marker.ADD
        marker.header.frame_id = "/world"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = marker.ARROW
        marker.scale.x, marker.scale.y, marker.scale.z = (0.4, 0.8, 1.0)
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = (
            1,
            0 + 0.20 * int(self.robotID),
            0 + 0.10 * int(self.robotID),
            1 - 0.20 * int(self.robotID),
        )
        marker.pose.orientation.w, marker.pose.orientation.y = 1, 0
        marker.points = [tail, tip]
        self.path_viz_publisher.publish(marker)

    def publish_waypoint_toRviz(self, wp_position_list):
        markerArray = MarkerArray()

        for wp_position in wp_position_list:
            marker = Marker()
            marker.header.frame_id = "/world"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x, marker.scale.y, marker.scale.z = (1, 1, 1)
            marker.color.a, marker.color.r, marker.color.g, marker.color.b = (
                1,
                1,
                1,
                0,
            )
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
                wp_position[1],
                wp_position[0],
                -wp_position[2],
            )  ## NED --> rviz(ENU)
            marker.pose.orientation.w = 1
            markerArray.markers.append(marker)

            id = 0
            for m in markerArray.markers:
                m.id = id
                id += 1

            self.wp_viz_publisher.publish(markerArray)

    def publish_desiredVelocity_toRviz(self, desiredVelocity):
        desiredArrow = self.cur_position + desiredVelocity
        tail = Point(
            self.cur_position[1], self.cur_position[0], -self.cur_position[2]
        )  ## NED --> rviz(ENU)
        tip = Point(
            desiredArrow[1], desiredArrow[0], -desiredArrow[2]
        )  ## NED --> rviz(ENU)

        marker = Marker()
        marker.action = marker.ADD
        marker.header.frame_id = "/world"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = marker.ARROW
        marker.scale.x, marker.scale.y, marker.scale.z = (0.2, 0.4, 0.5)
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = (1, 1, 0, 1)
        marker.pose.orientation.w, marker.pose.orientation.y = 1, 0
        marker.points = [tail, tip]
        self.velCmd_viz_publisher.publish(marker)

        point = Point()
        point.x, point.y, point.z = (
            desiredVelocity[0],
            desiredVelocity[1],
            desiredVelocity[2],
        )  ## NED
        self.velCmd_rqt_publisher.publish(point)

    # -----------------------------------

    def computeDesiredVector(self, path_vector, correction_vector, k=1):
        path_vector = np.array(path_vector)
        correction_vector = np.array(correction_vector)
        desire_vector = (1 - k) * path_vector + k * correction_vector
        desire_vector = self.clip_vector(
            desire_vector, self.desiredVector_upper_bnd, self.desiredVector_lower_bnd
        )
        if self.Dbg:
            print("[ compute_desired_vector ] desire_vector: ", desire_vector)
        return desire_vector

    def clip_vector(self, vector, upper_bnd, lower_bnd):
        cond = np.where(vector > upper_bnd)
        vector[cond] = upper_bnd[cond]
        cond = np.where(vector < lower_bnd)
        vector[cond] = lower_bnd[cond]
        return vector

    def computePathVector(self, path, cur_point, status):
        path_vector_North = path.End.North - path.Start.North
        path_vector_East = path.End.East - path.Start.East
        path_vector_Down = path.End.Down - path.Start.Down
        status.path_vector = [path_vector_North, path_vector_East, path_vector_Down]

        diff = np.zeros(3)
        diff[0] = cur_point[0] - path.Start.North
        diff[1] = cur_point[1] - path.Start.East
        diff[2] = cur_point[2] - path.Start.Down

        dot = (
            status.path_vector[0] * diff[0]
            + status.path_vector[1] * diff[1]
            + status.path_vector[2] * diff[2]
        )
        dist_path = np.linalg.norm(status.path_vector)
        status.fractional_progress = dot / (dist_path ** 2)
        if status.fractional_progress <= 0:
            status.fractional_progress = 0
        if self.Dbg:
            print(
                "[ computePathVector ] fractional_progress:", status.fractional_progress
            )

        track_point_0 = (
            status.fractional_progress * status.path_vector[0] + path.Start.North
        )
        track_point_1 = (
            status.fractional_progress * status.path_vector[1] + path.Start.East
        )
        track_point_2 = (
            status.fractional_progress * status.path_vector[2] + path.Start.Down
        )
        track_point = (track_point_0, track_point_1, track_point_2)

        status.correction_vector[0] = track_point[0] - cur_point[0]
        status.correction_vector[1] = track_point[1] - cur_point[1]
        status.correction_vector[2] = track_point[2] - cur_point[2]
        status.error = np.linalg.norm(status.correction_vector)
        if self.Dbg:
            print("[ computePathVector ] correction_vector:", status.correction_vector)

        velocity = (
            1 - status.fractional_progress
        ) * path.StartingVelocity + status.fractional_progress * path.EndingVelocity

        status.path_vector[0] = velocity * status.path_vector[0] / dist_path
        status.path_vector[1] = velocity * status.path_vector[1] / dist_path
        status.path_vector[2] = velocity * status.path_vector[2] / dist_path
        if self.Dbg:
            print("[ computePathVector ] path_vector", status.path_vector)

        return status

    # -----------------------------------

    def create_path_list(self, mission: dict):
        wp_position_list = mission["waypoints"]
        wp_velocity_list = mission["waypoints_velocity"]

        path_list = []
        for i in range(len(wp_position_list)):
            start = i
            end = 0 if i == len(wp_position_list) - 1 else i + 1

            start_position = wp_position_list[start]
            start_velocity = wp_velocity_list[start]
            end_position = wp_position_list[end]
            end_velocity = wp_velocity_list[end]
            path = Path(start_position, start_velocity, end_position, end_velocity)
            path_list.append(path)

            if self.Dbg:
                print(
                    f"[ create_path_list ] Path{i} start:{start_position} end:{end_position}"
                )

        self.path_list = path_list

    def path_select(self, status):
        path_num = len(self.path_list)
        if status.fractional_progress > self.path_progress_threshold:
            if status.path_index >= path_num - 1:
                status.path_index = 0
            else:
                status.path_index += 1

            print("[ path_select ] new path:", status.path_index)

        if self.Dbg:
            print(" [ path_select ] path_index", status.path_index)
        return status

    def update_status(self, cur_position, status):
        status = self.path_select(status)
        path = self.path_list[status.path_index]
        status = self.computePathVector(path, cur_position, status)
        return status

    def loop(self, missionID="0"):
        mission_name = "/mission_" + missionID
        mission = rospy.get_param(mission_name)
        print("mission waypoints: ", mission["waypoints"])
        print("mission velocity: ", mission["waypoints_velocity"])

        self.create_path_list(mission)

        status = Status()
        while not rospy.is_shutdown():
            status = self.update_status(self.cur_position, status)
            path = self.path_list[status.path_index]
            status.desired_velocity = self.computeDesiredVector(
                status.path_vector, status.correction_vector, k=0.2
            )

            self.publish_status(path, self.cur_position, status)
            self.publish_waypoint_toRviz(mission["waypoints"])
            self.publish_path_toRviz(path)
            self.publish_desiredVelocity_toRviz(status.desired_velocity)
            self.RATE.sleep()


if __name__ == "__main__":
    robotID = "0"
    if len(sys.argv) > 1:
        robotID = sys.argv[1]

    pathFollower = PathFollower(robotID=robotID)
    pathFollower.loop(missionID=robotID)
