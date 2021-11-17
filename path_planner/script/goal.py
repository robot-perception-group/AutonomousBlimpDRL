#!/usr/bin/env python

"""
Copyright (c) 2011, Willow Garage, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Willow Garage, Inc. nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import rospy
import copy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Quaternion, Point
from tf.broadcaster import TransformBroadcaster
from tf.transformations import quaternion_from_euler
from librepilot.msg import AutopilotInfo

import rospy
import copy
import time
import tf
from random import random
from math import pi, sin
import numpy as np


NAME_SPACE = "machine_"
GOAL_NAME = "goal_"

server = None
menu_handler = MenuHandler()
br = None
counter = 0


def frameCallback(msg):
    global counter, br
    time = rospy.Time.now()
    br.sendTransform((0, 0, 0), (0, 0, 0, 1.0), time, "goal_link", "world")
    counter += 1


def processFeedback(feedback):
    s = "Feedback from marker '" + feedback.marker_name
    s += "' / control '" + feedback.control_name + "'"

    mp = ""
    if feedback.mouse_point_valid:
        mp = " at " + str(feedback.mouse_point.x)
        mp += ", " + str(feedback.mouse_point.y)
        mp += ", " + str(feedback.mouse_point.z)
        mp += " in frame " + feedback.header.frame_id

    if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
        rospy.loginfo(s + ": button click" + mp + ".")
    elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
        rospy.loginfo(
            s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "."
        )
    elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        rospy.loginfo(s + ": pose changed")
    elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
        rospy.loginfo(s + ": mouse down" + mp + ".")
    elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
        rospy.loginfo(s + ": mouse up" + mp + ".")
    server.applyChanges()


def alignMarker(feedback):
    pose = feedback.pose

    pose.position.x = round(pose.position.x - 0.5) + 0.5
    pose.position.y = round(pose.position.y - 0.5) + 0.5

    rospy.loginfo(
        feedback.marker_name
        + ": aligning position = "
        + str(feedback.pose.position.x)
        + ","
        + str(feedback.pose.position.y)
        + ","
        + str(feedback.pose.position.z)
        + " to "
        + str(pose.position.x)
        + ","
        + str(pose.position.y)
        + ","
        + str(pose.position.z)
    )

    server.setPose(feedback.marker_name, pose)
    server.applyChanges()


def rand(min_, max_):
    return min_ + random() * (max_ - min_)


def makeBox(msg):
    marker = Marker()

    marker.type = Marker.CUBE
    marker.scale.x = msg.scale * 0.45
    marker.scale.y = msg.scale * 0.45
    marker.scale.z = msg.scale * 0.45
    marker.color.r = 0.5
    marker.color.g = 0.5
    marker.color.b = 0.5
    marker.color.a = 1.0

    return marker


def makeBoxControl(msg):
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(makeBox(msg))
    msg.controls.append(control)
    return control


def saveMarker(int_marker):
    server.insert(int_marker, processFeedback)


# Marker Creation
def makeQuadrocopterMarker(position, orientation):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "goal_link"
    int_marker.pose.position = position
    int_marker.pose.orientation = orientation
    int_marker.scale = 1

    int_marker.name = "goal"
    int_marker.description = "goal"

    makeBoxControl(int_marker)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE
    int_marker.controls.append(copy.deepcopy(control))
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)

    server.insert(int_marker, processFeedback)


def generate_goal(x_min, x_max, y_min, y_max, z_min, z_max, v_min, v_max):
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    v = np.random.uniform(v_min, v_max)

    phi = 0
    the = 0
    psi = np.random.uniform(-pi, pi)
    q = quaternion_from_euler(phi, the, psi)
    return x, y, z, v, phi, the, psi, q


def distance_to_origin_far_enough(position, origin=np.array([0, 0, 100])):
    dist = np.linalg.norm(position - origin)
    if dist > 30:
        return True
    else:
        return False


def sample_new_goal(x_min, x_max, y_min, y_max, z_min, z_max, v_min, v_max):
    far_enough = False
    while far_enough == False:
        x, y, z, v, phi, the, psi, q = generate_goal(
            x_min, x_max, y_min, y_max, z_min, z_max, v_min, v_max
        )
        far_enough = distance_to_origin_far_enough(np.array([x, y, z]))
    return x, y, z, v, phi, the, psi, q


if __name__ == "__main__":
    robotID = "0"
    x_max, x_min = 105, -105
    y_max, y_min = 105, -105
    z_max, z_min = 210, 5
    v_max, v_min = 8, 0

    if len(sys.argv) > 1:
        robotID = sys.argv[1]
        x_max, x_min = float(sys.argv[2]), -float(sys.argv[2])
        y_max, y_min = float(sys.argv[3]), -float(sys.argv[3])
        z_max, z_min = float(sys.argv[4]), float(sys.argv[5])
        v_max, v_min = float(sys.argv[6]), 0

    np.random.seed(123 + int(robotID))

    rospy.loginfo("[ Goal Node ] Launching...")
    rospy.init_node("GOAL_Node_" + robotID, anonymous=False)
    vel_cmd_pub = rospy.Publisher(
        "goal_" + robotID + "/AutopilotInfo", AutopilotInfo, queue_size=1
    )
    br = TransformBroadcaster()
    server = InteractiveMarkerServer("goal_" + robotID)

    menu_handler.insert("First Entry", callback=processFeedback)
    menu_handler.insert("Second Entry", callback=processFeedback)
    sub_menu_handle = menu_handler.insert("Submenu")
    menu_handler.insert("First Entry", parent=sub_menu_handle, callback=processFeedback)
    menu_handler.insert(
        "Second Entry", parent=sub_menu_handle, callback=processFeedback
    )

    times = 0
    while not rospy.is_shutdown():
        if times % 180 == 0:
            x, y, z, v, phi, the, psi, q = sample_new_goal(
                x_min, x_max, y_min, y_max, z_min, z_max, v_min, v_max
            )
            times = 0

            x, y, z = 50, 50, 100  # hack to fix the goal for testing impala
            v = 5

        position = Point(x, y, z)
        velocity = Point(v, 0, 0)
        orientation = Quaternion(q[0], q[1], q[2], q[3])
        makeQuadrocopterMarker(position=position, orientation=orientation)
        server.applyChanges()

        vel_cmd = AutopilotInfo()
        vel_cmd.VelocityDesired = velocity
        vel_cmd_pub.publish(vel_cmd)

        rospy.Timer(rospy.Duration(0.01), frameCallback)

        rospy.loginfo("[ Goal Node ] -----------------------")
        rospy.loginfo("[ Goal Node ] position = (%2.1f, %2.1f, %2.1f)\n" % (x, y, z))
        rospy.loginfo("[ Goal Node ] velocity = %2.1f\n" % (v))
        rospy.loginfo(
            "[ Goal Node ] orientation = (%2.1f, %2.1f, %2.1f)]\n" % (phi, the, psi)
        )

        times += 1
        time.sleep(1)
