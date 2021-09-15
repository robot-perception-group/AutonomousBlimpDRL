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
from uav_msgs.msg import uav_pose

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Quaternion, Point
from tf.broadcaster import TransformBroadcaster
from tf.transformations import quaternion_from_euler

import rospy
import copy
import time
from random import random
from math import pi
import numpy as np


NAME_SPACE = "machine_"
GOAL_NAME = "goal_"

server = None
menu_handler = MenuHandler()
br = None
counter = 0

goalID = "0"
position_matrix = [(50, 50, 50), (-50, 50, 50), (-50, -50, 50), (50, -50, 50)]

if len(sys.argv) > 1:
    goalID = sys.argv[1]


def frameCallback(msg):
    global counter, br
    time = rospy.Time.now()
    br.sendTransform((0, 0, 0), (0, 0, 0, 1.0), time, "goal_link" + goalID, "world")
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
    int_marker.header.frame_id = "goal_link" + goalID
    int_marker.pose.position = position
    int_marker.pose.orientation = orientation
    int_marker.scale = 1

    int_marker.name = "goal" + goalID
    int_marker.description = "goal" + goalID

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


# if __name__ == "__main__":
rospy.loginfo("[ Goal Node" + goalID + " ] Launching...")
rospy.init_node("GOAL_Node_" + goalID, anonymous=False)

br = TransformBroadcaster()

server = InteractiveMarkerServer("goal_" + goalID)

menu_handler.insert("First Entry", callback=processFeedback)
menu_handler.insert("Second Entry", callback=processFeedback)
sub_menu_handle = menu_handler.insert("Submenu")
menu_handler.insert("First Entry", parent=sub_menu_handle, callback=processFeedback)
menu_handler.insert("Second Entry", parent=sub_menu_handle, callback=processFeedback)

i = 0
cnt = 0
position = position_matrix[i]
q = quaternion_from_euler(0, 0, 0)
makeQuadrocopterMarker(Point(*position), Quaternion(*q))
server.applyChanges()


def reward_callback(msg):
    global cnt
    cnt += 1
    cnt %= 2
    success_reward = msg.x
    print(cnt)
    if success_reward == 1 and cnt % 2 == 0:
        print("[ Goal Node " + goalID + " ] Sample Next Goal")
        global i
        i += 1
        if i >= 4:
            i = 0

        position = position_matrix[i]
        q = quaternion_from_euler(0, 0, 0)

        rospy.loginfo(
            "[ Goal Node"
            + goalID
            + " ] marker_%d = (%2.1f, %2.1f, %2.1f)"
            % (int(i), position[0], position[1], position[2])
        )
        makeQuadrocopterMarker(Point(*position), Quaternion(*q))
        server.applyChanges()


rospy.Subscriber("machine_0/reward", Point, reward_callback)

rospy.Timer(rospy.Duration(0.01), frameCallback)

# rospy.spin()
rate = rospy.Rate(1)
while not rospy.is_shutdown():
    rate.sleep()
