#!/usr/bin/env python

import math
import rospy
import socket
from geometry_msgs.msg import PointStamped,TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion,quaternion_from_euler
from uav_msgs.msg import uav_pose

throttle=0

imutopic = ('GCSIMU',Imu)
postopic = ('GCSPOS',PointStamped)
veltopic = ('GCSVEL',TwistStamped)
airspeedtopic = ('GCSAIRSPEED',Float32)
bridgetopic = ('fakehitlstate',uav_pose)

hitlpublisher = rospy.Publisher(bridgetopic[0],bridgetopic[1],queue_size=3)


class State():
    position=[0.0,0.0,0.0]
    velocity=[0.0,0.0,0.0]
    acceleration=[0.0,0.0,0.0]
    orientation=[0.0,0.0,0.0,0.0]
    rotation=[0.0,0.0,0.0]
    pressure=1024.0
    airspeed=0.0
    temperature=20.0

    def __str__(self):
       s =str(self.rotation[0])+","
       s+=str(self.rotation[1])+","
       s+=str(self.rotation[2])+","
       s+=str(self.acceleration[0])+","
       s+=str(self.acceleration[1])+","
       s+=str(self.acceleration[2])+","
       s+=str(self.orientation[0])+","
       s+=str(self.orientation[1])+","
       s+=str(self.orientation[2])+","
       s+=str(self.airspeed)+","
       s+=str(self.velocity[0])+","
       s+=str(self.velocity[1])+","
       s+=str(self.velocity[2])+","
       s+=str(self.position[0])+","
       s+=str(self.position[1])+","
       s+=str(self.position[2])+","
       s+=str(self.temperature)+","
       s+=str(self.pressure)
       return s


currentstate = State()
M_PI=3.1415926535897932


def messageSender():
    #rotation[x,y,z]
    #accel[x,y,z]
    #orientation[R,P,Y]
    #airspeed
    #velocity[N,E,D]
    #position[N,E,D]
    #temperature
    #pressure
    #string="0,0,0,0,0,9.81,0,0,0,0,0,0,0,0,0,0,0,-1\n"
    #UDPSocket.sendto(str.encode(str(currentstate)),server)
    #print(str(currentstate))
    #rospy.loginfo("msg sent to gcs")
    msg=uav_pose()
    msg.position.x = currentstate.position[0]
    msg.position.y = currentstate.position[1]
    msg.position.z = currentstate.position[2]
    msg.velocity.x = currentstate.velocity[0]
    msg.velocity.y = currentstate.velocity[1]
    msg.velocity.z = currentstate.velocity[2]
    msg.orientation.x= currentstate.orientation[0]
    msg.orientation.y= currentstate.orientation[1]
    msg.orientation.z= currentstate.orientation[2]
    msg.orientation.w= currentstate.orientation[3]
    msg.POI.x = currentstate.acceleration[0]
    msg.POI.y = currentstate.acceleration[1]
    msg.POI.z = currentstate.acceleration[2]
    msg.angVelocity.x = currentstate.rotation[0]
    msg.angVelocity.y = currentstate.rotation[1]
    msg.angVelocity.z = currentstate.rotation[2]
    msg.thrust = currentstate.airspeed
    hitlpublisher.publish(msg)
    return

def ImuMessageSubscriber(msg):
    currentstate.rotation[0]=msg.angular_velocity.x* 180.0/M_PI
    currentstate.rotation[1]=-msg.angular_velocity.y* 180.0/M_PI
    currentstate.rotation[2]=-msg.angular_velocity.z* 180.0/M_PI
    currentstate.acceleration[0]=msg.linear_acceleration.x
    currentstate.acceleration[1]=-msg.linear_acceleration.y
    currentstate.acceleration[2]=-msg.linear_acceleration.z
    q0 = [msg.orientation.x,-msg.orientation.y,-msg.orientation.z,msg.orientation.w]
    e0 = euler_from_quaternion(q0)
    currentstate.orientation = quaternion_from_euler(e0[0],e0[1],e0[2]+(math.pi/2.0))
    
    global throttle
    throttle+=1
    if throttle % 4 == 0: # YuTang: reduce Imu msg overflow on UDP for hitl
    	messageSender()
    	throttle=0
    

def VelMessageSubscriber(msg):
    currentstate.velocity[0]=msg.twist.linear.y
    currentstate.velocity[1]=msg.twist.linear.x
    currentstate.velocity[2]=-msg.twist.linear.z

def PosMessageSubscriber(msg):
    currentstate.position[0]=msg.point.y
    currentstate.position[1]=msg.point.x
    currentstate.position[2]=-msg.point.z

def AirspeedMessageSubscriber(msg):
    currentstate.airspeed=msg.data
    if (currentstate.airspeed<0):
        currentstate.airspeed=0.0; #no negative airspeeds allowed

rospy.init_node('ROS2ROSHITL')
rospy.loginfo("Startup")
imusubscriber = rospy.Subscriber(imutopic[0],imutopic[1],ImuMessageSubscriber)
velsubscriber = rospy.Subscriber(veltopic[0],veltopic[1],VelMessageSubscriber)
possubscriber = rospy.Subscriber(postopic[0],postopic[1],PosMessageSubscriber)
airspeedsubscriber = rospy.Subscriber(airspeedtopic[0],airspeedtopic[1],AirspeedMessageSubscriber)
rospy.loginfo("Subscriber started")

rospy.spin()





