#!/bin/sh

ROSIP=$(hostname -I | cut -d' ' -f1)
MACHINE=$1
ROS_PORT=$2
GAZ_PORT=$3

export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
export ROS_IP=$ROSIP;\
export ROS_HOSTNAME=$ROSIP;\

echo "rosbag start recording machine_$MACHINE"
echo "log: ${LOGDIR}"

rosbag record -O ${LOGDIR}/rosbag_${MACHINE} /machine_$MACHINE/TransmitterInfo /machine_$MACHINE/pose /machine_$MACHINE/pose/raw /machine_$MACHINE/command /machine_$MACHINE/actuatorcommand /machine_$MACHINE/actuators /machine_$MACHINE/Imu /machine_$MACHINE/gyrobias  /machine_$MACHINE/target_tracker/pose /machine_$MACHINE/target_tracker/offset /machine_$MACHINE/diff_gps/groundtruth/gpspose /goal_$MACHINE/rviz_path /goal_$MACHINE/rviz_pos_cmd /goal_$MACHINE/rviz_waypoint_list /machine_$MACHINE/rviz_act /machine_$MACHINE/rviz_ang /machine_$MACHINE/rviz_ang_diff /machine_$MACHINE/rviz_ang_vel /machine_$MACHINE/rviz_base_act /machine_$MACHINE/rviz_pose_cmd /machine_$MACHINE/rviz_reward /machine_$MACHINE/rviz_state /machine_$MACHINE/rviz_vel /machine_$MACHINE/rviz_vel_diff __name:=my_bag_$MACHINE


