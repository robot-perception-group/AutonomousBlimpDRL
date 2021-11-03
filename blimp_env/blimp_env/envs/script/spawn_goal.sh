#!/bin/bash

# read params
robotID=0
ROS_PORT=11311
GAZ_PORT=11351
ROSIP=$(hostname -I | cut -d' ' -f1)

Xs=100
Ys=100
MAX_Zs=200
MIN_Zs=10
Vs=8

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--robotID) robotID="$2"; shift ;;
        -p|--gaz_port) GAZ_PORT="$2"; shift ;;
        -r|--ros_port) ROS_PORT="$2"; shift ;;
        -px|--pos_x) Xs="$2"; shift ;;
        -py|--pos_x) Ys="$2"; shift ;;
        -pza|--pos_z_min) MAX_Zs="$2"; shift ;;
        -pzb|--pos_z_max) MIN_Zs="$2"; shift ;;
        -v|--vel) Vs="$2"; shift ;;
        
        *) echo "Unknown parameter passed: $1";; 
    esac
    shift
done


# start business logics

echo "---- Spawning GOAL_${robotID} ----"
echo "robotID:$robotID"
screen -dmS GOAL_${robotID} screen bash -c "\
    export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
    export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
    export ROS_IP=$ROSIP;\
    export ROS_HOSTNAME=$ROSIP;\
    source ~/catkin_ws/devel/setup.bash;\
    roslaunch path_planner goal.launch robotID:=${robotID} X:=${Xs} Y:=${Ys} MAX_Z:=${MAX_Zs} MIN_Z:=${MIN_Zs} V:=${Vs};"
sleep 5
echo "---- Spawn GOAL_${robotID} Complete ----"

