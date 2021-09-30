#!/bin/bash

# read params
robotID=0
goalID=0
ROS_PORT=11311
GAZ_PORT=11351
ROSIP=$(hostname -I | cut -d' ' -f1)

Xs=(     100  50   0   0  0   0 100 100 100  0)
Ys=(     100  50   0   0  0   0 100 100 100  0)
MAX_Zs=( 200 120 200 400  50  50 200 200 200  5)
MIN_Zs=(   5  80   5 400  50  50   0   0   0  5)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--robotID) robotID="$2"; shift ;;
        -g|--goalID) goalID="$2"; shift ;; 
        -p|--gaz_port) GAZ_PORT="$2"; shift ;;
        -r|--ros_port) ROS_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";; 
    esac
    shift
done


# start business logics

echo "---- Spawning GOAL_${robotID} ----"
echo "robotID:$robotID goalID:$goalID"
screen -dmS GOAL_${robotID} screen bash -c "\
    export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
    export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
    export ROS_IP=$ROSIP;\
    export ROS_HOSTNAME=$ROSIP;\
    source ~/catkin_ws/devel/setup.bash;\
    roslaunch path_planner goal.launch robotID:=${robotID} X:=${Xs[$goalID]} Y:=${Ys[$goalID]} MAX_Z:=${MAX_Zs[$goalID]} MIN_Z:=${MIN_Zs[$goalID]};"
sleep 5
echo "---- Spawn GOAL_${robotID} Complete ----"

