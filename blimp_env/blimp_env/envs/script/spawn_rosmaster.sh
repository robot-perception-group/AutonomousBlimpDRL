#!/bin/bash

robotID=0
ROS_PORT=11311
GAZ_PORT=11351
ROSIP=$(hostname -I | cut -d' ' -f1)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--id) robotID="$2"; shift ;;
        -p|--gaz_port) GAZ_PORT="$2"; shift ;;
        -r|--ros_port) ROS_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";; 
    esac
    shift
done

echo "---- Start ROS MASTER ----"
echo "PORT:$ROS_PORT"
screen -dmS ROSMASTER_${robotID} screen bash -c "\
    export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
    export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
    export ROS_IP=$ROSIP;\
    export ROS_HOSTNAME=$ROSIP;\
	roscore -p $ROS_PORT"
sleep 3
echo "---- ROS MASTER Spawned----"


