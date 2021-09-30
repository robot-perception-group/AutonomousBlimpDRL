#!/bin/bash

WORLD="basic"
gui=true
robotID=0
GAZ_PORT=11351
ROS_PORT=11311
ROSIP=$(hostname -I | cut -d' ' -f1)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--id) robotID="$2"; shift ;;
        -d|--world) WORLD="$2"; shift ;;
        -g|--gui) gui="$2"; shift ;;
        -p|--gaz_port) GAZ_PORT="$2"; shift ;;
        -r|--ros_port) ROS_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";; 
    esac
    shift
done

echo "---- Start Gazebo World ----"
echo "WORLD:$WORLD gui:$gui"
screen -dmS WORLD_${robotID} screen bash -c "\
    export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
    export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
    export ROS_IP=$ROSIP;\
    export ROS_HOSTNAME=$ROSIP;\
	source ~/catkin_ws/devel/setup.bash;\
	roslaunch blimp_description world.launch world_name:=$WORLD gui:=$gui"
sleep 15
echo "---- Gazebo World Spawned----"


