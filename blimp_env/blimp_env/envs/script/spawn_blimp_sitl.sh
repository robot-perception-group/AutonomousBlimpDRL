#!/bin/bash

# read params
robotID=0

enable_wind=false
enable_meshes=false
wind_direction_x=1 
wind_direction_y=0 
wind_speed_mean=4

ROS_PORT=11311
GAZ_PORT=11351
ROSIP=$(hostname -I | cut -d' ' -f1)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--robotID) robotID="$2"; shift ;;
        -w|--enable_wind) enable_wind=true ;;
        -m|--enable_meshes) enable_meshes=true ;;
        -wx|--wind_direction_x) wind_direction_x="$2" ; shift ;;
        -wy|--wind_direction_y) wind_direction_y="$2" ; shift ;;
        -ws|--wind_speed) wind_speed="$2"; shift ;;
        -p|--gaz_port) GAZ_PORT="$2"; shift ;;
        -r|--ros_port) ROS_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";; 
    esac
    shift
done

Xs=(   0   40   40  -40  -40    0    0   40  -40    0   40   40  -40  -40    0    0   40  -40    0)
Ys=(   0   40  -40  -40   40   40  -40    0    0    0   40  -40  -40   40   40  -40    0    0    0)
# Zs=( 100  110  110  110  110  120  120  120  120  130  140  140  140  140  150  150  150  150  160)
Zs=( 30  110  110  110  110  120  120  120  120  130  140  140  140  140  150  150  150  150  160)


# start business logics

echo "---- Spawn Blimp_${robotID} ----"
echo "robotID:$robotID enable_wind:$enable_wind"
echo "X:=${Xs[$robotID]} Y:=${Ys[$robotID]} Z:=${Zs[$robotID]}"

echo "Start FW_${robotID}"
screen -dmS FW_${robotID} screen sh -c "\
    export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
    export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
    export ROS_IP=$ROSIP;\
    export ROS_HOSTNAME=$ROSIP;\
    cd ~/blimp_ws/src/airship_simulation/LibrePilot;\
	./build/firmware/fw_simposix/fw_simposix.elf ${robotID}; sleep 1"
sleep 2

echo "Spawning Blimp_${robotID}"
screen -dmS BLIMP_${robotID} screen bash -ic "\
    export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
    export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
    export ROS_IP=$ROSIP;\
    export ROS_HOSTNAME=$ROSIP;\
	source ~/blimp_ws/devel/setup.bash;\
	roslaunch blimp_description blimp_ros.launch robotID:=${robotID} X:=${Xs[$robotID]} Y:=${Ys[$robotID]} Z:=${Zs[$robotID]}\
    enable_meshes:=${enable_meshes} enable_wind:=${enable_wind} wind_direction_x:=${wind_direction_x} wind_direction_y:=${wind_direction_y} wind_speed_mean:=${wind_speed};"
sleep 15


echo "---- Spawn Blimp_${robotID} Complete ----"
