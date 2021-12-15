#!/bin/bash

# read params
robotID=0


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--robotID) robotID="$2"; shift ;;
        *) echo "Unknown parameter passed: $1";; 
    esac
    shift
done



# start business logics
echo "---- Spawning PATH_${robotID} ----"
echo "robotID:$robotID"
screen -dmS GOAL_${robotID} screen bash -c "\
    source ~/catkin_ws/devel/setup.bash;\
    roslaunch path_planner path.launch robotID:=${robotID};"
sleep 2

echo "---- Spawn PATH_${robotID} Complete ----"
