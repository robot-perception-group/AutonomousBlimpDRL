#!/bin/bash

n_machine=7
screen_name=ROSBAG
ROSIP=$(hostname -I | cut -d' ' -f1)

for ((i=0; i<=$(($n_machine)); i++)); do
	echo "---kill rosbag_$i---"
	ROS_PORT=$((11311+$i))
	GAZ_PORT=$((11351+$i))
	export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
	export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
	export ROS_IP=$ROSIP;\
	export ROS_HOSTNAME=$ROSIP;\

	rosnode kill /my_bag_$i
done


echo "kill $screen_name"
for session in $(screen -ls | grep $screen_name); do 
	echo "kill screen session $session"
	screen -S ${session} -X quit; 
done
