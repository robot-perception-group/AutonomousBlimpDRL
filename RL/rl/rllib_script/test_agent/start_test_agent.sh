#/bin/bash

expname=$1
auto_start_simulation=$2

if [[ $auto_start_simulation -eq 1 ]] 
then
  bash ./cleanup.sh
fi


echo "start test agent"
screen -d -m -S TestAgent screen bash -c "\
    source ~/catkin_ws/devel/setup.bash;\
	python3 ~/catkin_ws/src/AutonomousBlimpDRL/RL/rl/rllib_script/test_agent/test_agent.py $auto_start_simulation;"


echo "wait for environment to wake up"
sleep 280;

echo "start recording"
screen -d -m -S RECORD bash -i ./start_parallel_record.sh "$expname"

echo "record experiment for 30min"
sleep 18000; 

echo "finish recording"
screen -d -m -S RECORD bash -i ./killbag.sh 
