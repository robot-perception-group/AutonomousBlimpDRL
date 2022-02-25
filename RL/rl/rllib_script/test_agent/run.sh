#/bin/bash

expname=$1
auto_start_simulation=$2

script_full_path=$(dirname "$0")


if [[ $auto_start_simulation -eq 1 ]] 
then
  bash $script_full_path/cleanup.sh
fi


echo "start test agent"
screen -d -m -S TestAgent screen bash -c "\
  source ~/catkin_ws/devel/setup.bash;\
  python3 $script_full_path/test_agent.py $auto_start_simulation;"


echo "wait for environment to wake up"
sleep 280;

echo "start recording"
screen -d -m -S RECORD bash -i $script_full_path/start_parallel_record.sh "$expname"

echo "record experiment for 15min (30min in total due to 2x speed in gazebo)"
sleep 9000; 

echo "finish recording"
screen -d -m -S RECORD bash -i $script_full_path/killbag.sh 
