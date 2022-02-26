#/bin/bash

script_full_path=$(dirname "$0")

for ((i=0; i<=1; i++)); do
  for ((j=0; j<=2; j++)); do
    expname=agent_$i_$j
    trajecotry=$i
    windspeed=$j

    bash $script_full_path/cleanup.sh

    echo "start test agent"
    screen -d -m -S TestAgent screen bash -c "\
      source ~/catkin_ws/devel/setup.bash;\
      python3 $script_full_path/test_agent.py $trajecotry $windspeed;"


    echo "wait for environment to wake up"
    sleep 280;

    echo "start recording"
    screen -d -m -S RECORD bash -i $script_full_path/start_parallel_record.sh "$expname"

    echo "record experiment for 15min (30min in total due to 2x speed in gazebo)"
    sleep 900; 

    echo "finish recording"
    screen -d -m -S RECORD bash -i $script_full_path/killbag.sh 

  done
done

echo "finish all experiments"
