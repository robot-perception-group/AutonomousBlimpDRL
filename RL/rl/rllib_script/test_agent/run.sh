#/bin/bash

script_full_path=$(dirname "$0")

for ((i=1; i<=4; i++)); do
  expname=agent_adaptive_${i}
  run_pid=$i

  bash $script_full_path/cleanup.sh

  echo "start test agent"
  screen -d -m -S TestAgent screen bash -c "\
    source ~/catkin_ws/devel/setup.bash;\
    python3 $script_full_path/test_agent.py "$run_pid" ;"


  echo "wait for environment to wake up"
  sleep 280;

  #echo "start recording"
  #screen -d -m -S RECORD bash -i $script_full_path/start_parallel_record.sh "$expname"

  #echo "record experiment for 15min"
  echo "wait for 1.5hr"
  sleep 5400; 

  #echo "finish recording"
  #screen -d -m -S RECORD bash -i $script_full_path/killbag.sh 
  #sleep 10;
done

bash $script_full_path/cleanup.sh
echo "finish all experiments"
