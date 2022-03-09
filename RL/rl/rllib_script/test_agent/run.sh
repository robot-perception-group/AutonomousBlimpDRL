#/bin/bash

script_full_path=$(dirname "$0")
LOGDIR=${HOME}/ray_results/Robustness/rosbag

for ((i=0; i<=1; i++)); do
  for ((j=0; j<=2; j++)); do
    for ((k=0; k<=2; k++)); do
      for ((l=0; l<=1; l++)); do

        expname=evaluation_${i}_${j}_${k}_${l}
        run_pid=$i
        wind=$j
        buoyancy=$k
        trajectory=$l


        bash $script_full_path/cleanup.sh

        echo "start test agent"
        screen -d -m -S TestAgent screen bash -c "\
          source ~/catkin_ws/devel/setup.bash;\
          python3 $script_full_path/test_agent.py "$run_pid" "$wind" "$buoyancy" "$trajectory" ;"


        echo "wait for environment to wake up"
        sleep 285;

        echo "start recording"
        screen -d -m -S RECORD bash -i $script_full_path/start_parallel_record.sh "$expname" "$LOGDIR"

        echo "record experiment for 15min (which is 30min with gazebo double speed)"
        sleep 900;

        echo "finish recording"
        screen -d -m -S RECORD bash -i $script_full_path/killbag.sh 
        sleep 10;

      done
    done
  done
done

bash $script_full_path/cleanup.sh
echo "finish all experiments"
