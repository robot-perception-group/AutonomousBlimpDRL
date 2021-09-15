killall screen
NUM_ENVS=$1
for env in $(seq 0 $(($NUM_ENVS-1))); do
	env_id=$(($env+1))
	screen -S env_${env_id} -X screen bash -i -c ./script/cleanup.sh
done
screen -S DRL_Training -X screen bash -i -c ./script/cleanup.sh
./script/cleanup.sh
killall -9 roscore
killall -9 rosmaster
