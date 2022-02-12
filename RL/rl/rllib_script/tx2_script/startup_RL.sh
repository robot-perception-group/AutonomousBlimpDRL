#/bin/bash

LOGDIR=/mnt/raid/
FLIGHTNAME=$1
MACHINE=$2
CCOUNT=$3
SDATE=$( date "+%Y-%m-%d_%T" )
HOSTNAME=$( hostname )
if [ -z $CCOUNT ]; then
	echo "usage $0 [flightname] [copter ID] [copter count]"
	exit
fi

export FLIGHTNAME="$FLIGHTNAME"
export MACHINE=$MACHINE
export CCOUNT=$CCOUNT
export LOGDIR=$LOGDIR
echo "flightname is $FLIGHTNAME"

#create logdir
mkdir -p $LOGDIR

#delete symlink
filename="${LOGDIR}/${FLIGHTNAME}_${SDATE}_${MACHINE}_${HOSTNAME}"
mkdir $filename 
rm ${LOGDIR}/current
ln -s $filename ${LOGDIR}/current

# set working directory
cd ${HOME}/src/catkin_ws

# setup system write cache control
#sudo /usr/local/sbin/speedup.sh

# compile catkin modules
source devel/setup.bash
catkin_make

echo "built latest source, starting roscore ..."

# start a screen for command execution
#roscore >/dev/null 2>&1 &
screen -d -m -S ROSCORE bash -i -c ./roscore.sh
sleep 10;
echo "roscore started, starting LibrePilot Telemetry Link..."
screen -d -m -S TELEMETRY bash -i -c ./telemetry.sh
screen -d -m -S ROSLINK bash -i -c ./roslink.sh
sleep 10;
echo "link established, starting ROS logging.."
screen -d -m -S ROSBAGSELF bash -i -c ./log_RL.sh
#screen -d -m -S ROSBAGALL bash -i -c ./logall.sh
screen -d -m -S BANDWIDTHLOG bash -i -c ./bandwidth.sh

echo "starting RL node"
screen -d -m -S RLNODE screen bash -c "\
	source ~/venv/blimpRL/bin/activate;\
	python3 ~/src/AutonomousBlimpDRL/RL/rl/rllib_script/tx2_script/test_tx2.py;"


echo "everything is running. waiting for shutdown"
touch ${LOGDIR}/current/started

while [ ! -e ${LOGDIR}/current/abort ]; do
	sleep 1;
done
echo "shutdown initiated - starting cleanup, stuff should only fail now..." >>${LOGDIR}/current/faillog
./cleanup.sh
