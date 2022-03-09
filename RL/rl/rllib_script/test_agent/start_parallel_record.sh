#/bin/bash

FLIGHTNAME=$1
LOGDIR=$2
SDATE=$( date "+%Y-%m-%d_%T" )
HOSTNAME=$( hostname )
CCOUNT=1

ROS_PORT=11311
GAZ_PORT=11351
n_machine=7

if [ -z $CCOUNT ]; then
	echo "usage $0 [flightname]"
	exit
fi

export FLIGHTNAME="$FLIGHTNAME"
export LOGDIR=$LOGDIR
echo "flightname is $FLIGHTNAME"

#create logdir
mkdir -p $LOGDIR

#delete symlink
filename="${LOGDIR}/${FLIGHTNAME}_${SDATE}_${HOSTNAME}"
mkdir $filename 


# set working directory
LOGDIR=${filename}




for ((i=0; i<=$(($n_machine-1)); i++)); do
	echo "---start rosbag script_$i---"
	screen -d -m -S ROSBAG_$i bash -i ./start_bag.sh "$i" "$(($ROS_PORT+$i))" "$(($GAZ_PORT+$i))"
	echo "---started---"
done


