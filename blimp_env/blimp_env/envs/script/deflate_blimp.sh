#!/bin/bash

pi="3.14159"
namespace=$1
boost=$2
freeflop=$3
collapse=$4
helium=$5

ROS_PORT=$6
GAZ_PORT=$7
ROSIP=$(hostname -I | cut -d' ' -f1)


if [ -z "$namespace" -o "$namespace" = "-h" -o "$namespace" = "--help" ]; then
    echo "Usage: $0 <namespace> [deflation factor] [freeflop_angle] [collapse_factor] [buoancy]"
    echo -e "\tDeflation Factor:"
    echo -e "\t\t1.0  = Fully Inflated"
    echo -e "\t\t0.0  = Rigid"
    echo -e "\t\t5.0  = Floppy"
    echo -e "\tFreeflop Angle:"
    echo -e "\t\t0    = Properly rigged (default)"
    echo -e "\t\t5    = Ropes loose"
    echo -e "\t\t90   = No ropes"
    echo -e "\tCollapse Factor:"
    echo -e "\t\t0.0  = No collapse (default)"
    echo -e "\t\t0.05 = Saggy"
    echo -e "\t\t100  = Falling Apart"
    echo -e "\tBuoancy:"
    echo -e "\t\t1.0  = Full Helium volume"
    echo -e "\t\t0.0  = No Helium left"
    echo
    echo "Examples:"
    echo -e "\tIdeal Perfect Blimp:\t$0 blimp 0 0 0 1"
    echo -e "\tIntact Blimp:\t$0 blimp 1 0 0 1"
    echo -e "\tFloppy Blimp:\t$0 blimp 2 2 0.01 0.95"
    echo -e "\tBlimp after 2 days:\t$0 blimp 8 10 0.08 0.8"
    echo -e "\tDisassembly test:\t$0 blimp 100 100 100 0"
    exit
fi

if [ -z "$boost" ]; then
	boost=1
fi

math() {
    echo "scale=3; $@" |bc |sed -e 's/^\./0./'
}

if [ -z "$helium" ]; then
    helium=1
fi
if [ -z "$freeflop" ]; then
    freeflop=0
else
    freeflop=$( math "(${freeflop}*${pi}/180)" )
fi

if [ -z "$collapse" ]; then
    collapse=0
fi

if [ -z "$ROS_PORT" ]; then
    ROS_PORT=11311
fi

if [ -z "$GAZ_PORT" ]; then
    GAZ_PORT=11351
fi

export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT
export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT
export ROS_IP=$ROSIP
export ROS_HOSTNAME=$ROSIP

gondola_joints=( gondola_body_joint_link )
tail_joints=( top_rud_base_joint_link bot_rud_base_joint_link left_elv_base_joint_link right_elv_base_joint_link )

flex_joint_flexibility_factor="1.0"
flex_joint_flexibility_damping="0.5"
flex_joint_flexibility_erp="0.2"
base_erp="0.2"

gondola_factor="2.0"
tail_factor="1.0"
roll_factor="1.0"
pitch_factor="5.0"
yaw_factor="10.0"

default_helium_mass="1.723"
no_helium_mass="10" #approximation

calculated_helium_mass=$( math "${no_helium_mass} + ${helium}*( ${default_helium_mass} - ${no_helium_mass} )" )

sethelium() {
    mass=$1
    { rostopic pub -1 ${namespace}/heliummasstopic rotors_comm/WindSpeed "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
velocity:
  x: $mass
  y: 0.0
  z: 0.0" & }  >/dev/null
}

setjoint() {
    name=$1
    stopcfm=$2
    cfm=$3
    angle=$4
    { rosservice call /gazebo/set_joint_properties "joint_name: '${namespace}/${name}'
ode_joint_config:
  damping: [${flex_joint_flexibility_damping}]
  hiStop: [${angle}]
  loStop: [-${angle}]
  erp: [${base_erp}]
  cfm: [${cfm}]
  stop_erp: [${flex_joint_flexibility_erp}]
  stop_cfm: [${stopcfm}]
  fudge_factor: [-1]
  fmax: [-1]
  vel: [-1]" & } >/dev/null
}

setgroup() {
    name=$1
    factor=$2
    rollvalue=$( math "( $boost * $flex_joint_flexibility_factor ) / ($factor * $roll_factor)" )
    pitchvalue=$( math "( $boost * $flex_joint_flexibility_factor ) / ($factor * $pitch_factor)" )
    yawvalue=$( math "( $boost * $flex_joint_flexibility_factor ) / ($factor * $yaw_factor)" )
    setjoint ${joint}_joint0 $rollvalue 0 $freeflop
    setjoint ${joint}_joint1 $pitchvalue 0 $( math "${freeflop} / 2" )
    setjoint ${joint}_joint $yawvalue $( math "${collapse}/${factor}" ) $( math "${freeflop} / 4" )
}


for joint in ${gondola_joints[*]}; do
    setgroup ${joint} $gondola_factor
done

for joint in ${tail_joints[*]}; do
    setgroup ${joint} $tail_factor
done

sethelium ${calculated_helium_mass}
