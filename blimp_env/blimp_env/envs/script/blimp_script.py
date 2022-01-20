""" script """
import errno
import os
import pathlib
import socket
import subprocess
import time
from typing import Tuple

import rospy
from blimp_env.envs.common.utils import timeout

path = pathlib.Path(__file__).parent.resolve()

DEFAULT_ROSPORT = 11311
DEFAULT_GAZPORT = 11351


# ============ utility function ============#


def change_buoynacy(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    deflation: float = 1.0,
    freeflop_angle: float = 0.0,
    collapse: float = 0.0,
    buoyancy: float = 1.0,
):
    """
    Examples:
        Ideal Perfect Blimp:    ./deflate_blimp.sh blimp 0 0 0 1
        Intact Blimp:   ./deflate_blimp.sh blimp 1 0 0 1
        Floppy Blimp:   ./deflate_blimp.sh blimp 2 2 0.01 0.95
        Blimp after 2 days:     ./deflate_blimp.sh blimp 8 10 0.08 0.8
        Disassembly test:       ./deflate_blimp.sh blimp 100 100 100 0
    """
    machine_name = "machine_" + str(robot_id)

    call_reply = subprocess.check_call(
        str(path)
        + f"/deflate_blimp.sh {machine_name} {deflation} {freeflop_angle} {collapse} {buoyancy} "
        + f"{ros_port} {gaz_port}",
        shell=True,
    )
    rospy.loginfo(
        f"Change buoyancy state to (deflation, freeflop_angle, collapse, buoyancy): {(deflation, freeflop_angle, collapse, buoyancy)}",
    )
    return call_reply


# ============ check screen ============#


def find_screen_session(name: str):
    try:
        screen_name = subprocess.check_output(
            f"ls /var/run/screen/S-* | grep {name}",
            shell=True,
        )
    except subprocess.CalledProcessError:
        screen_name = None
    return screen_name


def check_screen_sessions_exist(
    names: list = ["ROSMASTER_", "WORLD_", "FW_", "BLIMP_"]
):
    all_exist = True
    for name in names:
        all_exist *= find_screen_session(name) is not None
    return bool(all_exist)


# ============ Spawn Script ============#


def spawn_ros_master(
    robot_id: int = 0, ros_port: int = DEFAULT_ROSPORT, gaz_port: int = DEFAULT_GAZPORT
) -> int:
    """spawn ros master at specified port number"""

    names = ["ROSMASTER_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        call_reply = subprocess.check_call(
            str(path)
            + f"/spawn_rosmaster.sh -i {robot_id} -p {gaz_port} -r {ros_port}",
            shell=True,
        )
    return int(call_reply)


def spawn_world(
    robot_id: int = 0,
    world: str = "basic",
    gui: bool = False,
    gaz_port: int = DEFAULT_GAZPORT,
    ros_port: int = DEFAULT_ROSPORT,
) -> int:
    """spawn gazebo world"""
    names = ["WORLD_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        call_reply = subprocess.check_call(
            str(path)
            + f"/spawn_world.sh -i {robot_id} -g {gui} -d {world} -p {gaz_port} -r {ros_port}",
            shell=True,
        )
    return int(call_reply)


def spawn_blimp(
    robot_id: int = 0,
    enable_wind: bool = False,
    enable_meshes: bool = False,
    wind_direction: tuple = (1, 0),
    wind_speed: float = 1.5,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    position: tuple = (0, 0, 100),
) -> int:
    """spawn blimp software in-the-loop"""
    wind_arg = "-w" if enable_wind else ""
    mesh_arg = "-m" if enable_meshes else ""

    names = ["FW_" + str(robot_id), "BLIMP_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        kill_screens(robot_id=robot_id, screen_names=names, sleep_times=[3, 10])
        call_reply = subprocess.check_call(
            str(path)
            + f"/spawn_blimp_sitl.sh -i {robot_id} {mesh_arg} {wind_arg}\
                -wx {wind_direction[0]} -wy {wind_direction[1]} -ws {wind_speed}\
                -r {ros_port} -p {gaz_port} -px {position[0]} -py {position[1]} -pz {position[2]}",
            shell=True,
        )
        time.sleep(3)

    return int(call_reply)


def spawn_goal(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    target_position_range: tuple = (100, 100, 10, 200),
    target_velocity_range: float = 8,
    **kwargs,
) -> int:
    """spawn goal type target"""
    range_x, range_y, min_z, max_z = target_position_range

    names = ["GOAL_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        call_reply = subprocess.check_call(
            str(path)
            + f"/spawn_goal.sh -i {robot_id} -r {ros_port} -p {gaz_port}\
                -px {range_x} -py {range_y} -pza {min_z} -pzb {max_z} -v {target_velocity_range}",
            shell=True,
        )
    return int(call_reply)


def spawn_square(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
) -> int:
    """spawn goal type target"""
    names = ["GOAL_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        call_reply = subprocess.check_call(
            str(path) + f"/spawn_square.sh -i {robot_id} -r {ros_port} -p {gaz_port}",
            shell=True,
        )
    return int(call_reply)


def spawn_path(robot_id: int = 0) -> int:
    """spawn path type target"""

    names = ["GOAL_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        call_reply = subprocess.check_call(
            str(path) + f"/spawn_path.sh -i {robot_id}",
            shell=True,
        )
    return int(call_reply)


def spawn_target(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    target_type: str = "InteractiveGoal",
    **kwargs,
) -> int:
    """spawn target"""
    if target_type == "InteractiveGoal":
        spawn_fn = spawn_goal
    elif target_type == "RandomGoal":
        spawn_fn = None
    elif target_type == "Path":
        spawn_fn = spawn_path
    elif target_type == "Square":
        spawn_fn = spawn_square
    else:
        raise ValueError("Unknown target type")

    if spawn_fn is not None:
        spawn_target_reply = spawn_fn(
            robot_id=robot_id, ros_port=ros_port, gaz_port=gaz_port, **kwargs
        )
        return int(spawn_target_reply)
    else:
        return None


# ============ Composite Spawn Script ============#


def spawn_simulation_on_different_port(
    robot_id: int = 0,
    gui: bool = True,
    world: str = "basic",
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    enable_meshes: bool = False,
    enable_wind: bool = False,
    wind_direction: tuple = (1, 0),
    wind_speed: float = 1.5,
    position: tuple = (0, 0, 100),
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """start blimp simulator on different ros or gazbo port"""

    ros_reply = spawn_ros_master(
        robot_id=robot_id, ros_port=ros_port, gaz_port=gaz_port
    )
    world_reply = spawn_world(
        robot_id=robot_id, world=world, gui=gui, ros_port=ros_port, gaz_port=gaz_port
    )
    blimp_reply = spawn_blimp(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        enable_meshes=enable_meshes,
        enable_wind=enable_wind,
        wind_direction=wind_direction,
        wind_speed=wind_speed,
        position=position,
    )
    proc_result = {
        "ros_reply": ros_reply,
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
    }
    rospy.loginfo("spawn process result:", proc_result)
    return proc_result


# ============ Kill Script ============#


def close_simulation() -> int:
    """kill all simulators"""
    reply = int(subprocess.check_call(str(path) + "/cleanup.sh"))
    return reply


def kill_screen(screen_name, sleep_time=1):
    reply = 1
    while find_screen_session(screen_name) is not None:
        try:
            reply = subprocess.check_call(
                f'for session in $(screen -ls | grep {screen_name}); do screen -S "${{session}}" -X quit; done',
                shell=True,
            )
            time.sleep(sleep_time)
        except:
            print(f"screen {screen_name} not found, skip kill")
    return reply


def kill_screens(
    robot_id: int,
    screen_names: list = ["GOAL_", "FW_", "BLIMP_", "WORLD_", "ROSMASTER_"],
    sleep_times: list = [3, 3, 10, 10, 5],
) -> Tuple[int]:
    """kill screen session by specifying screen name and robot_id

    Args:
        robot_id ([str]): [number of the robot]
        screen_names ([list]): [screen name]
        sleep_time ([list]): [sleep time after kill]

    Returns:
        [Tuple[int]]: [status of the script]
    """
    reply = {}
    for screen_name, sleep_time in zip(screen_names, sleep_times):
        kill_reply = kill_screen(screen_name + str(robot_id), sleep_time)
        reply.update({"kill_" + screen_name + str(robot_id): kill_reply})
    return reply


def kill_rosnode(node_name: str) -> str:
    reply = subprocess.check_call(
        f"for node in $(rosnode list | grep {node_name}) | xargs rosnode kill",
        shell=True,
    )
    return reply


def remove_blimp(robot_id: int) -> int:
    """remove blimp model from gazebo world"""
    reply = int(
        subprocess.check_call(
            f"rosservice call /gazebo/delete_model \"model_name: 'machine_{robot_id}' \"",
            shell=True,
        )
    )
    time.sleep(5)
    return reply


# ============ Respawn Script ============#


def respawn_target(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    target_type: str = "InteractiveGoal",
    **kwargs,
) -> int:
    kill_reply = kill_screens(
        robot_id=robot_id, screen_names=["GOAL_"], sleep_times=[3]
    )
    spawn_target_reply = spawn_target(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        target_type=target_type,
        **kwargs,
    )
    return {"kill_reply": kill_reply, "spawn_target_reply": spawn_target_reply}


@timeout(50, os.strerror(errno.ETIMEDOUT))
def respawn_model(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    enable_meshes: bool = False,
    enable_wind: bool = False,
    wind_direction: tuple = (1, 0),
    wind_speed: float = 1.5,
    position: tuple = (0, 0, 100),
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """respawn model
    first kill the screen session and then remove model from gazebo
    lastly spawn model again
    """
    kill_model_reply = kill_screens(
        robot_id=robot_id, screen_names=["FW_", "BLIMP_"], sleep_times=[3, 10]
    )
    remove_model_reply = remove_blimp(robot_id=robot_id)

    blimp_reply = spawn_blimp(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        enable_meshes=enable_meshes,
        enable_wind=False,  # TODO: check if wind plugin still function after model removal
        wind_direction=wind_direction,
        wind_speed=wind_speed,
        position=position,
    )

    return {
        "kill_model": kill_model_reply,
        "remove_model": remove_model_reply,
        "spawn_model": blimp_reply,
    }


def resume_simulation(
    robot_id: int = 0,
    gui: bool = True,
    world: str = "basic",
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    enable_meshes: bool = False,
    target_type: str = "RandomGoal",
    enable_wind: bool = False,
    wind_direction: tuple = (1, 0),
    wind_speed: float = 1.5,
    position: tuple = (0, 0, 100),
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """resume simulation
    first kill all screens
    then spawn gazebo world and blimp SITL

    Args:
        robot_id (int, optional): [description]. Defaults to 0.
        gui (bool, optional): [description]. Defaults to True.
        world (str, optional): [description]. Defaults to "basic".
        ros_port ([type], optional): [description]. Defaults to DEFAULT_ROSPORT.
        gaz_port ([type], optional): [description]. Defaults to DEFAULT_GAZPORT.
        enable_meshes (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [success]
    """
    kill_reply = kill_screens(robot_id=robot_id)
    master_reply = spawn_ros_master(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
    )
    world_reply = spawn_world(
        robot_id=robot_id,
        world=world,
        gui=gui,
        ros_port=ros_port,
        gaz_port=gaz_port,
    )

    blimp_reply = spawn_blimp(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        enable_meshes=enable_meshes,
        enable_wind=enable_wind,
        wind_direction=wind_direction,
        wind_speed=wind_speed,
        position=position,
    )

    target_reply = spawn_target(
        robot_id=robot_id,
        target_type=target_type,
        ros_port=ros_port,
        gaz_port=gaz_port,
    )
    proc_result = {
        "master_reply": master_reply,
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
        "target_reply": target_reply,
    }
    rospy.loginfo("spawn process result:", proc_result)

    return {
        "kill_all_screen": kill_reply,
        "proc_result": proc_result,
    }


# ============ remote ============#


def spawn_simulation_on_marvin(
    robot_id: int = 0,
    gui: bool = False,
    world: str = "basic",
    target_type: str = "InteractiveGoal",
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    host_id: str = "yliu2@frg07.ifr.uni-stuttgart.de",
    host_ip: str = "2222:129.69.124.167:22",
    **kwargs,  # pylint: disable=unused-argument
):
    """spawn simulation on another pc"""
    exe = f"ssh {host_id} -L {host_ip}"
    cmd = "bash ~/catkin_ws/src/airship_simulation/script/"
    args = f"spawn_blimp_simulation_on_different_port.sh -i \
        {robot_id} -f {target_type} -r {ros_port} -p {gaz_port} -d {world} -g {gui}"
    msg = exe + cmd + args
    os.system(msg)


def close_simulation_on_marvin(
    host_id: str = "yliu2@frg07.ifr.uni-stuttgart.de",
    host_ip: str = "2222:129.69.124.167:22",
):
    """clone simulation on another pc"""
    args = "cleanup.sh"
    cmd = "bash ~/catkin_ws/src/airship_simulation/script/"
    exe = f"ssh {host_id} -L {host_ip} "
    msg = exe + cmd + args
    os.system(msg)
