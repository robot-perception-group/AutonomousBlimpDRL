""" script """
import subprocess
import pathlib
from typing import Tuple
import os
import socket
import time

path = pathlib.Path(__file__).parent.resolve()

DEFAULT_ROSPORT = 11311
DEFAULT_GAZPORT = 11351

# ============ Spawn Script ============#


def spawn_world(
    robot_id: int = 0,
    world: str = "basic",
    gui: bool = False,
    gaz_port: int = DEFAULT_GAZPORT,
    ros_port: int = DEFAULT_ROSPORT,
) -> int:
    """spawn gazebo world"""
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

    call_reply = subprocess.check_call(
        str(path)
        + f"/spawn_blimp_sitl.sh -i {robot_id} {mesh_arg} {wind_arg}\
             -wx {wind_direction[0]} -wy {wind_direction[1]} -ws {wind_speed}\
             -r {ros_port} -p {gaz_port} -px {position[0]} -py {position[1]} -pz {position[2]}",
        shell=True,
    )

    return int(call_reply)


def spawn_goal(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    position_range: tuple = (100, 100, 10, 200),
    velocity_range: float = 8,
) -> int:
    """spawn goal type target"""
    range_x, range_y, min_z, max_z = position_range
    call_reply = subprocess.check_call(
        str(path)
        + f"/spawn_goal.sh -i {robot_id} -r {ros_port} -p {gaz_port}\
             -px {range_x} -py {range_y} -pza {min_z} -pzb {max_z} -v {velocity_range}",
        shell=True,
    )
    return int(call_reply)


def spawn_square(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
) -> int:
    """spawn goal type target"""
    call_reply = subprocess.check_call(
        str(path) + f"/spawn_square.sh -i {robot_id} -r {ros_port} -p {gaz_port}",
        shell=True,
    )
    return int(call_reply)


def spawn_path(robot_id: int = 0) -> int:
    """spawn path type target"""
    call_reply = subprocess.check_call(
        str(path) + f"/spawn_path.sh -i {robot_id}",
        shell=True,
    )
    return int(call_reply)


def spawn_target(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    target_type: str = "Goal",
    position_range: tuple = (100, 100, 10, 200),
    velocity_range: float = 8,
    **kwargs,
) -> int:
    """spawn target"""
    if target_type == "Path":
        call_reply = spawn_path(
            robot_id=robot_id,
            ros_port=ros_port,
            gaz_port=gaz_port,
        )
    elif target_type == "Square":
        call_reply = spawn_square(
            robot_id=robot_id,
            ros_port=ros_port,
            gaz_port=gaz_port,
        )
    elif target_type == "Goal":
        call_reply = spawn_goal(
            robot_id=robot_id,
            ros_port=ros_port,
            gaz_port=gaz_port,
            position_range=position_range,
            velocity_range=velocity_range,
        )
    else:
        raise ValueError("Unknown target type")

    return int(call_reply)


def spawn_ros_master(
    robot_id: int = 0, ros_port: int = DEFAULT_ROSPORT, gaz_port: int = DEFAULT_GAZPORT
) -> int:
    """spawn ros master at specified port number"""
    call_reply = subprocess.check_call(
        str(path) + f"/spawn_rosmaster.sh -i {robot_id} -p {gaz_port} -r {ros_port}",
        shell=True,
    )
    return int(call_reply)


# ============ Composite Spawn Script ============#


def spawn_simulation(
    n_env: int = 1,
    gui: bool = True,
    world: str = "basic",
    target_type: str = "Goal",
) -> dict:
    """start one or multiple blimp in one gazebo environment"""
    world_reply = spawn_world(world=world, gui=gui)
    for robot_id in range(n_env):
        blimp_reply = spawn_blimp(
            robot_id=robot_id,
        )
        target_reply = spawn_target(robot_id=robot_id, target_type=target_type)

    return {
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
        "target_reply": target_reply,
    }


def spawn_parallel_simulation(
    n_envs: int = 1,
    gui: bool = True,
    world: str = "basic",
) -> dict:
    """start blimp simulator on same ros port but different gazebo port"""
    for robot_id in range(n_envs):
        blimp_reply = spawn_simulation_on_different_port(
            robot_id=robot_id,
            gui=gui,
            world=world,
            ros_port=DEFAULT_ROSPORT + robot_id,
            gaz_port=DEFAULT_GAZPORT + robot_id,
        )

    return {
        "blimp_reply": blimp_reply,
    }


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
    target_type: str = "Goal",
    target_position_range: tuple = (100, 100, 10, 200),
    target_velocity_range: tuple = 8,
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
    target_reply = spawn_target(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        target_type=target_type,
        position_range=target_position_range,
        velocity_range=target_velocity_range,
    )
    proc_result = {
        "ros_reply": ros_reply,
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
        "target_reply": target_reply,
    }
    print("spawn process result:", proc_result)
    return proc_result


def spawn_simulation_on_marvin(
    robot_id: int = 0,
    gui: bool = False,
    world: str = "basic",
    target_type: str = "Goal",
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


def spawn_simulation_for_testing(
    n_env: int = 1,
    gui: bool = False,
    world: str = "basic",
    enable_wind: bool = False,
) -> dict:
    """only for testing purpose"""
    close_simulation()
    world_reply = spawn_world(world=world, gui=gui)
    for robot_id in range(n_env):
        blimp_reply = spawn_blimp(robot_id=robot_id, enable_wind=enable_wind)
        path_reply = spawn_path(robot_id=robot_id)
        goal_reply = spawn_goal(robot_id=robot_id)

    return {
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
        "path_reply": path_reply,
        "goal_reply": goal_reply,
    }


# ============ Kill Script ============#


def close_simulation() -> int:
    """kill all simulators"""
    return int(subprocess.check_call(str(path) + "/cleanup.sh"))


def kill_blimp_screen(robot_id: int) -> Tuple[int]:
    """kill blimp screen session by specifying screen name and robot_id

    Args:
        robot_id ([str]): [number of the robot]

    Returns:
        [Tuple[int]]: [status of the script]
    """
    try:
        kill_fw_reply = subprocess.check_call(
            f"screen -S FW_{robot_id} -X quit",
            shell=True,
        )
    except:
        print("fw screen not found, skip kill")
        kill_fw_reply = 1

    try:
        kill_blimp_reply = subprocess.check_call(
            f"screen -S BLIMP_{robot_id} -X quit",
            shell=True,
        )
    except:
        print("blimp screen not found, skip kill")
        kill_blimp_reply = 1

    return int(kill_blimp_reply), int(kill_fw_reply)


def kill_goal_screen(robot_id: int) -> int:
    """kill target screen session

    Args:
        robot_id ([str]): [robot_id]

    Returns:
        [int]: [status of the script]
    """
    try:
        reply = subprocess.check_call(
            f"screen -S GOAL_{robot_id} -X quit",
            shell=True,
        )
    except:
        print("goal screen not found, skip kill")
        reply = 1
    return int(reply)


def kill_world_screen(robot_id: int) -> int:
    """kill gazebo screen session

    Args:
        robot_id ([str]): [robot_id]

    Returns:
        [int]: [status of the script]
    """
    try:
        reply = subprocess.check_call(
            f"screen -S WORLD_{robot_id} -X quit",
            shell=True,
        )
    except:
        print("world screen not found, skip kill")
        reply = 1
    return int(reply)


def kill_master_screen(robot_id: int) -> int:
    """kill ros master screen session

    Args:
        robot_id ([str]): [robot_id]

    Returns:
        [int]: [status of the script]
    """
    try:
        reply = subprocess.check_call(
            f"screen -S ROSMASTER_{robot_id} -X quit",
            shell=True,
        )
    except:
        print("master screen not found, skip kill")
        reply = 1
    return int(reply)


def kill_all_screen(robot_id: int) -> dict:
    """kill all screen session by specifying screen name and robot_id

    Args:
        robot_id ([str]): [number of the robot]

    Returns:
        [str]: [status of the script]
    """
    kill_goal_reply = kill_goal_screen(robot_id)
    kill_blimp_reply, kill_fw_reply = kill_blimp_screen(robot_id)
    kill_world_reply = kill_world_screen(robot_id)
    kill_master_reply = kill_master_screen(robot_id)
    time.sleep(15)
    return {
        "kill_goal_reply": kill_goal_reply,
        "kill_blimp_reply": kill_blimp_reply,
        "kill_fw_reply": kill_fw_reply,
        "kill_world_reply": kill_world_reply,
        "kill_master_reply": kill_master_reply,
    }


def remove_blimp(robot_id: int) -> int:
    """remove blimp model from gazebo world"""
    return int(
        subprocess.check_call(
            f"rosservice call /gazebo/delete_model \"model_name: 'machine_{robot_id}' \"",
            shell=True,
        )
    )


# ============ Respawn Script ============#


def respawn_goal(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    position_range: tuple = (100, 100, 10, 200),
    velocity_range: float = 8,
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """respawn target

    Args:
        robot_id (int, optional): [description]. Defaults to 0.
        ros_port ([type], optional): [description]. Defaults to DEFAULT_ROSPORT.
        gaz_port ([type], optional): [description]. Defaults to DEFAULT_GAZPORT.

    Returns:
        [type]: [success or not]
    """
    kill_goal_reply = kill_goal_screen(robot_id=robot_id)

    spawn_goal_reply = spawn_goal(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        position_range=position_range,
        velocity_range=velocity_range,
    )

    return {
        "kill_goal_reply": kill_goal_reply,
        "spawn_goal_reply": spawn_goal_reply,
    }


def respawn_model(
    robot_id: int = 0,
    enable_meshes: bool = False,
    enable_wind: bool = False,
    wind_direction: tuple = (1, 0),
    wind_speed: float = 1.5,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """respawn model
    first kill the screen session and then remove model from gazebo
    lastly spawn model again
    """
    kill_model_reply = kill_blimp_screen(robot_id=robot_id)
    remove_model_reply = remove_blimp(robot_id=robot_id)
    spawn_model_reply = spawn_blimp(
        robot_id=robot_id,
        enable_meshes=enable_meshes,
        enable_wind=enable_wind,
        wind_direction=wind_direction,
        wind_speed=wind_speed,
        ros_port=ros_port,
        gaz_port=gaz_port,
    )
    return {
        "kill_model": kill_model_reply,
        "remove_model": remove_model_reply,
        "spawn_model": spawn_model_reply,
    }


def resume_simulation(
    robot_id: int = 0,
    gui: bool = True,
    world: str = "basic",
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    enable_meshes: bool = False,
    target_type: str = "Goal",
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
    kill_reply = kill_all_screen(robot_id=robot_id)
    world_reply = spawn_world(
        robot_id=robot_id, world=world, gui=gui, ros_port=ros_port, gaz_port=gaz_port
    )
    blimp_reply = spawn_blimp(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        enable_meshes=enable_meshes,
    )
    target_reply = spawn_target(
        robot_id=robot_id, target_type=target_type, ros_port=ros_port, gaz_port=gaz_port
    )
    proc_result = {
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
        "target_reply": target_reply,
    }
    print("spawn process result:", proc_result)

    return {
        "kill_model": kill_reply,
        "proc_result": proc_result,
    }
