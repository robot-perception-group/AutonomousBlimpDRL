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


def close_simulation():
    """kill all simulators"""
    call_reply = subprocess.check_call(str(path) + "/cleanup.sh")
    return call_reply


def kill_blimp_screen(robot_id):
    """kill blimp screen session by specifying screen name and robot_id

    Args:
        robot_id ([str]): [number of the robot]

    Returns:
        [str]: [status of the script]
    """
    call_reply = subprocess.check_call(
        "screen -S %s -X quit" % ("BLIMP_" + robot_id),
        shell=True,
    )
    call_reply = subprocess.check_call(
        "screen -S %s -X quit" % ("FW_" + robot_id),
        shell=True,
    )
    return call_reply


def kill_target_screen(robot_id):
    return subprocess.check_call(
        "screen -S %s -X quit" % ("GOAL_" + robot_id),
        shell=True,
    )


def kill_all_screen(robot_id):
    """kill all screen session by specifying screen name and robot_id

    Args:
        robot_id ([str]): [number of the robot]

    Returns:
        [str]: [status of the script]
    """
    kill_blimp_reply = subprocess.check_call(
        "screen -S %s -X quit" % ("BLIMP_" + robot_id),
        shell=True,
    )
    kill_fw_reply = subprocess.check_call(
        "screen -S %s -X quit" % ("FW_" + robot_id),
        shell=True,
    )
    kill_goal_reply = subprocess.check_call(
        "screen -S %s -X quit" % ("GOAL_" + robot_id),
        shell=True,
    )
    kill_world_reply = subprocess.check_call(
        "screen -S %s -X quit" % ("WORLD_" + robot_id),
        shell=True,
    )
    time.sleep(15)
    return {
        "kill_blimp_reply": kill_blimp_reply,
        "kill_fw_reply": kill_fw_reply,
        "kill_goal_reply": kill_goal_reply,
        "kill_world_reply": kill_world_reply,
    }


def remove_blimp(robot_id):
    """kill model"""
    call_reply = subprocess.check_call(
        "rosservice call /gazebo/delete_model \"model_name: '%s' \""
        % ("machine_" + robot_id),
        shell=True,
    )
    return call_reply


def respawn_target(
    robot_id=0, goal_id=0, ros_port=DEFAULT_ROSPORT, gaz_port=DEFAULT_GAZPORT, **kwargs
):
    try:
        kill_target_reply = kill_target_screen(robot_id=robot_id)
    except:
        kill_target_reply = 1

    spawn_goal_reply = spawn_goal(
        robot_id=robot_id, goal_id=goal_id, ros_port=ros_port, gaz_port=gaz_port
    )

    return {
        "kill_target_reply": kill_target_reply,
        "spawn_model": spawn_goal_reply,
    }


def respawn_model(
    robot_id,
    enable_meshes=False,
    enable_wind=False,
    wind_direction="1 0",
    wind_speed=1.5,
    ros_port=DEFAULT_ROSPORT,
    gaz_port=DEFAULT_GAZPORT,
    **kwargs,
):
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
    robot_id=0,
    gui=True,
    world="basic",
    task="navigate",
    ros_port=DEFAULT_ROSPORT,
    gaz_port=DEFAULT_GAZPORT,
    enable_meshes=False,
    **kwargs,
):
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
        robot_id=robot_id, task=task, ros_port=ros_port, gaz_port=gaz_port
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


def spawn_world(
    robot_id=0,
    world="basic",
    gui=False,
    gaz_port=DEFAULT_GAZPORT,
    ros_port=DEFAULT_ROSPORT,
):
    """spawn gazebo world"""
    call_reply = subprocess.check_call(
        str(path)
        + "/spawn_world.sh %s %s %s %s %s %s %s %s %s %s"
        % ("-i", robot_id, "-g", gui, "-d", world, "-p", gaz_port, "-r", ros_port),
        shell=True,
    )
    return call_reply


def spawn_blimp(
    robot_id=0,
    enable_wind=False,
    enable_meshes=False,
    wind_direction=(1, 0),
    wind_speed=1.5,
    ros_port=DEFAULT_ROSPORT,
    gaz_port=DEFAULT_GAZPORT,
):
    """spawn blimp software in-the-loop"""
    wind_arg = "-w" if enable_wind else ""
    mesh_arg = "-m" if enable_meshes else ""
    call_reply = subprocess.check_call(
        str(path)
        + "/spawn_blimp_sitl.sh %s %s %s %s %s %s %s %s %s %s %s %s %s %s"
        % (
            "-i",
            robot_id,
            mesh_arg,
            wind_arg,
            "-wx",
            wind_direction[0],
            "-wy",
            wind_direction[1],
            "-ws",
            wind_speed,
            "-r",
            ros_port,
            "-p",
            gaz_port,
        ),
        shell=True,
    )

    return call_reply


def spawn_goal(
    robot_id,
    goal_id,
    ros_port=DEFAULT_ROSPORT,
    gaz_port=DEFAULT_GAZPORT,
):
    """spawn goal type target"""
    call_reply = subprocess.check_call(
        str(path)
        + "/spawn_goal.sh %s %s %s %s %s %s %s %s"
        % ("-i", robot_id, "-g", goal_id, "-r", ros_port, "-p", gaz_port),
        shell=True,
    )
    return call_reply


def spawn_square(
    robot_id,
    ros_port=DEFAULT_ROSPORT,
    gaz_port=DEFAULT_GAZPORT,
):
    """spawn goal type target"""
    call_reply = subprocess.check_call(
        str(path)
        + "/spawn_square.sh %s %s %s %s %s %s"
        % ("-i", robot_id, "-r", ros_port, "-p", gaz_port),
        shell=True,
    )
    return call_reply


def spawn_path(robot_id):
    """spawn path type target"""
    call_reply = subprocess.check_call(
        str(path) + "/spawn_path.sh %s %s" % ("-i", robot_id),
        shell=True,
    )
    return call_reply


task_goal_dict = {
    "navigate_goal": 0,
    "hover_goal": 1,
    "vertical_hover_goal": 2,
    "vertical_upward": 3,
    "hover_fixed_goal": 4,
    "test": 9,
}


def spawn_target(robot_id, task, ros_port=DEFAULT_ROSPORT, gaz_port=DEFAULT_GAZPORT):
    """spawn target depend on the task"""
    if task == "navigate":
        call_reply = spawn_path(robot_id=robot_id)
    elif task == "square":
        call_reply = spawn_square(robot_id=robot_id)
    else:
        call_reply = spawn_goal(
            robot_id=robot_id,
            goal_id=task_goal_dict[task],
            ros_port=ros_port,
            gaz_port=gaz_port,
        )

    return call_reply


def ros_master(robot_id=0, ros_port=DEFAULT_ROSPORT, gaz_port=DEFAULT_GAZPORT):
    """spawn ros master at specified port number"""
    call_reply = subprocess.check_call(
        str(path)
        + "/spawn_rosmaster.sh %s %s %s %s %s %s"
        % (
            "-i",
            robot_id,
            "-p",
            gaz_port,
            "-r",
            ros_port,
        ),
        shell=True,
    )
    return call_reply


def spawn_simulation(
    n_env=1,
    gui=True,
    world="basic",
    task="navigate",
):
    """start one or multiple blimp in one gazebo environment"""
    world_reply = spawn_world(world=world, gui=gui)
    for robot_id in range(n_env):
        blimp_reply = spawn_blimp(
            robot_id=robot_id,
        )
        target_reply = spawn_target(robot_id=robot_id, task=task)

    return {
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
        "target_reply": target_reply,
    }


def spawn_parallel_simulation(
    n_envs=1,
    gui=True,
    world="basic",
    task="navigate",
):
    """start blimp simulator on same ros port but different gazebo port"""
    for robot_id in range(n_envs):
        blimp_reply = spawn_simulation_on_different_port(
            robot_id=robot_id,
            gui=gui,
            world=world,
            task=task,
            ros_port=DEFAULT_ROSPORT + robot_id,
            gaz_port=DEFAULT_GAZPORT + robot_id,
        )

    return {
        "blimp_reply": blimp_reply,
    }


def spawn_simulation_on_different_port(
    robot_id=0,
    gui=True,
    world="basic",
    task="navigate",
    ros_port=DEFAULT_ROSPORT,
    gaz_port=DEFAULT_GAZPORT,
    enable_meshes=False,
    **kwargs,
):
    """start blimp simulator on different ros or gazbo port"""
    ros_reply = ros_master(robot_id=robot_id, ros_port=ros_port, gaz_port=gaz_port)
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
        robot_id=robot_id, task=task, ros_port=ros_port, gaz_port=gaz_port
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
    robot_id=0,
    gui=False,
    world="basic",
    task="navigate_goal",
    ros_port=DEFAULT_ROSPORT,
    gaz_port=DEFAULT_GAZPORT,
    **kwargs,
):
    """spawn simulation on another pc"""
    goal_id = task_goal_dict[task]
    args = f"spawn_blimp_simulation_on_different_port.sh -i \
        {robot_id} -f {goal_id} -r {ros_port} -p {gaz_port} -d {world} -g {gui}"
    cmd = "bash ~/catkin_ws/src/airship_simulation/script/"
    exe = "ssh yliu2@frg07.ifr.uni-stuttgart.de -L 2222:129.69.124.167:22 "
    msg = exe + cmd + args
    os.system(msg)


def close_simulation_on_marvin():
    """clone simulation on another pc"""
    args = "cleanup.sh"
    cmd = "bash ~/catkin_ws/src/airship_simulation/script/"
    exe = "ssh yliu2@frg07.ifr.uni-stuttgart.de -L 2222:129.69.124.167:22 "
    msg = exe + cmd + args
    os.system(msg)


def spawn_simulation_for_testing(
    n_env=1,
    gui=False,
    world="basic",
    enable_wind=False,
):
    """only for testing purpose"""
    close_simulation()
    world_reply = spawn_world(world=world, gui=gui)
    for robot_id in range(n_env):
        blimp_reply = spawn_blimp(robot_id=robot_id, enable_wind=enable_wind)
        path_reply = spawn_path(robot_id=robot_id)
        goal_reply = spawn_goal(robot_id=robot_id, goal_id=1)

    return {
        "world_reply": world_reply,
        "blimp_reply": blimp_reply,
        "path_reply": path_reply,
        "goal_reply": goal_reply,
    }
