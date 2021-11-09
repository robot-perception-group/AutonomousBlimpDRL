from stable_baselines3.common.vec_env import SubprocVecEnv

DEFAULT_ROS_PORT = 11311
DEFAULT_GAZ_PORT = 11351


def create_env(
    robot_id, env, env_config, ros_port=DEFAULT_ROS_PORT, gaz_port=DEFAULT_GAZ_PORT
):
    simulation = {
        "robot_id": str(robot_id),
        "gui": False,
    }
    config = {
        "robot_id": str(robot_id),
        "simulation": simulation,
    }

    simulation.update(env_config["simulation"])
    config.update(env_config)
    config["simulation"].update(simulation)
    return env(config)


def parallel_envs_func_handle(
    robotid, env, env_config, ros_port=DEFAULT_ROS_PORT, gaz_port=DEFAULT_GAZ_PORT
):
    return lambda: create_env(
        robotid, env, env_config, ros_port=ros_port, gaz_port=gaz_port
    )


def subprocvecenv_handle(
    n_envs, env, env_config, ros_port=DEFAULT_ROS_PORT, gaz_port=DEFAULT_GAZ_PORT
):
    envs = []
    for i in range(n_envs):
        envs.append(
            parallel_envs_func_handle(
                i, env, env_config, ros_port=ros_port, gaz_port=gaz_port
            )
        )
    return SubprocVecEnv(envs, start_method="forkserver")
