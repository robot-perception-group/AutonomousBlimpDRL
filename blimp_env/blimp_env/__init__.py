""" register gym environment """
from gym.envs.registration import register

register(
    id="planar_navigate-v0",
    entry_point="blimp_env.envs:PlanarNavigateEnv",
)
