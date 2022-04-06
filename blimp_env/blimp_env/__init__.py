""" register gym environment """
from gym.envs.registration import register

register(
    id="planar_navigate-v0",
    entry_point="blimp_env.envs:PlanarNavigateEnv",
)
register(
    id="residual_planar_navigate-v0",
    entry_point="blimp_env.envs:ResidualPlanarNavigateEnv",
)
register(
    id="yaw_control-v0",
    entry_point="blimp_env.envs:YawControlEnv",
)
register(
    id="aerobatic-v0",
    entry_point="blimp_env.envs:AerobaticEnv",
)
