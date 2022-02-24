from ray.rllib.agents import ppo


def find_nearest_power_of_two(a):
    """return closest power of two number smaller than a
    i.e. if a = 1025, return 1024

    """
    i = 0
    while a > 1:
        a = a / 2
        i += 1
    return 2 ** (i - 1)
