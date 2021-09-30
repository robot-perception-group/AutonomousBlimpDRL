from setuptools import find_packages, setup

with open("README.rst", "r") as longdesc:
    long_description = longdesc.read()

setup(
    name="rl",
    description="rl agents",
    long_description=long_description,
    author="Yu Tang Liu",
    version="0.0.1",
    packages=find_packages(where="rl/"),
    package_dir={"": "rl"},
    install_requires=["gym", "stable_baselines3", "sb3-contrib", "tensorboard"],
)
