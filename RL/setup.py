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
    install_requires=[
        "gym==0.18.0",
        "stable_baselines3==1.1.0",
        "sb3-contrib==1.1.0",
        "tensorboard==2.4.1",
    ],
)
