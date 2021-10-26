""" action module of the environment """
#!/usr/bin/env python
from re import S
from typing import TYPE_CHECKING, Optional, Tuple, Union
from blimp_env.envs.common import utils

import numpy as np
import rospy
from blimp_env.envs.common.data_processor import DataProcessor, SimpleDataProcessor
from blimp_env.envs.common.utils import RangeObj
from blimp_env.envs.script import respawn_model, resume_simulation
from gym import spaces
from librepilot.msg import LibrepilotActuators
from rospy.client import ERROR
from uav_msgs.msg import uav_pose
from geometry_msgs.msg import Point
import time

if TYPE_CHECKING:
    from blimp_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType:
    """abstract action type"""

    def __init__(
        self,
        env: "AbstractEnv",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.env = env

    def space(self) -> spaces.Space:
        """action space"""
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """perform action

        Args:
            action (Action): action
        """
        raise NotImplementedError

    def get_cur_act(self) -> np.ndarray:  # pylint: disable=no-self-use
        """return current action with a standardized format"""
        raise NotImplementedError

    def action_rew(self, scale: float) -> float:
        """calculate action reward from current action state

        Args:
            scale (float): [scale]

        Returns:
            float: [action reward indicates quality of current action state]
        """
        raise NotImplementedError


class ROSActionType(ActionType):
    """ROS abstract action type"""

    def __init__(
        self,
        env: "AbstractEnv",
        robot_id: str = "0",
        flightmode: int = 3,
        name_space: str = "machine_0",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env=env, **kwargs)

        self.robot_id = robot_id
        self.name_space = name_space
        self.flightmode = flightmode

        self.action_publisher = rospy.Publisher(
            self.name_space + "/actuatorcommand", LibrepilotActuators, queue_size=1
        )
        self.flightmode_publisher = rospy.Publisher(
            self.name_space + "/command", uav_pose, queue_size=1
        )

        self.act_dim = 8
        self.cur_act = self.init_act = np.zeros(self.act_dim)

    def check_publishers_connection(self) -> None:
        """check actuator publisher connections"""
        waiting_time = 0
        respawn_time = 0

        while self.action_publisher.get_num_connections() == 0:
            rospy.loginfo("No subscriber to action_publisher yet, wait...")
            waiting_time += 1
            time.sleep(1)

            if waiting_time >= 5:
                rospy.loginfo("respawn model...")
                respawn_model(**self.env.config["simulation"])
                respawn_time += 1
                waiting_time = 0

            if respawn_time > 3:
                rospy.loginfo("Simulation Crashed...resume simulation")
                reply = resume_simulation(**self.env.config["simulation"])
                respawn_time = 0
                rospy.loginfo("Simulation Resumed:", reply)

        rospy.logdebug("action_publisher connected")
        return self.action_publisher.get_num_connections() > 0

    def set_init_pose(self):
        """set initial actions"""
        self.check_publishers_connection()
        self.cur_act = self.init_act.copy()
        self.act(self.init_act)

    def act(self, action: Action):
        raise NotImplementedError

    def action_rew(self, scale: float):
        raise NotImplementedError

    def space(self):
        raise NotImplementedError


class ContinuousAction(ROSActionType):
    """continuous action space"""

    ACTION_RANGE = (1000, 2000)

    def __init__(
        self,
        env: "AbstractEnv",
        robot_id: str = "0",
        flightmode: int = 3,
        dbg_act: bool = False,
        name_space: str = "machine_0",
        act_noise_stdv: float = 0.05,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """action channel
        0: m2
        1: lfin
        2: rfin
        3: tfin
        4: bfin
        5: stick
        6: m1
        7: unused
        8: m0
        9: unused
        10: unused
        11: unused

        """

        super().__init__(
            env=env,
            robot_id=robot_id,
            name_space=name_space,
            flightmode=flightmode,
            **kwargs,
        )
        self.dbg_act = dbg_act

        self.act_dim = 8
        self.act_range = self.ACTION_RANGE
        self.act_noise_stdv = act_noise_stdv

        self.cur_act = self.init_act = np.zeros(self.act_dim)

    def space(self) -> spaces.Box:
        return spaces.Box(
            low=np.full((self.act_dim), -1),
            high=np.full((self.act_dim), 1),
            dtype=np.float32,
        )

    def process_action(self, action: np.array):
        """map agent action to actuator specification

        Args:
            action ([np.array]): agent action

        Returns:
            [np.array]: formated action with 12 channels
        """
        proc = action + np.random.normal(0, self.act_noise_stdv, action.shape)
        proc = utils.lmap(proc, [-1, 1], self.act_range)
        proc = np.clip(proc, -1, 1)
        proc = self.augment_action(proc)
        return proc

    @classmethod
    def augment_action(cls, action):
        """fill empty channels to fulfill requirement for the gcs

        Args:
            action ([np array]): actions with empty action channels

        Returns:
            [np array]: actions with all action channels filled

        """
        action = action.reshape(8, 1)
        aug_action = np.zeros(12).reshape(12, 1)

        aug_action[0:7] = action[0:7]
        aug_action[8] = action[7]

        aug_action[7] = 1500
        aug_action[9:12] = 1500

        return aug_action

    def act(self, action: np.ndarray) -> None:
        """publish action

        Args:
            action (np.ndarray): agent action
        """
        self.cur_act = action
        processed_action = self.process_action(action)

        act_msg = LibrepilotActuators()
        act_msg.header.stamp = rospy.Time.now()
        act_msg.data.data = processed_action

        mode = uav_pose()
        mode.flightmode = self.flightmode

        self.action_publisher.publish(act_msg)
        self.flightmode_publisher.publish(mode)

        if self.dbg_act:
            print("[ ContinuousAction ] act: mode:", self.flightmode)
            print("[ ContinuousAction ] act: action:", action)
            print("[ ContinuousAction ] act: processed_action:", processed_action)

    def action_rew(self, scale=0.5):
        """calculate action reward to penalize using motors

        Args:
            action ([np.ndarray]): agent action
            scale (float, optional): reward scale. Defaults to 0.5.

        Returns:
            [float]: action reward
        """
        motors = self.get_cur_act()[[0, 6, 7]]
        motors_rew = np.exp(-scale * np.linalg.norm(motors))
        return motors_rew

    def get_cur_act(self):
        """get current action"""
        cur_act = self.cur_act.copy()
        cur_act = self.augment_action(cur_act)[[0, 1, 2, 3, 4, 5, 6, 8]]
        return cur_act.reshape(
            8,
        )


class SimpleContinuousDifferentialAction(ContinuousAction):
    """simplified action space by binding action that has similar dynamic effect"""

    def __init__(self, *args, **kwargs):
        """action channel
        0: back motor + top fin + bot fin
        1: left fin + right fin
        2: servo
        3: left motor + right motor
        """
        super().__init__(*args, **kwargs)
        self.act_dim = 4

        self.diff_act_scale = np.array([0.1, 0.1, 0.1, 0.04])

        self.init_act = np.zeros(self.act_dim)
        self.cur_act = np.zeros(self.act_dim)

    def space(self):
        return spaces.Box(
            low=np.full((self.act_dim), self.action_range.scaled_min),
            high=np.full((self.act_dim), self.action_range.scaled_max),
            dtype=np.float32,
        )

    def act(self, action: np.ndarray) -> None:
        """publish action

        Args:
            action (np.ndarray): agent action [-1, 1]
        """
        processed_action = self.process_action(action)

        act_msg = LibrepilotActuators()
        act_msg.header.stamp = rospy.Time.now()
        act_msg.data.data = processed_action

        mode = uav_pose()
        mode.flightmode = self.flightmode

        self.action_publisher.publish(act_msg)
        self.flightmode_publisher.publish(mode)

        if self.dbg_act:
            print("[ ContinuousDiffAction ] act: mode:", self.flightmode)
            print("[ ContinuousDiffAction ] act: action:", action)
            print(
                "[ ContinuousDiffAction ] act: processed_action:",
                processed_action,
            )

    def process_action(self, action: np.array):
        """map agent action to actuator specification

        Args:
            action ([np.array]): agent action [-1,1]

        Returns:
            [np.array]: formated action with 12 channels
        """
        self.cur_act += self.diff_act_scale * action
        self.cur_act = np.clip(self.cur_act, -1, 1)
        # only allow foward thrust
        if self.cur_act[3] < 0:
            self.cur_act[3] = 0
        # only allow forward servo
        if self.cur_act[2] > 0:
            self.cur_act[2] = 0

        action = self.cur_act.copy()

        proc = action + np.random.normal(0, self.act_noise_stdv, action.shape)
        proc = utils.lmap(proc, [-1, 1], self.act_range)
        proc = np.clip(proc, -1, 1)
        proc = self.augment_action(proc)

        return proc

    def augment_action(self, action):
        """fill empty channels to fulfill requirement for the gcs

        Args:
            action ([np array]): actions with empty action channels

        Returns:
            [np array]: actions with all action channels filled

        """
        action = action.reshape(4, 1)
        aug_action = np.zeros(12).reshape(12, 1)

        aug_action[0] = action[0]
        aug_action[1] = action[1]
        aug_action[2] = action[1]
        aug_action[3] = action[0]
        aug_action[4] = action[0]
        aug_action[5] = action[2]
        aug_action[6] = action[3]
        aug_action[8] = action[3]

        aug_action[7] = 1500
        aug_action[9:12] = 1500

        return aug_action

    def action_rew(self, scale: float = 0.57735) -> float:
        """compute action reward

        Args:
            scale (float, optional): [scale,
            the greater this number harder to get reward]. Defaults to 0.57735.

        Returns:
            [float]: [reward of current action state]
        """
        motors = self.get_cur_act()[[0, 6, 7]]
        motors_rew = -np.linalg.norm(motors) * scale
        return motors_rew

    def get_cur_act(self):
        """get current action"""
        cur_act = self.cur_act.copy()
        cur_act = self.augment_action(cur_act)[[0, 1, 2, 3, 4, 5, 6, 8]]
        return cur_act.reshape(
            8,
        )


def action_factory(  # pylint: disable=too-many-return-statements
    env: "AbstractEnv",
    config: dict,
) -> ActionType:
    """control action type"""
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    elif config["type"] == "SimpleContinuousDifferentialAction":
        return SimpleContinuousDifferentialAction(env, **config)
    else:
        raise ValueError("Unknown action type")
