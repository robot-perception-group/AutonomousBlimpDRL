""" Environment Data Processor """
from typing import Dict
import numpy as np


class DataProcessor:
    """data processor for the environment"""

    @classmethod
    def add_noise(cls, value: np.ndarray, noise_level: float = 0.05):
        """add noise to the input value

        Args:
            value ([np.array]): [value to add noise]
            noise_level (float, optional): [noise level]. Defaults to 0.05.

        Returns:
            [np.array]: noised value

        """
        noise = np.random.normal(0, noise_level, value.shape)
        value = value + noise
        return value

    @classmethod
    def scaler(cls, inputs, inputs_min_max, outputs_min_max):
        """scale input range to output range

        Args:
            inputs ([float]): can be a number or array
            inputs_min_max ([float]): minimum maximum inputs
            outputs_min_max ([float]):  minimum maximum outputs

        Returns:
            [float]: scaled inputs

        """
        inputs_min, inputs_max = inputs_min_max[0], inputs_min_max[1]
        outputs_min, outputs_max = outputs_min_max[0], outputs_min_max[1]

        input_mean = (inputs_max + inputs_min) / 2
        output_mean = (outputs_max + outputs_min) / 2

        scale = (inputs_max - inputs_min) / (outputs_max - outputs_min)
        results = (inputs - input_mean) / scale + output_mean
        return results

    @classmethod
    def clip(cls, value, val_min_max):
        """clip value to range

        Args:
            value ([np.array]): value need to be clipped
            val_min_max ([int]): min max value

        Returns:
            [np array]: clipped value

        """
        return np.clip(value, val_min_max[0], val_min_max[1])

    def scale_action(self, action, action_range):
        """scale action from (-1, 1) to (1000, 2000)

        Args:
            action ([np.array]): actions to be scaled
            action_range ([range_object]): range object class
                contains "min", "max", "scaled_min", "scaled_max" objects

        Returns:
            [np.array]: scaled action

        """
        return self.scaler(
            action,
            action_range.get_scale_range(),
            action_range.get_range(),
        )

    def scale_and_clip(self, value, value_range):
        """scale data from (-vmin, vmax) to (vsmin, vsmax) and clip by (vsmin, vsmax)

        Args:
            value ([int, np.array]): value int or array
            value_range ([range_object]): range object class
                contains "min", "max", "scaled_min", "scaled_max" objects

        Returns:
            [np.array]: scaled and clipped value

        """
        scaled_value = self.scaler(
            value, value_range.get_range(), value_range.get_scale_range()
        )
        results = self.clip(scaled_value, value_range.get_scale_range())
        return results

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

    def scale_and_clip_data_obj(self, obj_value, obj_range) -> np.ndarray:
        """scale and clip a data object

        Args:
            obj_value ([rosmsg object]): ros msg object has attributes, i.e. "x", "y", "z"
            obj_range ([range object]): range of value
            attr_list (list, optional): keys of value. Defaults to ["x", "y", "z", "w"].

        Returns:
            np.ndarray: [processed value]

        """
        attr_list = ["x", "y", "z", "w"]

        processed_value = []
        for attr in attr_list:
            try:
                value = getattr(obj_value, attr)
                value_range = obj_range[attr]
                processed_value.append(self.scale_and_clip(value, value_range))
            except AttributeError:
                # print("attr not found", err)
                pass

        return np.array(processed_value)

    def normalize_data_obj_dict(self, obj_dict) -> Dict:
        """normalize data object value by range

        Args:
            obj_dict ([data_obj]): contain value and range

        Returns:
            Dict: dictionary with scaled value
        """
        scaled_obj_dict = {}
        for key, obj in obj_dict.items():
            scaled_obj_dict[key] = self.scale_and_clip_data_obj(obj.value, obj.vrange)

        return scaled_obj_dict

    @classmethod
    def quaternion_from_euler(cls, roll, pitch, yaw) -> tuple:
        """convert a euler angle to a quaternion

        Args:
            roll ([float]): roll angle in rad
            pitch ([float]): pitch angle in rad
            yaw ([float]): yaw angle in rad

        Returns:
            [list]: [quaternion]
        """

        q_x = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        q_y = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        q_z = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        q_w = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)

        return (q_x, q_y, q_z, q_w)

    @classmethod
    def euler_from_quaternion(cls, q_x, q_y, q_z, q_w) -> list:
        """convert quaternion to euler angle

        Args:
            q_x ([float]): [q_x]
            q_y ([float]): [q_y]
            q_z ([float]): [q_z]
            q_w ([float]): [q_w]

        Returns:
            list: [euler angle in radians]
        """
        t_0 = +2.0 * (q_w * q_x + q_y * q_z)
        t_1 = +1.0 - 2.0 * (q_x * q_x + q_y * q_y)
        roll = np.arctan2(t_0, t_1)

        t_2 = +2.0 * (q_w * q_y - q_z * q_x)
        t_2 = +1.0 if t_2 > +1.0 else t_2
        t_2 = -1.0 if t_2 < -1.0 else t_2
        pitch = np.arcsin(t_2)

        t_3 = +2.0 * (q_w * q_z + q_x * q_y)
        t_4 = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)
        yaw = np.arctan2(t_3, t_4)

        return [roll, pitch, yaw]

    @classmethod
    def angle_diff(cls, ang_a: np.ndarray, ang_b: np.ndarray):
        """calculate angle differences,
        consider discontinuety of euler angle at PI, which is 1.0 after scale
        first substract then divide by 2 because range doubled after substraction

        Args:
            ang_a ([np.ndarray]): [angle ranges (-1,1)]
            ang_b ([np.ndarray]): [angle ranges (-1,1)]

        Returns:
            [np.ndarray]: [goal_difference ranges (-1,1)]
        """
        ang_diff = ang_a - ang_b

        bool_ang = ang_diff > 1
        ang_diff[bool_ang] = ang_diff[bool_ang] - 2
        bool_ang = ang_diff < -1
        ang_diff[bool_ang] = ang_diff[bool_ang] + 2

        return ang_diff

    @classmethod
    def quat_diff(cls, q_a, q_b):
        """calculate quaternion differences

        Args:
            q_a ([np.array]): [a quaternion with 4D]
            q_b ([np. array]): [a quaternion with 4D]

        Returns:
            [ang]: [angle between q_a and q_b ranges (0,1)]
        """
        qdot = np.dot(q_a, q_b)
        return np.arccos(np.minimum(np.abs(qdot), 1.0)) * 2.0 / np.pi


class SimpleDataProcessor(DataProcessor):
    """data processor for simplified action interface"""

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
