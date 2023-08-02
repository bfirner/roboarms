#!/usr/bin/env python3

# Test the data_utility
# Recording was probably done with something like this:
# > ros2 bag record /arm1/joint_states /arm2/joint_states /image_raw/compressed /arm1/robot_description /arm2/robot_description /camera_info

# arm1 is the manipulator and arm2 is the pupper. Images should be suitable for DNN training.

import io
import pytest

import data_utility

example_config = {
    'port': '/dev/ttyUSB0',
    'groups':
    {
        'arm':
        {
            'operating_mode': 'position',
            'profile_type': 'time',
            'profile_velocity': 2000,
            'profile_acceleration': 300,
            'torque_enable': True,
        }
    },
    'singles':
    {
        'gripper':
        {
            'operating_mode': 'pwm',
            'torque_enable': True,
        }
    }
}
example_config_str = """
port: /dev/ttyUSB0

groups:
  arm:
    operating_mode: position
    profile_type: time
    profile_velocity: 2000
    profile_acceleration: 300
    torque_enable: true

singles:
  gripper:
    operating_mode: pwm
    torque_enable: true
"""

def test_readLabels(tmp_path):
    # Write out an example table with known yaml representation
    path = tmp_path / "config.yaml"
    testfile = io.open(path, "w")

    # Write out a test yaml to read in.
    testfile.write(example_config_str)
    testfile.close()
    config_values = data_utility.readLabels(path)
    assert example_config == config_values


def test_writeLabels(tmp_path):
    # Write out an example table and then read it back
    path = tmp_path / "config.yaml"

    data_utility.writeLabels(path, example_config)
    config_values = data_utility.readLabels(path)
    assert example_config == config_values
