#!/usr/bin/env python3

# This script can be used to help find the calibration values of one of the interbotix arms.
# The 
# Copyright 2023 Bernhard Firner


import argparse
import cv2
import ffmpeg
import numpy
import pathlib
import rclpy
import sys
import time
import torch
import yaml

from queue import Queue

# Ros includes
from rclpy.node import Node
from sensor_msgs.msg import (Image, JointState)

# Includes from this project
from arm_utility import (ArmReplay)
from data_utility import vidSamplingCommonCrop



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm_model', default='px150')
    parser.add_argument('--arm_name', default='arm1')
    parser.add_argument('--arm_calibration', default='configs/arm1_calibration.yaml',
        help="The calibration that is being tuned.")

    # Parse the arguments
    args = parser.parse_args(sys.argv[1:])

    # Set up the queue for communication from the neural network to the robot
    position_commands = Queue()

    # Create the robot
    bot = ArmReplay(args.arm_model, args.arm_name, corrections={}, cmd_queue=position_commands)

    # Get calibration corrections and start the robot
    with open(args.arm_calibration, 'r') as data:
        calibration = yaml.safe_load(data)

    # The calibration is added to the position to correct for any offsets, so subtract from 0 to get
    # the values that should send the robot to the 0 position
    zero_position = [calibration[joint_name] for joint_name in bot.arm.group_info.joint_names]

    # Go to the calibrated zero position in a second
    bot.goto(position=zero_position, delay=1.0)

    # Simply end. To tune, change the values in the calibration file and rerun.


if __name__ == '__main__':
    main()
