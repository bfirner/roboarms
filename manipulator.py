#!/usr/bin/env python3

# This script disables torque on a robot to allow for human manipulation. The gripper state is
# controlled via keyboard inputs. The state of the robot can be used by a puppet node to control a
# second arm.
#
# Copyright 2023 Bernhard Firner

# First launch the controller for the arm1
# ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150 mode_configs:=configs/modes_1.yaml robot_name:=arm1 use_gripper:=true load_configs:=false

# Set write_eeprom_on_startup to false after the first time flashing a robot with the
# load_configs:=false option.

# For sim launch:
# ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150 use_sim:=true robot_name:=arm1 use_gripper:=true

# The API for interbotix_xsarm_control is here:
# https://github.com/Interbotix/interbotix_ros_manipulators/tree/galactic/interbotix_ros_xsarms/interbotix_xsarm_control/demos/python_ros2_api

# To "fully" close the default grippers with attached soft pads, subtract 0.0012 from the default
# value: 0.014999999664723873 - 0.0012 = 0.013799999664723873

import argparse
import copy
import sys
from queue import Queue
from threading import Thread
import time

from interbotix_common_modules.angle_manipulation import angle_manipulation as ang
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
#from interbotix_xs_msgs.msg import ArmJoy
import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args


class ArmController(InterbotixManipulatorXS):
    """
    Disables torque and publishes the arm state.
    """

    waist_step = 0.06
    rotate_step = 0.04
    translate_step = 0.01
    gripper_pressure_step = 0.125
    current_loop_rate = 25
    current_gripper_pressure = 0.5
    loop_rates = {'coarse': 25, 'fine': 25}
    #joy_msg = ArmJoy()
    #joy_mutex = Lock()
    grip_commands = Queue()
    finished = False


    def __init__(self, pargs, args=None):
        InterbotixManipulatorXS.__init__(
            self,
            robot_model=pargs.robot_model,
            robot_name=pargs.robot_name,
            moving_time=0.2,
            accel_time=0.1,
            start_on_init=True,
            args=args
        )
        self.rate = self.core.create_rate(self.current_loop_rate)
        self.num_joints = self.arm.group_info.num_joints
        self.waist_index = self.arm.group_info.joint_names.index('waist')
        self.waist_ll = self.arm.group_info.joint_lower_limits[self.waist_index]
        self.waist_ul = self.arm.group_info.joint_upper_limits[self.waist_index]
        self.T_sy = np.identity(4)
        self.T_yb = np.identity(4)
        self.update_T_yb()

        # The actual minimum grip number comes from self.gripper.gripper_info.joint_lower_limits[0],
        # but in the gripper logic only this internal variable is used. This is the most
        # straightforward way to change it.
        if pargs.adjust_gripper_lower_limit is not None:
            # 0.0046 works okay with the straw.
            self.gripper.left_finger_lower_limit -= pargs.adjust_gripper_lower_limit
        else:
            self.gripper.left_finger_lower_limit -= 0.0012

        # TODO: Normalize the position with the self.gripper.left_finger_upper_limit
        # and self.gripper.left_finger_lower_limit variables.
        self.grip_step = (self.gripper.left_finger_upper_limit - self.gripper.left_finger_lower_limit)/10.0
        #self.core.create_subscription(ArmJoy, 'commands/joy_processed', self.joy_control_cb, 10)
        time.sleep(0.5)
        self.core.get_logger().info('Ready for manipulation.')

    def start_robot(self) -> None:
        try:
            self.start()
            self.core.robot_torque_enable(cmd_type='group', name='arm', enable=False)
            #self.core.robot_torque_enable(cmd_type='single', name='gripper', enable=False)
            #self.update_gripper_pressure(0.0)
            while rclpy.ok() and not self.finished:
                self.controller()
                self.rate.sleep()
            print("Arm controller shutting down.")
        except KeyboardInterrupt:
            self.shutdown()
        self.shutdown()

    def update_T_yb(self) -> None:
        """Calculate the pose of the end-effector w.r.t. T_y."""
        T_sb = self.arm.get_ee_pose_command()
        rpy = ang.rotation_matrix_to_euler_angles(T_sb[:3, :3])
        self.T_sy[:2, :2] = ang.yaw_to_rotation_matrix(rpy[2])
        self.T_yb = np.dot(ang.trans_inv(self.T_sy), T_sb)

    def update_gripper_pressure(self, gripper_pressure: float) -> None:
        """
        Update gripper pressure.

        :param gripper_pressure: desired gripper pressure from 0 - 1
        """
        self.current_gripper_pressure = gripper_pressure
        self.gripper.set_pressure(self.current_gripper_pressure)
        self.core.get_logger().info(
            f'Gripper pressure is at {self.current_gripper_pressure * 100.0:.2f}%.'
        )

    def cur_grip(self) -> float:
        return self.core.joint_states.position[self.gripper.left_finger_index]

    def controller(self) -> None:
        """Run main arm manipulation control loop."""
        # TODO Control the gripper with an external input for easier manipulation
        # Use self.gripper.gripper_controller with effort from -1*gripper_value to
        # close and gripper_value to open. Lower values move more slowly. Command something
        # very close to 0, but not 0 (there are checks for 0) to make it stop moving. There
        # is not command to move it to a fixed position.

        # Wait 50ms before rechecking the gripper position.
        time_delay = 0.05
        # Wait at most 250ms for the gripper to reach a position. If it hasn't, then something is
        # probably in its grasp.
        max_step = 5

        while not self.grip_commands.empty():
            cmd = self.grip_commands.get()
            print(f"Controller got command {cmd}")
            # Interpret this as a command to open or close the gripper
            # Get the gripper pose with
            # self.core.joint_states.position[self.gripper.left_finger_index]
            steps = max_step
            match cmd:
                case 'q':
                    self.finished = True
                case '<':
                    # Close
                    if self.cur_grip() > self.gripper.left_finger_lower_limit:
                        self.gripper.gripper_controller(effort=-1.0*self.gripper.gripper_value,
                                delay=0.)
                case '>':
                    # Open
                    if self.cur_grip() < self.gripper.left_finger_upper_limit:
                            self.gripper.gripper_controller(effort=1.0*self.gripper.gripper_value,
                                    delay=0.)
                case ',':
                    # Close slightly
                    target_position = self.cur_grip() - self.grip_step
                    if target_position > self.gripper.left_finger_lower_limit:
                        self.gripper.gripper_controller(effort=-1.0*self.gripper.gripper_value,
                                delay=0.)
                        while self.cur_grip() > target_position and 0 < steps:
                            time.sleep(time_delay)
                            steps -= 1
                        # Stop the gripper
                        self.gripper.gripper_controller(effort=-0.001, delay=0.)
                case '.':
                    # Open slightly
                    target_position = self.cur_grip() + self.grip_step
                    if target_position < self.gripper.left_finger_upper_limit:
                        self.gripper.gripper_controller(effort=1.0*self.gripper.gripper_value,
                                delay=0.)
                        while self.cur_grip() < target_position and 0 < steps:
                            time.sleep(time_delay)
                            steps -= 1
                        # Stop the gripper
                        self.gripper.gripper_controller(effort=0.001, delay=0.)
                case _:
                    pass


def command_reader(cmd_queue):
    """If a command is read then lock the lock and write the command to cmd_buf."""
    print("Waiting for keyboard inputs.")
    while True:
        try:
            cmd = sys.stdin.read(1)
            if cmd != '\n':
                print(f"Got command {cmd}")
            cmd_queue.put(cmd)
            # Terminate after encountering a 'q'
            if 'q' == cmd:
                return
        except KeyboardInterrupt:
            return


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument('--robot_model', default='px150')
    p.add_argument('--robot_name', default='arm1')
    p.add_argument('--adjust_gripper_lower_limit', type=float, required=False)
    p.add_argument('args', nargs=argparse.REMAINDER)

    command_line_args = remove_ros_args(args=sys.argv)[1:]
    cmd_args = p.parse_args(command_line_args)

    controller = ArmController(cmd_args, args=args)

    reader = Thread(target=command_reader, args=[controller.grip_commands]).start()

    controller.start_robot()
    if reader:
        reader.join()
    return


if __name__ == '__main__':
    main()

