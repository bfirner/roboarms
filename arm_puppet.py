#!/usr/bin/env python3

# This script subscribes to the state of the robot named "arm1" and replicated its actions with a
# robot named "arm2."
# The actions and state of the robot are recorded for later training.
# Copyright 2023 Bernhard Firner

# We assume that manipulator.py will run and control a robot named "arm2". This program control a
# robot named arm2.

# First launch the controller for the second arm:
# ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150 mode_configs:=configs/modes_2.yaml robot_name:=arm2 use_gripper:=true load_configs:=false

# Set write_eeprom_on_startup to false after the first time flashing a robot with the
# load_configs:=false option.

# For sim launch:
# ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150 use_sim:=true robot_name:=arm2 use_gripper:=true

# The /arm1/robot_state_publisher has several topics that describe the manipulator position:
# > Use ros2 topic echo <topic name>
# > /arm1/joint_states [sensor_msgs/msg/JointState]
# > /arm1/robot_description [std_msgs/msg/String]
# The joint states look like this:
# > header:
# >   stamp:
# >     sec: 1685051164
# >     nanosec: 266295673
# >   frame_id: ''
# > name:
# > - waist
# > - shoulder
# > - elbow
# > - wrist_angle
# > - wrist_rotate
# > - gripper
# > - left_finger
# > - right_finger
# > position:
# > - 0.003067961661145091
# > - -1.7855536937713623
# > - 1.6889128684997559
# > - 0.5399612784385681
# > - -0.0076699042692780495
# > - 0.5737088322639465
# > - 0.02852068655192852
# > - -0.02852068655192852
# > velocity:
# > - 0.0
# > - 0.0
# > - 0.0
# > - 0.0
# > - 0.0
# > - 0.0
# > - 0.0
# > - 0.0
# > effort:
# > - 0.0
# > - 0.0
# > - 0.0
# > - -29.59000015258789
# > - 0.0
# > - 0.0
# > - 0.0
# > - 0.0


import argparse
import rclpy
import sys
import time
import yaml
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from std_msgs.msg import String

from sensor_msgs.msg import JointState
from interbotix_common_modules.angle_manipulation import angle_manipulation as ang
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

class ArmPuppet(InterbotixManipulatorXS):
    # This class has a core type through which we will access robot data.

    waist_step = 0.06
    rotate_step = 0.04
    translate_step = 0.01
    gripper_pressure_step = 0.125
    # Speed of the control loop in Hz
    current_loop_rate = 30
    joint_names = None
    com_position = None
    com_velocity = None
    com_effort = None
    grip_moving = 0

    def __init__(self, pargs, corrections, args=None):
        InterbotixManipulatorXS.__init__(
            self,
            robot_model=pargs.puppet_model,
            robot_name=pargs.puppet_name,
            moving_time=0.2,
            accel_time=0.1,
            start_on_init=True,
            args=args
        )
        self.corrections = [0,0,0,0,0]
        for joint_name in corrections.keys():
            # Don't try to correct the gripper or finger joints, only the arm joints
            if joint_name in self.arm.group_info.joint_names:
                joint_idx = self.arm.group_info.joint_names.index(joint_name)
                self.corrections[joint_idx] = corrections[joint_name]

        self.rate = self.core.create_rate(self.current_loop_rate)

        # The actual minimum grip number comes from self.gripper.gripper_info.joint_lower_limits[0],
        # but in the gripper logic only this internal variable is used. This is the most
        # straightforward way to change it.
        self.gripper.left_finger_lower_limit -= 0.0012

        time.sleep(0.5)
        
        # TODO Home position is a non-thunking position since nothing will contact.
        # TODO Really need to calibrate the arms since they don't match.
        #self.arm.go_to_sleep_pose()
        self.core.get_logger().info('Ready to receive puppet commands.')
        self.sub_controller = self.core.create_subscription(
            msg_type=JointState,
            topic=f"/{pargs.control_name}/joint_states",
            callback=self.controller_state_update,
            qos_profile=1,
        )

    def start_robot(self) -> None:
        try:
            self.start()
            while rclpy.ok():
                self.controller()
                self.rate.sleep()
        except KeyboardInterrupt:
            self.shutdown()

    def controller_state_update(self, msg: JointState) -> None:
        if self.joint_names is None:
            self.core.get_logger().info(f"Initializing from message {msg}")
            self.joint_names = msg.name

        if self.com_position is None or self.com_position != msg.position:
            # TODO FIXME Should move into the initial position slowly -- or have the other robot
            # begin from home, but that will make puppeting it difficult
            #if self.com_position is None:
                # Move into the initial position
                #self.core.get_logger().info(f"Moving to initial puppet position.")
                #self.arm.set_joint_positions(joint_positions=msg.position[:5],
                #        moving_time=2.0, blocking=True)
            self.com_position = msg.position
            # Apply the calibration correction
            for idx in range(len(self.corrections)):
                self.com_position[idx] += self.corrections[idx]
            self.com_velocity = msg.velocity
            self.com_effort = msg.effort

    def cur_grip(self) -> float:
        return self.core.joint_states.position[self.gripper.left_finger_index]

    def controller(self) -> None:
        """Update the puppet robot position based upon the controller."""
        # The first five positions correspond to the arm
        # TODO Pull in from the arm.group_info.joint_names instead
        arm_joints = 5
        # Robot seems to 'die' when put into strange positions, not clear what happens. Could be
        # that too many high torque commands are being sent rapidly.
        # TODO FIXME self.core.joint_states.position already has self.pup_position, get rid of that
        # stuff.
        if self.com_position is not None and self.core.joint_states.position is not None:
            # Check for actual differences in the position
            joint_distance = 0.
            for i in range(arm_joints):
                joint_distance += abs(self.com_position[i] - self.core.joint_states.position[i]) / arm_joints
            # If the average distance is greater than 0.002 radians then move to match
            if joint_distance > 0.002:
                succ = self.arm.set_joint_positions(joint_positions=self.com_position[:arm_joints],
                moving_time=1.0/self.current_loop_rate, blocking=False)
                #if not succ:
                #    self.core.get_logger().warn('Attempting to move puppet beyond max positions.')
            # The last three joints are for the gripper.
            # They are: gripper, left_finger, right_finger. We can look at any of the values to see
            # if the gripper should move, but since self.gripper.left_finger_lower_limit and
            # left_finger_upper_limit are used to determine the range, we will use the left finger
            # position.
            manip_left_finger = self.com_position[self.gripper.left_finger_index]
            # TODO Only supporting fully open and closed for the moment.
            if self.cur_grip() < 0.9*manip_left_finger and \
                    self.cur_grip() < self.gripper.left_finger_upper_limit:
                # Open
                self.gripper.gripper_controller(effort=1.0*self.gripper.gripper_value,
                        delay=1./self.current_loop_rate)
            elif self.cur_grip() > 1.2*manip_left_finger and \
                    self.cur_grip() > self.gripper.left_finger_lower_limit:
                # Close
                self.gripper.gripper_controller(effort=-1.0*self.gripper.gripper_value,
                        delay=1./self.current_loop_rate)

            #if self.cur_grip() < manip_left_finger - 0.0005 and \
            #        self.cur_grip() < self.gripper.left_finger_upper_limit:
            #    # Open
            #    self.gripper.gripper_controller(effort=1.0*self.gripper.gripper_value, delay=0.)
            #    self.grip_moving = 1
            #elif self.cur_grip() > manip_left_finger + 0.0005 and \
            #        self.cur_grip() > self.gripper.left_finger_lower_limit:
            #    # Close
            #    self.gripper.gripper_controller(effort=-1.0*self.gripper.gripper_value, delay=0.)
            #    self.grip_moving = -1
            #elif self.grip_moving > 0 and self.cur_grip() > manip_left_finger:
            #    # Stop the gripper, we've opened more than the manipulator
            #    self.gripper.gripper_controller(effort=0.001, delay=0.)
            #    self.grip_moving = 0
            #elif self.grip_moving < 0 and self.cur_grip() < manip_left_finger:
            #    # Stop the gripper, we've closed more than the manipulator
            #    self.gripper.gripper_controller(effort=-0.001, delay=0.)
            #    self.grip_moving = 0


def get_calibration_diff(manip_yaml, puppet_yaml) -> dict:
    """Find the differences between the manipulator and puppet servos.

    Arguments:
        manip_yaml  (str): Path to manipulator calibration
        puppet_yaml (str): Path to puppet calibration
    Returns:
        correction (dict[str, float]): Correction to apply to manipulator positions.
    """
    manip_values = {}
    puppet_values = {}

    with open(manip_yaml, 'r') as data:
        manip_values = yaml.safe_load(data)

    with open(puppet_yaml, 'r') as data:
        puppet_values = yaml.safe_load(data)

    # Do some sanity checks
    if 0 == len(manip_values) or 0 == len(puppet_values):
        raise RuntimeError(f"Failed to load calibration values.")

    if list(manip_values.keys()) != list(puppet_values.keys()):
        raise RuntimeError(f"Calibration parameters do not match.")

    # Create the corrections and return time.
    corrections = {}
    for joint in manip_values.keys():
        corrections[joint] = puppet_values[joint] - manip_values[joint]

    return corrections


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument('--puppet_model', default='px150')
    p.add_argument('--puppet_name', default='arm2')
    p.add_argument('--control_name', default='arm1')
    p.add_argument('--control_calibration', default='configs/arm1_calibration.yaml')
    p.add_argument('--puppet_calibration', default='configs/arm2_calibration.yaml')
    p.add_argument('args', nargs=argparse.REMAINDER)

    command_line_args = remove_ros_args(args=sys.argv)[1:]
    ros_args = p.parse_args(command_line_args)

    # Get calibration corrections
    corrections = get_calibration_diff(ros_args.control_calibration, ros_args.puppet_calibration)

    bot = ArmPuppet(ros_args, corrections, args=args)
    bot.start_robot()


if __name__ == '__main__':
    main()

