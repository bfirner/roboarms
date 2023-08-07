#!/usr/bin/env python3

# This script reads a rosbag and replays the state of the provided robot name at the given command
# rate.
# Copyright 2023 Bernhard Firner


import argparse
import pathlib
import rclpy
import sys
import time
import yaml

from queue import Queue
from threading import Thread

from data_utility import (ArmDataInterpolator, readArmRecords)

from interbotix_common_modules.angle_manipulation import angle_manipulation as ang
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

class ArmReplay(InterbotixManipulatorXS):
    # This class has a core type through which we will access robot data.

    waist_step = 0.06
    rotate_step = 0.04
    translate_step = 0.01
    gripper_pressure_step = 0.125
    # Speed of the control loop in Hz
    # This must be faster than the rate that position commands are filled
    current_loop_rate = 60
    position_commands = Queue()
    joint_names = None
    com_position = None
    com_velocity = None
    com_effort = None
    grip_moving = 0

    def __init__(self, puppet_model, puppet_name, corrections, args=None):
        InterbotixManipulatorXS.__init__(
            self,
            robot_model=puppet_model,
            robot_name=puppet_name,
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
        print("Joint position corrections from calibration are {}".format(self.corrections))

        self.rate = self.core.create_rate(self.current_loop_rate)

        # The actual minimum grip number comes from self.gripper.gripper_info.joint_lower_limits[0],
        # but in the gripper logic only this internal variable is used. This is the most
        # straightforward way to change it.
        self.gripper.left_finger_lower_limit -= 0.0012

        # Home position is a non-thunking position since nothing will contact.
        self.arm.go_to_home_pose(moving_time = 1.0, blocking = True)
        self.core.get_logger().info('Arm is in home position {}'.format(self.core.joint_states.position[:5]))

        self.core.get_logger().info('Ready to receive commands.')

    def start_robot(self) -> None:
        try:
            self.start()
            while rclpy.ok():
                self.rate.sleep()
                while not self.position_commands.empty():
                    position, delay = self.position_commands.get()
                    # Finish if a 'None' is provided for the position
                    if position is None:
                        print("No more positions. Robot shutting down.")
                        self.shutdown()
                        return
                    self.goto(position, delay)
            print("rclpy status is not okay. Robot shutting down.")
        except KeyboardInterrupt:
            print("Interrupt received. Robot shutting down.")
            self.shutdown()

    def cur_grip(self) -> float:
        return self.core.joint_states.position[self.gripper.left_finger_index]

    def goto(self, position, delay) -> None:
        """Update the puppet robot position.

        This is a blocking call.

        Arguments:
            position (list[float]): Joint positions
            delay          (float): Delay in seconds
        Returns:
            (bool): Success of the operation
        """
        # The first five positions correspond to the arm
        arm_joints = 5
        self.core.get_logger().info('Moving to {} in {}s.'.format(position[:arm_joints], delay))
        succ = self.arm.set_joint_positions(joint_positions=position[:arm_joints],
            moving_time=delay, blocking=False)
        print("Movement success: {}".format(succ))


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


def position_handling_thread(robot_joint_names, arm_records, position_queue, update_delay_s):
    """Puts positions into the position queue.

    Arguments:
        robot_joint_names (list[str]): Joint names in the robot arm.
        arm_records (ros records):
        position_queue    (Queue): Input queue to the robot
        update_delay_s    (float): Update delay in seconds
    """
    print("Record reading thread started.")
    print("Joint names being controlled are {}".format(robot_joint_names))
    print("Replaying from {} joint messages.".format(len(arm_records)))

    # We will need to map from the message joints onto the robot joints. They should match if the
    # robots are the same, but this could catch some errors later.
    message_joint_names = arm_records[0]['name']

    # Go through the commands at the expected rate.
    cur_time = arm_records[0]['timestamp']
    cur_position = arm_records[0]['position']
    # Go to the first position in three quarters of a seconds, just in case it is far away
    position_queue.put(([cur_position[message_joint_names.index(name)] for name in robot_joint_names], 0.75))

    # Wait for the robot to finish that movement before cramming the queue full with the faster
    # update rate of later commands.
    #time.sleep(0.75)

    # The records may not align to our update rate, so we will interpolate to get the correct
    # positions.
    arm_interpolate = ArmDataInterpolator(arm_records)

    # Now execute the recording
    # Convert the updates per second into nanoseconds when working with ros2 timestamps.
    update_delay_nanos = int(update_delay_s * 10**9)
    try:
        while arm_interpolate.last_idx < len(arm_records):
            cur_time = cur_time + update_delay_nanos
            next_record = arm_interpolate.interpolate(cur_time)
            position_queue.put(([next_record['position'][message_joint_names.index(name)] for name in robot_joint_names], 0.75))
            time.sleep(update_delay_s)
    except IndexError:
        # We probably ran out of records. Time to exit.
        print("End of records in the position handling thread.")

    # Tell the arm that we are doin with actuation.
    position_queue.put((None, None))
    print("Message replaying thread ending.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--puppet_model', default='px150')
    parser.add_argument('--puppet_name', default='arm2')
    parser.add_argument('--control_calibration', default='configs/arm2_calibration.yaml',
        help="If control_calibration and puppet_calibration differ, correct during replay.")
    parser.add_argument('--puppet_calibration', default='configs/arm2_calibration.yaml',
        help="If control_calibration and puppet_calibration differ, correct during replay.")
    parser.add_argument(
        'bag_path',
        type=str,
        help="The path to the bag directory with sql db file.")
    parser.add_argument(
        'cps',
        type=int,
        default=10,
        help="Commands per second. The log will be converted to this rate.")
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args(sys.argv[1:])

    # Read in the ros messages and replay them to the robot
    # Check for required paths
    path = pathlib.Path(args.bag_path)

    # There should be a single file match for each of these paths. Do a quick sanity check.
    db_paths = list(path.glob("rosbag2*.db3"))

    if 0 == len(db_paths):
        print("No database found in bag path {}".format(args.bag_path))
        return
    if 1 < len(db_paths):
        print("Too many (expecing 1) db files found in {}".format(args.bag_path))
        return

    # Open the rosbag db file and read the arm topic
    arm_topic = f"/{args.puppet_name}/joint_states"
    arm_records = readArmRecords(args.bag_path, arm_topic)

    # Get calibration corrections and start the robot
    corrections = get_calibration_diff(args.control_calibration, args.puppet_calibration)

    bot = ArmReplay(args.puppet_model, args.puppet_name, corrections)

    # Start the record reading thread before calling the blocking start_robot() call.
    reader = Thread(target=position_handling_thread,
        args=(bot.arm.group_info.joint_names, arm_records, bot.position_commands, (1. / args.cps))).start()

    # This will block until the replay is complete.
    bot.start_robot()

    # TODO FIXME: Go to a home position?
    print("Finished replaying.")


if __name__ == '__main__':
    main()
