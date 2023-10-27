#!/usr/bin/env python3

# This script reads a rosbag and replays the state of the provided robot name at the given command
# rate.
# Copyright 2023 Bernhard Firner


import argparse
import pathlib
import sys
import time

from queue import Queue
from threading import Thread

from arm_utility import (ArmReplay, getCalibrationDiff)
from data_utility import (ArmDataInterpolator, readArmRecords)


def position_handling_thread(robot_joint_names, control_calibration, arm_records, position_queue, update_delay_s):
    """Puts positions into the position queue.

    Arguments:
        robot_joint_names (list[str]): Joint names in the robot arm.
        control_calibration (dict): Joint angle calibrations (add to get their calibration position)
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
    calibrated_positions = []
    for joint_name in robot_joint_names:
        joint_index = message_joint_names.index(joint_name)
        calibration_positions.append(cur_position[joint_index] + control_calibration[joint_name])
    position_queue.put((calibrated_positions, 0.75))

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
            calibrated_positions = []
            for joint_name in robot_joint_names:
                joint_index = message_joint_names.index(joint_name)
                calibration_positions.append(cur_position[joint_index] + control_calibration[joint_name])

            position_queue.put((calibrated_positions, 0.75))
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
    parser.add_argument('--src_robot', default='arm2',
        help="The name of the robot in the bag path whose actions to copy.")
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
    # TODO FIXME Try using the command messages rather than the joint states themselves, the
    # actual poses of the robot are probably a combination of the goal position and gravity.
    # Or, alternatively, increase the gain values for position.
    arm_topic = f"/{args.src_robot}/joint_states"
    arm_records = readArmRecords(args.bag_path, arm_topic)

    # Get calibration corrections for the robots
    with open(args.control_calibration, 'r') as data:
        control_calibration = yaml.safe_load(data)
    with open(args.puppet_calibration, 'r') as data:
        puppet_calibration = yaml.safe_load(data)

    position_commands = Queue()
    bot = ArmReplay(args.puppet_model, args.puppet_name, puppet_calibration, position_commands)

    # Start the record reading thread before calling the blocking start_robot() call.
    reader = Thread(target=position_handling_thread,
        args=(bot.arm.group_info.joint_names, control_calibration, arm_records, bot.position_commands, (1. / args.cps))).start()

    # This will block until the replay is complete.
    bot.start_robot()

    # TODO FIXME: Go to a home position?
    print("Finished replaying.")


if __name__ == '__main__':
    main()
