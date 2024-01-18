#!/usr/bin/env python3

# This script reads a rosbag and replays the state of the provided robot name to create a simulated video.
# Copyright 2023 Bernhard Firner


import argparse
import pathlib
import random
import sys
import time
import yaml

from queue import Queue
from threading import Thread

from arm_utility import (ArmReplay, computeGripperPosition, getCalibrationDiff, rSolver, XYZToRThetaZ)
from data_utility import (ArmDataInterpolator, readArmRecords)
import sim_utility


def recordingToOriginalPosition(positions, joint_names, control_calibration, hand_length):
    calibrated_positions = []
    for joint_name in joint_names:
        joint_index = joint_names.index(joint_name)
        # TODO If these positions were set by the human, aren't they the desired location?
        # That would mean that the calibration should be added.
        calibrated_positions.append(positions[joint_index] - control_calibration[joint_name])
    return calibrated_positions

def recordingToSolvedPosition(positions, joint_names, control_calibration, hand_length):
    calibrated_positions = []
    for joint_name in joint_names:
        joint_index = joint_names.index(joint_name)
        calibrated_positions.append(positions[joint_index] - control_calibration[joint_name])
    rtz = XYZToRThetaZ(*computeGripperPosition(calibrated_positions, segment_lengths=[0.104, 0.158, 0.147, hand_length]))
    middle_joints = rSolver(rtz[0], rtz[2], segment_lengths=[0.104, 0.158, 0.147, hand_length])
    solved_position = [
        # Waist
        rtz[1],
        # Shoulder
        middle_joints[0],
        # Elbow
        middle_joints[1],
        # Wrist angle
        middle_joints[2],
        # Wrist rotate
        calibrated_positions[4],
    ]
    return solved_position

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--puppet_model', default='px150')
    parser.add_argument('--puppet_name', default='arm2')
    parser.add_argument('--src_robot', default='arm2',
        help="The name of the robot in the bag path whose actions to copy.")
    parser.add_argument('--control_calibration', default='configs/arm2_calibration2.yaml',
        help="The source robot's calibration.")
    parser.add_argument(
        '--solver_poses',
        required=False,
        default=False,
        action='store_true',
        help='Transform all poses into the standard ones from arm_utility.rSolver.')
    parser.add_argument(
        '--smooth',
        required=False,
        default=False,
        action='store_true',
        help='Smooth end effector positions if true. Otherwise use simple interpolation.')
    parser.add_argument(
        '--hand_length',
        required=False,
        default=0.175,
        type=float,
        help='The length of the last segment of the arm, in meters.')
    parser.add_argument(
        '--vfov',
        required=False,
        default=61,
        type=float,
        help='The vertical field of view (in degrees).')
    parser.add_argument(
        '--hfov',
        required=False,
        default=92,
        type=float,
        help='The horizontal field of view (in degrees).')
    parser.add_argument(
        '--camera_bases',
        required=False,
        type=float,
        nargs=9,
        default=[0., 0., -1., -1., 0., 0., 0., 1., 0.],
        help="The bases vectors (three values for each basis) for the camera in a right-hand system.")
    parser.add_argument(
        '--robot_bases',
        required=False,
        type=float,
        nargs=9,
        default=[1., 0., 0., 0., 1., 0., 0., 0., 1.],
        help="The bases vectors (three values for each basis) for the camera in a right-hand system.")
    parser.add_argument(
        '--robot_coordinates',
        required=False,
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="x,y,z coordinates (forward, lateral, and vertical position in robot coordinates) of the robot")
    parser.add_argument(
        '--camera_coordinates',
        required=False,
        type=float,
        nargs=3,
        #default=[0.35, 0.0, 0.50],
        default=[-0.5, -0.35, 0.0],
        help="x,y,z coordinates (forward, lateral, and vertical position in camera coordinates) of the camera")
    parser.add_argument(
        '--resolution',
        required=False,
        type=int,
        nargs=2,
        default=[720, 1280],
        help="Height and width of the output images")
    parser.add_argument(
        '--letter_config',
        required=False,
        type=str,
        default=None,
        help="The path to a yaml configuration file with letter locations and orientations."
    )
    parser.add_argument(
        'bag_path',
        type=str,
        help="The path to the bag directory with sql db file.")
    parser.add_argument(
        'pps',
        type=int,
        default=10,
        help="Frames per second. The log will be into this many poses per second.")
    parser.add_argument(
        'output_path',
        type=str,
        help="The video file to save the robot poses into.")
    # TODO
    # * minimum movement speed (to remove human hesitation)
    # * line smoothing (to remove human jitter)

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

    # Get the joint names from the rosbag messages
    joint_names = arm_records[0]["name"][0:5]

    # Get calibration corrections for the robots
    with open(args.control_calibration, 'r') as data:
        control_calibration = yaml.safe_load(data)

    # Set up the record to joint position function
    if not args.solver_poses:
        recordingToNextPosition = recordingToOriginalPosition
    else:
        recordingToNextPosition = recordingToSolvedPosition

    # Create the renderer
    segment_lengths = [0.104, 0.158, 0.147, args.hand_length]
    arm_origin = args.robot_coordinates
    camera_origin = args.camera_coordinates
    camera_fovs = [args.vfov, args.hfov]
    robot_bases = [args.robot_bases[0:3], args.robot_bases[3:6], args.robot_bases[6:]]
    camera_bases = [args.camera_bases[0:3], args.camera_bases[3:6], args.camera_bases[6:]]
    renderer = sim_utility.JointStatesToImage(segment_lengths, arm_origin, robot_bases, camera_fovs, camera_origin, camera_bases, args.resolution)

    # Add the letters to the render, if they are present
    if args.letter_config is not None:
        # Get letter locations
        with open(args.letter_config, 'r') as letter_file:
            letter_locations = yaml.safe_load(letter_file)
        for letter in letter_locations.keys():
            renderer.addLetter(letter, letter_locations[letter])

    renderer.beginVideo(args.output_path)

    # Loop through the arm records

    # Go through the commands at the expected rate.
    cur_time = arm_records[0]['timestamp']
    cur_position = arm_records[0]['position']

    # The records may not align to our update rate, so we will interpolate to get the correct
    # positions.
    arm_interpolate = ArmDataInterpolator(arm_records)

    # Now execute the recording
    # Convert the updates per second into nanoseconds when working with ros2 timestamps.
    thirty_fps_seconds = 1 / 30.
    thirty_fps_nanos = int(thirty_fps_seconds * 10**9)
    total_frames = 0
    try:
        while arm_interpolate.last_idx < len(arm_records):
            # Read the next record and convert into calibrated joint positions
            if args.smooth:
                next_record = arm_interpolate.localAverage(cur_time)
            else:
                next_record = arm_interpolate.interpolate(cur_time)
            calibrated_positions = recordingToNextPosition(next_record['position'], joint_names, control_calibration, args.hand_length)

            # Advance the time for the next record
            cur_time = cur_time + thirty_fps_nanos

            # Render the calibrated positions
            renderer.writeFrame(calibrated_positions)
            total_frames += 1
    except IndexError:
        # We probably ran out of records. Time to exit.
        print("End of records in the position handling thread.")
    renderer.endVideo()

    print("Finished rendering {} frames.".format(total_frames))


if __name__ == '__main__':
    main()
