#!/usr/bin/env python3

# This script renders synthetic data with a robot arm touching a sequence of letters.
# Copyright 2024 Bernhard Firner


import argparse
import csv
import io
import math
import os
import pathlib
import random
import sys
import time
import yaml

from pathlib import Path
from queue import Queue
from threading import Thread

from arm_utility import (rSolver, XYZToRThetaZ)
from data_utility import (writeLabels, writeYamlArmRecords)
import sim_utility

def xyzToSolvedPosition(xyz_positions, hand_length):
    rtz = XYZToRThetaZ(*xyz_positions)
    middle_joints = rSolver(rtz[0], rtz[2], segment_lengths=[0.104, 0.158, 0.147, hand_length])
    if False == middle_joints[0]:
        return None
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
        0.,
    ]
    return solved_position

def pointPlusNoise(point, offsets, noise):
    """For each point, gaussian with mean = value + offset and stddev = noise"""
    return [random.gauss(mu = value + offset, sigma = noise) for value, offset, noise in zip(point, offsets, noise)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--puppet_model',
        default='px150',
        help='The model of the simulated robot arm.'
    )
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
        '--robot_speed',
        required=False,
        type=float,
        default=0.10,
        help="The apparent arm speed, in meters per second.")
    parser.add_argument(
        'fps',
        type=int,
        default=30,
        help="Frames per second in the video.")
    parser.add_argument(
        'output_path',
        type=str,
        help="The directory serving as the base path for the video, labels, etc. Will create "
        "intermediate paths and overwrite data in existing directories.")
    parser.add_argument(
        'letter_config',
        type=str,
        help="The path to a yaml configuration file with letter locations and orientations."
    )
    parser.add_argument(
        'action_config',
        type=str,
        help="The path to a yaml configuration file with actions to take."
    )

    args = parser.parse_args(sys.argv[1:])

    # Create the renderer
    segment_lengths = [0.104, 0.158, 0.147, args.hand_length]
    arm_origin = args.robot_coordinates
    camera_origin = args.camera_coordinates
    camera_fovs = [args.vfov, args.hfov]
    robot_bases = [args.robot_bases[0:3], args.robot_bases[3:6], args.robot_bases[6:]]
    camera_bases = [args.camera_bases[0:3], args.camera_bases[3:6], args.camera_bases[6:]]
    renderer = sim_utility.JointStatesToImage(segment_lengths, arm_origin, robot_bases, camera_fovs, camera_origin, camera_bases, args.resolution)

    # Add the letters to the render, if they are present
    # Get letter locations
    letter_centers = {}
    with open(args.letter_config, 'r') as letter_file:
        letter_locations = yaml.safe_load(letter_file)
    for letter in letter_locations.keys():
        renderer.addLetter(letter, letter_locations[letter])
        letter_centers[letter] = [
            letter_locations[letter][0][0] + (letter_locations[letter][2][0] - letter_locations[letter][0][0])/2.,
            letter_locations[letter][0][1] + (letter_locations[letter][2][1] - letter_locations[letter][0][1])/2.,
            letter_locations[letter][0][2] + (letter_locations[letter][2][2] - letter_locations[letter][0][2])/2.,
        ]

    # Open the output path
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Prepare the video rendering and timestamp file
    renderer.beginVideo(os.path.join(args.output_path, "sim_video.mp4"))
    # TODO Technically this should be checked for failure.
    timestamp_csv_file = io.open(os.path.join(args.output_path, "sim_video.csv"), "w")
    timestamps = csv.writer(timestamp_csv_file, delimiter=",")
    timestamps.writerow(["frame_number", "time_sec", "time_ns"])

    # Labelling setup
    labels_file = os.path.join(args.output_path, "labels.yaml")
    labels = {}
    # Segment behaviors and maneuver begin/end marks. All set to None unless there is a label
    labels['behavior'] = []
    labels['mark'] = []

    # Get the action config
    with open(args.action_config, 'r') as action_file:
        actions = yaml.safe_load(action_file)

    # Set up some local variables
    cur_position = actions['arm_start']
    total_frames = 0
    sequence_completions = 0
    sequence_position = 0
    speed_per_frame = args.robot_speed / args.fps
    # ROS uses nanoseconds
    nanos_per_frame = (1.0 / args.fps) * 10**9

    # For the arm records used during dataprep
    record_seconds = []
    record_nanos = []
    record_joints = []
    total_distance = 0.
    record_distances = []

    # Get the target position, with noise and offsets as configured
    target_letter = actions['sequence'][sequence_position]
    if "random" != target_letter:
        target_position = pointPlusNoise(letter_centers[target_letter], actions['target_offsets'], actions['target_noise'])
    else:
        # If the target is random then pick an arbitrary point within the allowed range.
        target_position = [random.uniform(*actions['r_range']),
            random.uniform(*actions['t_range']), random.uniform(*actions['z_range'])]

    while sequence_completions < actions['repeats']:
        # Move from cur_position towards the target position
        # Total delta, used to move towards the target and the measure the goal distance
        delta = [target - cur for (cur, target) in zip(cur_position, target_position)]
        distance_remaining = sum([abs(dist) for dist in delta])

        if distance_remaining <= 0.:
            # Remain here, but increment the sequence position.
            sequence_position += 1
            if sequence_position >= len(actions['sequence']):
                sequence_position = 0
                sequence_completions += 1

            # Set the next target position, with random noise
            target_letter = actions['sequence'][sequence_position]
            if "random" != target_letter:
                target_position = pointPlusNoise(letter_centers[target_letter], actions['target_offsets'], actions['target_noise'])
            else:
                # If the target is random then pick an arbitrary point within the allowed range.
                target_position = [random.uniform(*actions['r_range']),
                    random.uniform(*actions['t_range']), random.uniform(*actions['z_range'])]
            labels['mark'].append(None)
        elif distance_remaining <= speed_per_frame:
            # Move to the target position
            cur_position = target_position
            labels['mark'].append(target_letter)
            total_distance += distance_remaining
        else:
            # Incrementally approach at args.robot_speed. The speed_per_frame < distance_remaining
            # (the alternative was checked in the previous if condition).
            increment = speed_per_frame / distance_remaining
            cur_position = [cur + inc for (cur, inc) in zip(cur_position, [axis * increment for axis in delta])]
            labels['mark'].append(None)
            total_distance += speed_per_frame

        joint_positions = xyzToSolvedPosition(xyz_positions=cur_position, hand_length=args.hand_length)
        if joint_positions is None:
            print("Position {} cannot be reached. Aborting.".format(cur_position))
            return

        # Render the joint positions and timestamps
        renderer.writeFrame(joint_positions)
        timestamps.writerow([
            total_frames,
            total_frames // args.fps,
            math.floor((total_frames % args.fps) * nanos_per_frame)])
        labels['behavior'].append('keep')
        record_seconds.append(total_frames // args.fps)
        record_nanos.append((total_frames % args.fps) * nanos_per_frame)
        record_joints.append(joint_positions)
        record_distances.append(total_distance)
        total_frames += 1
    renderer.endVideo()
    # Close the csv file
    timestamp_csv_file.close()
    # Write the labels
    writeLabels(labels_file, labels)
    # Write the arm records
    writeYamlArmRecords(os.path.join(args.output_path, "records.yaml"), record_seconds, record_nanos, record_joints, record_distances)

    print("Finished rendering {} frames.".format(total_frames))


if __name__ == '__main__':
    main()
