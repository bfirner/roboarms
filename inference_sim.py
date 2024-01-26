#!/usr/bin/env python3

# This script runs a trained neural network to control a robot arm in a basic simulator.
# Copyright 2024 Bernhard Firner


import argparse
import cv2
import ffmpeg
import numpy
import pathlib
import sys
import time
import torch
import yaml

from queue import Queue
from threading import Event, Lock, Thread

# Ros includes
from rclpy.node import Node
from sensor_msgs.msg import (Image, JointState)

# Includes from this project
from arm_utility import (ArmReplay, computeGripperPosition, getDistance, interpretRTZPrediction, rSolver, RThetaZtoXYZ, XYZToRThetaZ)
from data_utility import vidSamplingCommonCrop


# Insert the bee analysis repository into the path so that the python modules of the git submodule
# can be used properly.
import sys
sys.path.append('bee_analysis')

from bee_analysis.utility.model_utility import (createModel2, hasNormalizers, restoreModel,
        restoreNormalizers)


# Terminate inference when the user kills the program
# TODO Use an event to terminate instead of a variable
exit_event = Event()
def handler(signum, frame):
    exit_event.set()

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
        'letter_config',
        type=str,
        help="The path to a yaml configuration file with letter locations and orientations."
    )
    parser.add_argument(
        'action_config',
        type=str,
        help="The path to a yaml configuration file with actions to take."
    )
    ############# DNN arguments
    parser.add_argument(
        '--video_scale',
        type=float,
        required=False,
        default=1.0,
        help="Scaling to apply to x and y dimensions before cropping."
        "A value of 0.5 will yield 0.25 resolution.")
    parser.add_argument(
        '--crop_x_offset',
        type=int,
        required=False,
        default=0,
        help='The offset (in pixels) of the crop location on the original image in the x dimension.')
    parser.add_argument(
        '--crop_y_offset',
        type=int,
        required=False,
        default=0,
        help='The offset (in pixels) of the crop location on the original image in the y dimension.')
    parser.add_argument(
        '--width',
        type=int,
        required=False,
        default=224,
        help='Width of output images (obtained via cropping, after applying scale).')
    parser.add_argument(
        '--height',
        type=int,
        required=False,
        default=224,
        help='Height of output images (obtained via cropping, after applying scale).')
    parser.add_argument(
        '--out_channels',
        type=int,
        required=False,
        choices=[1, 3],
        default=3,
        help='Channels of output images.')
    parser.add_argument(
        '--modeltype',
        type=str,
        required=False,
        default="alexnet",
        choices=["alexnet", "resnet18", "resnet34", "bennet", "compactingbennet", "dragonfly", "resnext50", "resnext34", "resnext18",
        "convnextxt", "convnextt", "convnexts", "convnextb"],
        help="Model to use for training.")
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        required=True,
        help='File with model weights to restore.')
    parser.add_argument(
        '--interval',
        type=int,
        required=False,
        default=0,
        help='Frames to skip between frames in a sample.')
    parser.add_argument(
        '--frames_per_sample',
        type=int,
        required=False,
        default=1,
        help='Number of frames in each sample.')
    parser.add_argument(
        '--image_topic',
        type=str,
        required=False,
        default="/image_raw",
        help='The name of the image topic.')
    parser.add_argument(
        '--dnn_outputs',
        type=str,
        # Support an array of strings to have multiple different label targets.
        nargs='+',
        required=False,
        default=["target_position"],
        help='DNN outputs')
    parser.add_argument(
        '--vector_inputs',
        type=str,
        # Support an array of strings to have multiple different label targets.
        nargs='+',
        required=False,
        default=[],
        help='DNN vector inputs')
    parser.add_argument(
        '--normalize_video',
        required=False,
        default=False,
        action="store_true",
        help=("Normalize video: input = (input - mean) / stddev. "
            "Some normalization is already done through ffmpeg, but png and jpeg differences could be fixed with this."))

    # Parse the arguments
    args = parser.parse_args(sys.argv[1:])

    ################ Sim setup
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

    # Get the action config. This may be used for status inputs.
    with open(args.action_config, 'r') as action_file:
        actions = yaml.safe_load(action_file)

    ################ DNN setup
    num_arm_joints = 5
    # Find the size of the model image inputs, vector inputs, and outputs
    vector_input_size = 0
    for input_name in args.vector_inputs:
        # Vector inputs are size 1 unless they are the current robot position
        if input_name == 'current_arm_position':
            vector_input_size += num_arm_joints
        else:
            vector_input_size += 1

    dnn_output_size = 0
    for output_name in args.dnn_outputs:
        # Outputs are size 1 unless they are the predicted robot position
        if output_name == 'target_arm_position':
            dnn_output_size += num_arm_joints
        else:
            dnn_output_size += 1

    # Create the model and load the weights from the given checkpoint.
    checkpoint = torch.load(args.model_checkpoint)
    # Get the model arguments from the training metadata stored in the checkpoint
    net = createModel2(checkpoint['metadata']['modeltype'], checkpoint['metadata']['model_args'])
    restoreModel(args.model_checkpoint, net)
    # Restore the denormalization network, if it was used.
    if hasNormalizers(args.model_checkpoint):
        _, denormalizer = restoreNormalizers(args.model_checkpoint)
        denormalizer.eval().cuda()
    else:
        denormalizer = None
    net.eval().cuda()

    # Initialize the vector inputs tensor
    goal_idx = 0
    vector_input_locations = {}
    vector_size = 0
    # History goal distance is initialized to 10cm
    prev_goal_distance = 0.1
    for input_name in vector_inputs:
        # Vector inputs are size 1 unless they are the current robot position
        if input_name == 'current_position':
            # This uses all joints. 'current_arm_position' has only the arm ones
            vector_input_locations[input_name] = slice(vector_size, vector_size + num_arm_joints + 3)
            vector_size += num_arm_joints + 3
        elif input_name == 'current_arm_position':
            vector_input_locations[input_name] = slice(vector_size, vector_size + num_arm_joints)
            vector_size += num_arm_joints
        elif input_name == 'current_rtz_position':
            vector_input_locations[input_name] = slice(vector_size, vector_size + 3)
            vector_size += 3
        elif input_name == 'goal_mark':
            vector_input_locations[input_name] = vector_size
            vector_size += 1
        elif input_name[:len("goal_distance_prev_")] == "goal_distance_prev_":
            vector_input_locations[input_name] = vector_size
            vector_size += 1
            # TODO We will assume that the prediction_distance and the previous goal distance are
            # the same, so we only need to buffer a single previous goal distance prediction.
            goal_distance_history = int(input_name[len("goal_distance_prev_"):-2])
        else:
            Exception("Unknown vector input: {}".format(input_name))

    # Create tensor inputs
    vector_input_buffer = torch.zeros([1, vector_size]).float().cuda()

    # Mark the locations of the model outputs
    output_locations = {}
    out_idx = 0
    for output_name in args.dnn_outputs:
        if output_name == 'target_position':
            output_locations[output_name] = slice(out_idx, out_idx+num_arm_joints + 3)
            out_idx += num_arm_joints + 3
        elif output_name == 'target_arm_position':
            output_locations[output_name] = slice(out_idx, out_idx+num_arm_joints)
            out_idx += num_arm_joints
        elif output_name == 'target_rtz_position':
            output_locations[output_name] = slice(out_idx, out_idx+3)
            out_idx += num_arm_joints
        elif output_name == 'rtz_classifier':
            rtz_classifier_size = 10
            output_locations[output_name] = slice(out_idx, out_idx+rtz_classifier_size)
            out_idx += rtz_classifier_size
        else:
            output_locations[output_name] = out_idx
            out_idx +=1




    # TODO Need to handle the video by using these parameters
    video_args = {
        # Subscribe to the calibrated joint positions so that processing is robot agnostic
        'arm_topic_string': "/{}/calibrated_joint_states".format(args.puppet_name),
        'video_topic_string': args.image_topic,
        'scale': args.video_scale,
        'crop_width': args.width,
        'crop_height': args.height,
        'crop_x': args.crop_x_offset,
        'crop_y': args.crop_y_offset,
        'frames_per_sample': args.frames_per_sample,
        'channels_per_frame': args.out_channels
    }

    # Start the DNN inference thread before calling the blocking start_robot() call.
    # TODO FIXME Put this setup boilerplate into a common location


    # TODO Initialize robot arm position and renderer

    # Set up some local variables
    cur_position = actions['arm_start']
    total_frames = 0
    sequence_completions = 0

    # Get the target position, with noise and offsets as configured
    target_letter = actions['sequence'][goal_idx]
    target_position = letter_centers[target_letter], actions['target_offsets'], actions['target_noise'])


    # TODO Generate the first frame

    with torch.no_grad():
        # Keep simulating until the exit event is called.
        while not exit_event.is_set():
            # Forward the frame through the DNN model
            # This always uses the most recent frame, so if the update rate is much faster than the
            # frame rate the only thing that will update between commands is the robot's current
            # position.
            # TODO Get the image from the renderer
            # TODO new_frame should have image data
            # TODO joint_positions should have arm state

            new_frame = new_frame.cuda()
            # Normalize inputs: input = (input - mean)/stddev
            if checkpoint['metadata']['normalize_images']:
                v, m = torch.var_mean(new_frame)
                new_frame = (new_frame - m) / v

            # Fill in vector inputs
            for input_name in args.vector_inputs:
                # Vector inputs are size 1 unless they are the current robot position
                outslice = vector_input_locations[input_name]
                if input_name == 'current_position':
                    vector_input_buffer[0, outslice].copy_(torch.tensor(joint_positions))
                elif input_name == 'current_arm_position':
                    vector_input_buffer[0, outslice].copy_(torch.tensor(joint_positions))
                elif input_name == 'current_rtz_position':
                    current_rtz_position = XYZToRThetaZ(*computeGripperPosition(joint_positions, segment_lengths=[0.104, 0.158, 0.147, 0.175]))
                    print("Writing location {} into slice {}".format(current_rtz_position, outslice))
                    vector_input_buffer[0, outslice].copy_(torch.tensor(current_rtz_position))
                elif input_name == 'goal_mark':
                    # Convert the goal character into a number. 'A' becomes 0, and the rest of the
                    # letters are offset from there.
                    vector_input_buffer[0, outslice.start] = ord(actions.sequence[goal_idx]) - ord('A')
                elif input_name[:len("goal_distance_prev_")] == "goal_distance_prev_":
                    vector_input_buffer[0, outslice.start] = prev_goal_distance

            # goal_mark determines the state to feed back to the network via the vector inputs
            # goal_distance is used to determine when to switch goals.
            # The target_position output contains the next set of joint positions.

            if 0 == len(args.vector_inputs):
                net_out = net(new_frame)
            else:
                net_out = net(new_frame, vector_input_buffer)
            if denormalizer is not None:
                net_out = denormalizer(net_out)
            predicted_distance = 1.0
            if 'goal_distance' in args.dnn_outputs:
                net_out[0, output_locations['goal_distance']].item()
            if 'target_arm_position' in output_locations:
                next_position = net_out[0, output_locations['target_arm_position']].tolist()
            elif 'rtz_classifier' in output_locations:
                # TODO This was a training parameter, it should be pulled from the dataset
                # Setting the threshold to 1cm here
                threshold = 0.01
                current_xyz = computeGripperPosition(joint_positions)
                predictions = net_out[0, output_locations['rtz_classifier']].tolist()
                next_rtz_position = interpretRTZPrediction(*XYZToRThetaZ(*current_xyz), threshold, predictions)
                next_xyz_position = RThetaZtoXYZ(*next_rtz_position)
                # Solve for the joint positions
                middle_joints = rSolver(next_rtz_position[0], next_rtz_position[2], segment_lengths=[0.104, 0.158, 0.147, 0.175])
                next_position = [
                    # Waist
                    next_rtz_position[1],
                    # Shoulder
                    middle_joints[0],
                    # Elbow
                    middle_joints[1],
                    # Wrist angle
                    middle_joints[2],
                    # Wrist rotate
                    0.0,
                ]
            else:
                raise RuntimeError("No target position for robot in DNN outputs!")

            # TODO This shouldn't be a magic variable
            if predicted_distance < 0.01:
                goal_idx = (goal_idx + 1) % len(actions.sequence)
                print("Switching to goal {}".format(actions.sequence[goal_idx]))
                # History goal distance is initialized to 10cm
                prev_goal_distance = 0.1
            else:
                # Save the current goal distance to be used as a status input.
                prev_goal_distance = predicted_distance

            # TODO Move to the new position, from the next_position variable

    # Clean up
    # TODO close the renderer

    print("DNN control ending.")


    print("Finished.")


if __name__ == '__main__':
    main()
