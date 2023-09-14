#!/usr/bin/env python3

# Prepare data for DNN training by converting videos and rosbags into a webdataset.
# TODO In the long-term, webdatasets are not the most optimal storage format for larger amounts of
# data. Instead, a columnar format, such as parquet, may be more suitable.

import argparse
import csv
import cv2
import ffmpeg
import io
import itertools
import math
import modern_robotics
import numpy
import os
import pathlib
import random
import sys
import torch
import webdataset as wds
import yaml

from data_utility import (ArmDataInterpolator, readArmRecords, readLabels, readVideoTimestamps)
from bee_analysis.utility.video_utility import (getVideoInfo, VideoSampler)
# These are the robot descriptions to match function calls in the modern robotics package.
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from nml_bag import Reader
# Helper function to convert to images
from torchvision import transforms

def writeSample(dataset, sample_labels, frames, video_source, frame_nums):
    """Write the given sample to a webdataset.

    Arguments:
        dataset          (TarWriter): Dataset to write into.
        sample_labels ({str: float}): Target labels
        frames        (list[tensor]): The frames.
        video_source           (str): The source of this data.
        frame_nums       (list[int]): The frame numbers.
    Returns:
        None
    """
    frame_count = len(frames)
    channels = frames[0].size(1)
    # Convert the images to pngs
    buffers = []
    for frame in frames:
        img = transforms.ToPILImage()(frame[0]/255.0).convert('RGB')
        # Now save the image as a png into a buffer in memory
        buf = io.BytesIO()
        img.save(fp=buf, format="png")
        buffers.append(buf)
    # TODO FIXME Need to fetch the robot state for this
    base_name = '_'.join([video_source] + [str(fnum) for fnum in frame_nums])
    sample = {
        "__key__": base_name,
    }
    # Write the labels into the dataset
    for key, value in sample_labels.items():
        sample[key] = str(value).encode('utf-8')
    for i in range(frame_count):
        sample[f"{i}.png"] = buffers[i].getbuffer()
    dataset.write(sample)


def getGripperPosition(robot_model, arm_record):
    """Get the x,y,z position of the gripper relative to the waist.

    This is basically the same as the get_ee_pose function for an interbotix arm, but takes in an
    arbitrary set of joint states. Like that function, this just calls something from the
    modern_robotics package.

    The robot model should be fetched from
         getattr(interbotix_xs_modules.xs_robot.mr_descriptions, 'px150')

    Arguments:
        robot_model      (class): String that identifies this robot arm in the interbotix modules.
        arm_record ({str: data}): ROS message data for the robot arm.
    Returns:
        x,y,z tuple of the gripper location.
    """
    # TODO Since a mini-goal of this project is to handle actuation without calibration we won't be
    # using this for labels (because it would require calibration for the x,y,z location of the
    # gripper to be meaningful), but will be using this to determine the moved distance of the
    # gripper, and from that we will determine the next pose to predict.
    # The arm joints are separate from the gripper, which is represented by three "joints" even
    # though it is a single motor.
    gripper_names = ["gripper", "left_finger", "right_finger"]
    names = [name for name in arm_record['name'] if name not in gripper_names]
    joint_positions = [
        arm_record['position'][names.index(name)] for name in names
    ]
    # 'M' is the home configuration of the robot, Slist has the joint screw axes at the home
    # position
    T = modern_robotics.FKinSpace(robot_model.M, robot_model.Slist, joint_positions)
    # Return the x,y,z components of the translation matrix.
    return (T[0][-1], T[1][-1], T[2][-2])


def getDistance(record_a, record_b):
    """Return the Euclidean distance of the manipulator in record a and record b"""
    return math.sqrt(sum([(a-b)**2 for a, b in zip(record_a, record_b)]))


def getStateAtNextPosition(reference_record, arm_records, movement_distance, robot_model):
    """Get the arm state after the end affector moves the given distance

    Arguments:
        reference_record  ({str: data}): Reference position
        arm_records (list[{str: data}]): ROS data for the robot arm. Search begins at index 0.
        movement_distance (float): Distance in meters desired from the reference position.
        robot_model      (class): String that identifies this robot arm in the interbotix modules.
    Returns:
        tuple({str: data}, int): The arm record at the desired distance, or None if the records end
                                 before reaching the desired distance. Also returns the index of
                                 this record.
    """
    # Loop until we run out of records or hit the desired movement distance.
    reference_position = getGripperPosition(robot_model, reference_record)
    distance = 0
    idx = 0
    next_record = None
    # Search for the first record with the desired distance
    while idx < len(arm_records) and distance < movement_distance:
        next_record = arm_records[idx]
        next_position = getGripperPosition(robot_model, next_record)
        # Find the Euclidean distance from the reference position to the current position
        distance = getDistance(reference_position, next_position)
        idx += 1

    # If we ended up past the end of the records then they don't have anything at the desired
    # distance
    if idx >= len(arm_records):
        return None, None

    return next_record, idx-1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'outpath',
        type=str,
        help='Output path for the WebDataset.')
    parser.add_argument(
        'bag_paths',
        nargs='+',
        type=str,
        help="The paths to the bag directories, with sql db, video, and timestamps csv.")
    parser.add_argument(
        '--label_file',
        type=str,
        required=False,
        default='labels.yaml',
        help="The name of the labels file inside of the ros directories.")
    parser.add_argument(
        '--train_robot',
        type=str,
        default='arm1',
        help="The robot whose state is used as the training target.")
    parser.add_argument(
        '--video_scale',
        type=float,
        required=False,
        default=1.0,
        help="Scaling to apply to x and y dimensions before cropping."
        "A value of 0.5 will yield 0.25 resolution.")
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
        '--crop_noise',
        type=int,
        required=False,
        default=0,
        help='The noise (in pixels) to randomly add to the crop location in both the x and y axis.')
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
        '--sample_prob',
        type=float,
        required=False,
        default=0.2,
        help='Probability of sampling any frame.')
    parser.add_argument(
        '--out_channels',
        type=int,
        required=False,
        choices=[1, 3],
        default=3,
        help='Channels of output images.')
    parser.add_argument(
        '--threads',
        type=int,
        required=False,
        default=1,
        help='Number of thread workers')
    parser.add_argument(
        '--prediction_distance',
        type=float,
        required=False,
        default=0.01,
        help='Distance (in meters) of the future position of the robot gripper in the dataset.')
    parser.add_argument(
        '--goals',
        type=int,
        # Goals to give the robot arm.
        nargs='+',
        required=True,
        default=[],
        help='Goals to give the robot arm.')
    parser.add_argument(
        '--robot_model',
        type=str,
        required=False,
        default="px150",
        help='The name of the interbotix robot model.')
    parser.add_argument(
        '--nonormalize',
        required=False,
        default=False,
        action='store_true',
        help='Images are normalized by default, but this can be disabled.')

    args = parser.parse_args()

    # TODO Everything

    # Create a writer for the WebDataset
    datawriter = wds.TarWriter(args.outpath, encoder=False)

    model_generator = getattr(mrd, args.robot_model)
    robot_model = model_generator()

    # Loop over each rosbag directory
    # TODO Split this part into threads
    for rosdir in args.bag_paths:
        print("Processing {}".format(rosdir))
        # Check for required paths
        path = pathlib.Path(rosdir)

        # There should be a single file match for each of these paths.
        db_paths = list(path.glob("rosbag2*.db3"))
        ts_paths = list(path.glob("robo_video_*.csv"))
        vid_paths = list(path.glob("robo_video_*.mp4"))

        # Perform some sanity checks
        if 0 == len(db_paths):
            print("No database found in bag path {}".format(rosdir))
            return
        if 1 < len(db_paths):
            print("Too many (expecing 1) db files found in {}".format(rosdir))
            return

        if 0 == len(ts_paths):
            print("No video timestamp found in bag path {}".format(rosdir))
            return
        if 1 < len(ts_paths):
            print("Too many (expecing 1) video timestamp files found in {}".format(rosdir))
            return
        ts_path = ts_paths[0]

        if 0 == len(vid_paths):
            print("No database found in bag path {}".format(rosdir))
            return
        if 1 < len(vid_paths):
            print("Too many (expecing 1) db files found in {}".format(rosdir))
            return
        vid_path = vid_paths[0]

        ################
        # Data loading
        # Set up readers for the timestamps, video, and database
        video_timestamps = readVideoTimestamps(ts_path)

        # Open the rosbag db file and read the arm topic
        arm_topic = f"/{args.train_robot}/joint_states"
        arm_data = ArmDataInterpolator(readArmRecords(rosdir, arm_topic))

        # Now go through video frames.
        width, height, total_frames = getVideoInfo(vid_path)
        print("Found {} frames in the video {}".format(total_frames, vid_path))

        # Get the labels
        if args.label_file is None:
            args.label_file = "labels.yaml"
        label_path = os.path.join(rosdir, args.label_file)
        labels = readLabels(label_path)

        # Record the mark times and names to make things easier when producing labels
        marks = []
        last_mark = None
        for idx, mark in enumerate(labels['mark']):
            # Fill in the last mark if there is no marker at this frame
            # TODO Also fill in some progress measurement, in terms of total motion towards the goal
            if mark is None:
                marks.append(last_mark)
            else:
                last_mark = mark
                marks.append(mark)

        # Loop through the video frames, exporting as requested into the webdataset
        num_samples = int(args.sample_prob * (total_frames // args.frames_per_sample))
        sampler = VideoSampler(vid_path, num_samples, args.frames_per_sample, args.interval,
            out_width=args.width, out_height=args.height, crop_noise=args.crop_noise,
            scale=args.video_scale, crop_x_offset=args.crop_x_offset,
            crop_y_offset=args.crop_y_offset, channels=3, normalize=(not args.nonormalize))

        frames_sampled = 0
        for frame_data in sampler:
            sample_frames, video_path, frame_nums = frame_data

            # Fetch the arm data for the latest of the frames (or the only frame if there is
            # only a single frame per sample)
            first_frame = int(frame_nums[0])
            last_frame = int(frame_nums[-1])
            current_data = arm_data.interpolate(video_timestamps[last_frame])
            # TODO Also fetch history? Option on the command line?
            # Fetch the next state after some end effector movement distance to be the prediction
            # target
            next_state, offset = getStateAtNextPosition(current_data, arm_data.future_records(),
                    args.prediction_distance, robot_model)

            if next_state is not None:
                # Find the frame number that corresponds to that next_state
                # If this sequence attempts to go through a 'mark' label, stop the next state at that
                # position instead of the one at the requested movement distance. This allows for a
                # clear separation between actions, ensuring that the robot will fully complete an
                # action by moving all the way into the desired position before moving on to the next
                # action.
                while (last_frame < len(video_timestamps) and
                        labels['mark'][last_frame] is not None and
                        video_timestamps[last_frame] < next_state['timestamp']):
                    last_frame += 1

                # If we exited because of the mark then we need to set a new next state
                if labels['mark'][last_frame] is not None:
                    next_state = arm_data.interpolate(video_timestamps[last_frame])

                # Verify that this data should be used
                # Check from the past frames to the frame at the target position in the future
                frame_range = list(range(first_frame, last_frame+1))
                any_discards = any([labels['behavior'][frame_num] == "discard" for frame_num in frame_range])

                if not any_discards:
                    # Combine the target labels and the current state into a single table for this
                    # sample.
                    sample_labels = {}
                    for key, value in next_state.items():
                        sample_labels["target_{}".format(key)] = value
                    for key, value in current_data.items():
                        sample_labels["current_{}".format(key)] = value
                    # Find the mark that we are progressing towards. Use 'none' if the current mark
                    # remains in effect through the end of the data. Use itertools.groupby to get
                    # the transitions for each mark.
                    # Each tuple in next_marks will be a pair of the mark itself and a list of all
                    # of that same kay, which can be used to determine how many frames that key
                    # remains the target.
                    #next_marks = list(itertools.groupby(marks[(int(frame_nums[-1])):]))
                    future_marks = marks[(int(frame_nums[-1])):]
                    cur_mark = future_marks[0]
                    marks_until_transition = 0
                    while marks_until_transition < len(future_marks) and future_marks[marks_until_transition] == cur_mark:
                        marks_until_transition += 1
                    if marks_until_transition < len(future_marks):
                        next_mark = future_marks[marks_until_transition]
                    else:
                        next_mark = None
                    # This is the mark of the initial state for this maneuver
                    sample_labels["initial_mark"] = cur_mark

                    # We cannot use this sample if there is no goal for the current motion or no
                    # existing state.
                    # TODO FIXME Only accept goals in our training list
                    if cur_mark is not None and next_mark is not None and cur_mark in args.goals:
                        # The goal mark is the one currently being moved towards
                        sample_labels["goal_mark"] = cur_mark
                        # Get the distance to the next mark
                        cur_pos = getGripperPosition(robot_model, arm_data.next_record())
                        mark_record = arm_data.future_records()[marks_until_transition]
                        mark_pos = getGripperPosition(robot_model, mark_record)
                        sample_labels["goal_distance"] = getDistance(cur_pos, mark_pos)

                        # Now write the sample labels and frames.
                        writeSample(datawriter, sample_labels, sample_frames, rosdir.replace('/', ''),
                            frame_nums)
                        # Keep the user updated on dataset progress
                        frames_sampled += 1
                        if 0 == frames_sampled % 500:
                            print("Accepted {} from samples {}".format(frames_sampled, vid_path))


    # Finished
    print("Data creation complete.")
    datawriter.close()
    return


if __name__ == '__main__':
    main()
