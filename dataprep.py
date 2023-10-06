#!/usr/bin/env python3

# Prepare data for DNN training by converting videos and rosbags into a webdataset.
# TODO In the long-term, webdatasets are not the most optimal storage format for larger amounts of
# data. Instead, a columnar format, such as parquet, may be more suitable.

import argparse
import csv
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

from arm_utility import (getDistance, getGripperPosition, getStateAtNextPosition)
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
        help="Scaling to apply to x and y dimensions before cropping. "
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
            current_frame = int(frame_nums[-1])
            if (labels['behavior'][current_frame] == 'keep' and
                video_timestamps[current_frame] <= arm_data.last_time()):
                # TODO FIXME Verify the timestamps of the records being used for interpolation
                current_data = arm_data.interpolate(video_timestamps[current_frame])
                # Fetch the next state after some end effector movement distance to be the prediction
                # target
                # TODO FIXME This uses the end effector distance, which can lead to some weird
                # behavior in the presence of tight curves. Would it be better to use total distance
                # moved instead? Would that require smoothing the input data to remove human jitter
                # as a preprocessing step?
                # The path distance should be used if we are attempting to reconstruct the path
                # exactly rather than moving to the target location.
                next_state, next_state_offset = getStateAtNextPosition(current_data, arm_data.future_records(),
                        args.prediction_distance, args.robot_model, use_path_distance=True)
            else:
                # If the video is longer than the ros records (which isn't an error, they could have
                # been stopped in any order) then there is no 'next state'.
                next_state = None

            if next_state is not None:
                # Find earliest frame number that corresponds to that next_state desired state.
                # If this sequence attempts to go through a 'mark' label, stop the next state at that
                # position instead of the one at the requested movement distance. This allows for a
                # clear separation between actions, ensuring that the robot will fully complete an
                # action by moving all the way into the desired position before moving on to the next
                # action.
                next_frame = current_frame + 1
                while (next_frame < len(video_timestamps) and
                        labels['mark'][next_frame] is None and
                        video_timestamps[next_frame] < next_state['timestamp']):
                    next_frame += 1

                # If we exited the previous loop because we encountered a new mark before reaching
                # the timestamp of the next state (determined by distance) then we need to set a new
                # next state at the frame before that mark. Walk forward from the current data to
                # the time of the transition
                if labels['mark'][next_frame] is not None:
                    while next_state_offset > 0 and arm_data.future_records()[next_state_offset]['timestamp'] > video_timestamps[next_frame]:
                        next_state_offset -= 1
                    next_state = arm_data.future_records()[next_state_offset]

                # Verify that this data should be used
                # Check from the past frames to the frame at the target position in the future
                frame_range = list(range(current_frame, next_frame+1))
                any_discards = any([labels['behavior'][frame_num] == "discard" for frame_num in frame_range])
                any_nones = any([labels['behavior'][frame_num] == None for frame_num in frame_range])

                #if not any_discards and not any_nones:
                if labels['behavior'][current_frame] == 'keep' and labels['behavior'][next_frame] == 'keep':
                    # Combine the target labels and the current state into a single table for this
                    # sample.
                    sample_labels = {}
                    for key, value in next_state.items():
                        sample_labels["target_{}".format(key)] = value
                        # Expand the full list of joint positions to the arm position and gripper
                        # position
                        # TODO This assumes a joint setup as in the Interbotix px150 where the first
                        # five joints are the arm
                        if key == 'position':
                            sample_labels["target_{}".format('arm_position')] = value[0:5]
                    for key, value in current_data.items():
                        sample_labels["current_{}".format(key)] = value
                        # Expand the full list of joint positions to the arm position and gripper
                        # position
                        # TODO This assumes a joint setup as in the Interbotix px150 where the first
                        # five joints are the arm
                        if key == 'position':
                            sample_labels["current_{}".format('arm_position')] = value[0:5]
                    # Find the mark that we are progressing towards. Use 'none' if the current mark
                    # remains in effect through the end of the data. Use itertools.groupby to get
                    # the transitions for each mark.
                    # Each tuple in next_marks will be a pair of the mark itself and a list of all
                    # of that same kay, which can be used to determine how many frames that key
                    # remains the target.
                    #next_marks = list(itertools.groupby(marks[(int(frame_nums[-1])):]))
                    future_marks = marks[(int(frame_nums[-1])):]
                    cur_mark = future_marks[0]
                    marks_until_transition = 1
                    while marks_until_transition < len(future_marks) and future_marks[marks_until_transition] == None:
                        marks_until_transition += 1
                    if marks_until_transition < len(future_marks):
                        next_mark = future_marks[marks_until_transition]
                    else:
                        next_mark = None
                    # This is the mark of the initial state for this maneuver
                    sample_labels["initial_mark"] = cur_mark

                    # We cannot use this sample if there is no goal for the current motion or no
                    # existing state.
                    # Only accept goals in our training list
                    if cur_mark is not None and next_mark is not None and cur_mark in args.goals:
                        # The goal mark is the one currently being moved towards
                        sample_labels["goal_mark"] = cur_mark
                        # Get the distance to the next mark
                        cur_pos = getGripperPosition(args.robot_model, current_data)
                        mark_record = arm_data.future_records()[marks_until_transition]
                        mark_pos = getGripperPosition(args.robot_model, mark_record)
                        sample_labels["goal_distance"] = getDistance(cur_pos, mark_pos)

                        # Go backwards to create several different status inputs for previous mark
                        # distances.
                        prev_1cm_record = arm_data.get_record_at_distance(0.01)
                        if prev_1cm_record is None:
                            # Use a default 10cm distance if the distance to the goal didn't exist
                            sample_labels["goal_distance_prev_1cm"] = 0.1
                        else:
                            prev_1cm_pos = getGripperPosition(args.robot_model, prev_1cm_record)
                            sample_labels["goal_distance_prev_1cm"] = getDistance(prev_1cm_pos, mark_pos)

                        prev_2cm_record = arm_data.get_record_at_distance(0.02)
                        if prev_2cm_record is None:
                            # Use a default 10cm distance if the distance to the goal didn't exist
                            sample_labels["goal_distance_prev_2cm"] = 0.1
                        else:
                            prev_2cm_pos = getGripperPosition(args.robot_model, prev_2cm_record)
                            sample_labels["goal_distance_prev_2cm"] = getDistance(prev_2cm_pos, mark_pos)

                        prev_3cm_record = arm_data.get_record_at_distance(0.03)
                        if prev_3cm_record is None:
                            # Use a default 10cm distance if the distance to the goal didn't exist
                            sample_labels["goal_distance_prev_3cm"] = 0.1
                        else:
                            prev_3cm_pos = getGripperPosition(args.robot_model, prev_3cm_record)
                            sample_labels["goal_distance_prev_3cm"] = getDistance(prev_3cm_pos, mark_pos)

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
