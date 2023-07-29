#!/usr/bin/env python3

# Prepare data for DNN training by converting videos and rosbags into a webdataset.
# TODO In the long-term, webdatasets are not the most optimal storage format for larger amounts of
# data. Instead, a columnar format, such as parquet, may be more suitable.

import argparse
import csv
import cv2
import ffmpeg
import io
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

from data_utility import (getVideoInfo, readArmRecords, readLabels, readVideoTimestamps,
    vidSamplingCommonCrop)
# These are the robot descriptions to match function calls in the modern robotics package.
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from nml_bag import Reader
# Helper function to convert to images
from torchvision import transforms

def writeSample(dataset, frames, video_source, sample_frames, other_data):
    """Write the given sample to a webdataset.

    Arguments:
        dataset       (TarWriter): Dataset to write into.
        frames     (list[tensor]): The frames.
        video_source        (str): The source of this data.
        sample_frames (list[int]): The frame numbers.
        other_data ({str: float}): Other training data.
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
    base_name = '_'.join([video_source] + [str(fnum) for fnum in sample_frames])
    sample = {
        "__key__": base_name,
    }
    # Write the other data into the dataset
    for key, value in other_data.items():
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
         getattr(interbotix_xs_modules.xs_robot.mr_descriptions.ModernRoboticsDescription, 'px150')

    Arguments:
        arm_record ({str: data}): Ros message data for the robot arm.
    Returns:
        x,y,z tuple of the gripper location.
    """
    # TODO Since a mini-goal of this project is to handle actuation without calibration we won't be
    # using this for labels (because it would require calibration for the x,y,z location of the
    # gripper to be meaningful), but will be using this to determine the moved distance of the
    # gripper, and from that we will determine the next pose to predict.
    joint_positions = [
        arm_record['position']['name'] for name in arm_record['name']
    ]
    # 'M' is the home configuration of the robot, Slist has the joint screw axes at the home
    # position
    return modern_robotics.FKinSpace(robot_model.M, robot_model.Slist, joint_positions)
    pass


# Create a memoizing version of arm interpolation function. Since it is likely that calls will be
# made in time order we should always begin searching records close the to index of the last search.
class ArmDataInterpolator:
    def __init__(self, arm_records):
        """
        Arguments:
            arm_records ({str: data}): Ros message data for the robot arm.
        """
        self.last_idx = 0
        self.records = arm_records

    def interpolate(self, timestamp):
        """Interpolate arm data to match the given video timestamp.

        Arguments:
            timestamp           (int): The ros timestamp (in ns)
        Returns:
            A table (str: float) of the data at this timestamp.
        """
        # Extract the position, velocity, and effort values for each of the joints.
        # The data looks like this:
        # data = {
        #     'timestamp': time_sec * 10**9 + time_ns,
        #     'name': record['name'],
        #     'position': record['position'],
        #     'velocity': record['velocity'],
        #     'effort': record['effort'],
        # }

        # First find the index before this event
        # Go forwards if necessary to find the first index after the event
        while (self.last_idx < len(self.records) and
            self.records[self.last_idx]['timestamp'] < timestamp):
            self.last_idx += 1
        if self.last_idx >= len(self.records):
            raise Exception("Requested timestamp {} is beyond the data range.".format(timestamp))
        # Go backwards if necessary to find the first index before this event
        while 0 < self.last_idx and self.records[self.last_idx]['timestamp'] > timestamp:
            self.last_idx -= 1
        if self.last_idx < 0:
            raise Exception("Requested timestamp {} comes before the data range.".format(timestamp))

        # This index is the state before the given timestamp
        before_state = self.records[self.last_idx]

        # Go forward one index
        self.last_idx += 1
        if self.last_idx >= len(self.records):
            raise Exception("Requested timestamp {} is beyond the data range.".format(timestamp))

        # This index is the state after the given timestamp
        after_state = self.records[self.last_idx]

        # Interpolation details
        before_time = before_state['timestamp']
        after_time = after_state['timestamp']
        delta = (timestamp - before_time) / (after_time - before_time)

        # Assemble a new record
        new_record = {
            'timestamp': timestamp,
            'name': before_state['name']
        }

        # Interpolate data from each of the robot records
        for dataname in ['position', 'velocity', 'effort']:
            new_record['position'] = [data[0] + (data[1]-data[0])*delta for data in
                zip(before_state['position'], after_state['position'])]

        # Return the assembled record
        return new_record


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
        default='arm2',
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
        default=0.5,
        help='Distance of the future position of the robot gripper in the dataset.')

    args = parser.parse_args()

    # TODO Everything

    # Create a writer for the WebDataset
    datawriter = wds.TarWriter(args.outpath, encoder=False)

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

        if 0 == len(db_paths):
            print("No database found in bag path {}".format(rosdir))
            return
        if 1 < len(db_paths):
            print("Too many (expecing 1) db files found in {}".format(rosdir))
            return
        db_path = db_paths[0]

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

        # Loop through the video frames, exporting as requested into the webdataset
        out_width, out_height, out_crop_x, out_crop_y = vidSamplingCommonCrop(
            height, width, args.height, args.width, args.video_scale, args.crop_x_offset, args.crop_y_offset)
        print("{}, {}, {}, {}".format(out_width, out_height, out_crop_x, out_crop_y))
        pix_fmt='rgb24'
        channels = 3
        filter_width = out_width + 2 * args.crop_noise
        filter_height = out_height + 2 * args.crop_noise
        # TODO FIXME debugging
        filter_width = int(args.video_scale * width)
        filter_height = int(args.video_scale * height)
        video_process = (
            ffmpeg
            .input(vid_path)
            # Scale
            .filter('scale', args.video_scale*width, -1)
            # The crop is automatically centered if the x and y parameters are not used.
            # Don't crop to out_width and out_height so that we can have crop jitter
            .filter('crop', out_w=filter_width, out_h=filter_height, x=out_crop_x, y=out_crop_y)
            # TODO Starting off without normalization
            # Normalize color channels.
            #.filter('normalize', independence=0.0)
            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
            .run_async(pipe_stdout=True, quiet=True)
        )
        # Generate frames and put them into the dataset
        # Keep track of the frame number so we know which frames to sample
        frame = 0
        # Read in frames from this video.
        in_bytes = True
        while in_bytes:
            # Get ready to fetch the next frame
            partial_sample = []
            sample_frames = []
            # Use the same crop location for each sample in multiframe sequences.
            rand_crop_x = random.choice(range(0, 2 * args.crop_noise + 1))
            rand_crop_y = random.choice(range(0, 2 * args.crop_noise + 1))
            while in_bytes and len(partial_sample) < args.frames_per_sample:
                in_bytes = video_process.stdout.read(filter_width * filter_height * channels)
                if in_bytes:
                    # Check if this frame will be sampled.
                    sample_frame = False
                    sample_in_progress = 0 < len(partial_sample)
                    # If not already sampling a frame, then use the sample probability to see if
                    # this frame should be sampled.
                    if not sample_in_progress:
                        sample_frame = (random.random() < args.sample_prob)
                        source_frame = frame
                    # If we are sampling a muiltiframe input then see if this frame is at the
                    # current interval
                    else:
                        sample_frame = (frame - source_frame) % (args.interval + 1) == 0

                    if sample_frame:
                        # Convert to numpy, and then to torch.
                        np_frame = numpy.frombuffer(in_bytes, numpy.uint8)
                        in_frame = torch.tensor(data=np_frame, dtype=torch.uint8,
                            ).reshape([1, filter_height, filter_width, channels])
                        # Apply the random crop
                        in_frame = in_frame[:, rand_crop_y:rand_crop_y+out_height,
                                rand_crop_x:rand_crop_x+out_width, :]
                        in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float)
                        partial_sample.append(in_frame)
                        sample_frames.append(frame)
                    frame += 1
                else:
                    # Reached the end of the video
                    video_process.wait()
                # If multiple frames are being returned then concat them along the channel
                # dimension. Otherwise just return the single frame.
                # First verify that we collect a full frame
                if len(partial_sample) == args.frames_per_sample:
                    # TODO Fetch the arm data for these frames
                    # TODO Also fetch history?
                    # TODO Fetch the next state after some movement distance to be the prediction target
                    # TODO That timestamp should be an option on the command line
                    other_data = arm_data.interpolate(video_timestamps[sample_frames[-1]])
                    writeSample(datawriter, partial_sample, rosdir.replace('/', ''), sample_frames, other_data)


    # Finished
    print("Data creation complete.")
    datawriter.close()
    return


if __name__ == '__main__':
    main()
