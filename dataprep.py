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
import numpy
import os
import pathlib
import sys
import torch
import webdataset as wds
import yaml

from nml_bag import Reader


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
        img = transforms.ToPILImage()(frame/255.0).convert('RGB')
        # Now save the image as a png into a buffer in memory
        buf = io.BytesIO()
        img.save(fp=buf, format="png")
        buffers.append(buf)
    # TODO FIXME Need to fetch the robot state for this
    sample = {
        "__key__": '_'.join((base_name, '_'.join(frame_num))),
        "cls": row[class_col].encode('utf-8'),
        "metadata.txt": metadata.encode('utf-8')
    }
    # Write the other data into the dataset
    for key, value in other_data:
        sample[key] = value.encode('utf-8')
    for i in range(frame_count):
        sample[f"{i}.png"] = buffers[i].getbuffer()
    datawriter.write(sample)



def interpolateArmData(arm_records, timestamp):
    """Interpolate arm data to match the given video timestamp.

    Arguments:
        arm_records ({str: data}): Ros message data for the robot arm.
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
        type=int,
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

    args = parser.parse_args()

    # TODO Everything

    # Create a writer for the WebDataset
    datawriter = wds.TarWriter(args.outpath, encoder=False)

    # Loop over each rosbag directory
    # TODO Split this part into threads
    for rosdir in args.bag_paths:
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
        arm_records = readArmRecords(rosdir)

        # Now go through video frames.
        width, height, total_frames = getVideoInfo(vid_path)
        print("Found {} frames in the video {}.".format(total_frames, vid_path))

        # Get the labels
        if args.label_file is None:
            args.label_file = "labels.yaml"
        label_path = os.path.join(rosdir, args.label_file)
        labels = readLabels(label_path)

        # Loop through the video frames, exporting as requested into the webdataset
        out_width, out_height, out_crop_x, out_crop_y = vidSamplingCommonCrop(
            height, width, args.height, args.width, args.scale, args.crop_x_offset, args.crop_y_offset)
        pix_fmt='rgb24'
        channels = 3
        video_process = (
            ffmpeg
            .input(vid_path)
            # Scale
            .filter('scale', args.scale*width, -1)
            # The crop is automatically centered if the x and y parameters are not used.
            # Don't crop to out_width and out_height so that we can have crop jitter
            .filter('crop', out_w=width, out_h=height, x=out_crop_x, y=out_crop_y)
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
            crop_x = random.choice(range(0, 2 * self.crop_noise + 1))
            crop_y = random.choice(range(0, 2 * self.crop_noise + 1))
            while in_bytes and len(partial_sample) < self.frames_per_sample:
                in_bytes = video_process.stdout.read(width * height * channels)
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
                        sample_frame = (frame - source_frame) % (self.interval + 1) == 0


                    if sample_frame:
                        # Convert to numpy, and then to torch.
                        np_frame = numpy.frombuffer(in_bytes, numpy.uint8)
                        in_frame = torch.tensor(data=np_frame, dtype=torch.uint8,
                            ).reshape([1, in_height, in_width, channels])
                        # Apply the random crop
                        in_frame = in_frame[:, out_crop_y:out_crop_y+self.out_height,
                                out_crop_x:out_crop_x+out_width, :]
                        in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float)
                        partial_sample.append(in_frame)
                        sample_frames.append(str(frame))
                    frame += 1
                else:
                    # Reached the end of the video
                    video_process.wait()
                # If multiple frames are being returned then concat them along the channel
                # dimension. Otherwise just return the single frame.
                # First verify that we collect a full frame
                if len(partial_sample) == self.frame_per_sample:
                    # TODO Fetch the arm data for these frames
                    # TODO Also fetch history?
                    # TODO Fetch the next state after some timestamp
                    # TODO That timestamp should be an option on the command line
                    other_data = interpolateArmData(arm_records, video_timestamps[sample_frames[-1]])
                    writeSample(datawriter, partial_sample, rosdir, sample_frames, other_data)


    # Finished
    datawriter.close()
    return


if __name__ == '__main__':
    main()
