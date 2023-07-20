#!/usr/bin/env python3

# Read some data from a rosbag
# Recording was probably done with something like this:
# > ros2 bag record /arm1/joint_states /arm2/joint_states /image_raw/compressed /arm1/robot_description /arm2/robot_description /camera_info

# arm1 is the manipulator and arm2 is the pupper. Images should be suitable for DNN training.

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

from nml_bag import Reader
# For annotation drawing
from PIL import ImageDraw, ImageFont, ImageOps


def isKey(cvkey, key):
    # OpenCV's waitKey function returns not just the key, but also any keyboard modifiers. This
    # means that the returned value cannot be compared to just the key.
    return key == (cvkey & 0xFF)

arrow_left  = 81
arrow_up    = 82
arrow_right = 83
arrow_down  = 84


def getVideoInfo(video_path):
    """
    Get the total frames and size of a video.

    Arguments:
        video_path (str): The path to the video file.
    Returns:
        int: Width
        int: Height
        int: The total number of frames.
    """
    # Following advice from https://kkroening.github.io/ffmpeg-python/index.html
    # First find the size, then set up a stream.
    probe = ffmpeg.probe(video_path)['streams'][0]
    width = probe['width']
    height = probe['height']

    if 'duration' in probe:
        numer, denom = probe['avg_frame_rate'].split('/')
        frame_rate = float(numer) / float(denom)
        duration = float(probe['duration'])
        total_frames = math.floor(duration * frame_rate)
    else:
        # If the duration is not in the probe then we will need to read through the entire video
        # to get the number of frames.
        # It is possible that the "quiet" option to the python ffmpeg library may have a buffer
        # size problem as the output does not go to /dev/null to be discarded. The workaround
        # would be to manually poll the buffer.
        process1 = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='gray')
            #.output('pipe:', format='rawvideo', pix_fmt='yuv420p')
            .run_async(pipe_stdout=True, quiet=True)
        )
        # Count frames
        frame = 0
        while True:
            # Using pix_fmt='gray' we should get a single channel of 8 bits per pixel
            in_bytes = process1.stdout.read(width * height)
            if in_bytes:
                frame += 1
            else:
                process1.wait()
                break
        total_frames = frame
    return width, height, total_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bag_path', type=str,
        help="The path to the bag directory, with sql db, video, and timestamps csv.")
    p.add_argument('--train_robot', default='arm2')
    p.add_argument('--video_scale', required=False, default=1.0, type=float,
        help="The default video scale during labelling.")
    # Path to the label file.
    p.add_argument('--label_file', default=None)

    args = p.parse_args()

    # Check for required paths
    bagpath = pathlib.Path(args.bag_path)

    # There should be a single file match for each of these paths.
    db_paths = list(bagpath.glob("rosbag2*.db3"))
    ts_paths = list(bagpath.glob("robo_video_*.csv"))
    vid_paths = list(bagpath.glob("robo_video_*.mp4"))

    if 0 == len(db_paths):
        print("No database found in bag path {}".format(args.bag_path))
        return
    if 1 < len(db_paths):
        print("Too many (expecing 1) db files found in {}".format(args.bag_path))
        return
    db_path = db_paths[0]

    if 0 == len(ts_paths):
        print("No video timestamp found in bag path {}".format(args.bag_path))
        return
    if 1 < len(ts_paths):
        print("Too many (expecing 1) video timestamp files found in {}".format(args.bag_path))
        return
    ts_path = ts_paths[0]

    if 0 == len(vid_paths):
        print("No database found in bag path {}".format(args.bag_path))
        return
    if 1 < len(vid_paths):
        print("Too many (expecing 1) db files found in {}".format(args.bag_path))
        return
    vid_path = vid_paths[0]

    if args.label_file is None:
        args.label_file = os.path.join(args.bag_path, "labels.csv")

    print(f"Labels will be saved to {args.label_file}")


    # Set up readers for the timestamps, video, and database

    # TODO Technically this should be checked for failure.
    time_csv_file = io.open(ts_path, "r", newline='')
    time_csv = csv.reader(time_csv_file, delimiter=",")

    # Verify that this is a timestamp file of the expected format
    first_row = next(time_csv)
    expected_row = ['frame_number', 'time_sec', 'time_ns']
    if expected_row != first_row:
        print("First timestamp csv row should be {}, but found {}".format(expected_row, first_row))
        return
    # Read all of the rows into an array
    video_timestamps = []
    for row in time_csv:
        frame_number, time_sec, time_ns = row
        frame_number = int(frame_number)
        time_sec = int(time_sec)
        time_ns = int(time_ns)

        # Keep a sanity check on the frame numbers
        assert frame_number == len(video_timestamps)

        # Combine the two time components into a big number
        video_timestamps.append(time_sec * 10**9 + time_ns)
    time_csv_file.close()

    # Open the rosbag db file.
    arm_topic = f"/{args.train_robot}/joint_states"
    camera_info = "/camera_info"
    reader = Reader(filepath=args.bag_path, topics=[arm_topic])

    # Print the topics.
    print(f'The bag contains the following topics:')
    print(reader.topics)
    
    # Print the mapping between topics and message types.
    print(f'The message types associated with each topic are as follows:')
    print(reader.type_map)


    # The entire ros2bag can be fetched with reader.records(), which preloads and caches the result.
    # This is the preferred method of labelling, so long as we can guarantee that enough memory will
    # always be available.
    print("Loading rosbag records.")
    arm_records = list(reader)
    print("Done!")

    # There is only one topic in the records (the joint states messages) alreadyh in time order.


    # Now go through video frames.
    # Prepare a font for label UI. Just using the default font for now.
    try:
        font = ImageFont.truetype(font="DejaVuSans.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()

    width, height, total_frames = getVideoInfo(vid_path)

    # Begin the video input process from ffmpeg.
    # Read as 3 channel rgb24 images
    channels = 3
    input_process = (
        ffmpeg
        .input(vid_path)
        # Scale if requested
        #.filter('scale', args.video_scale, -1)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, quiet=True)
    )

    cur_frame = 0
    cur_record = 0

    finished = False

    while cur_frame < total_frames and not finished:
        # Fetch the next frame
        print(f"Reading frame {cur_frame}")
        in_bytes = input_process.stdout.read(int(width * height * channels * (args.video_scale**2)))
        #in_bytes = input_process.stdout.read(None)
        if in_bytes:
            # Convert to numpy, and then to a display image
            np_frame = numpy.frombuffer(in_bytes, numpy.uint8).reshape(height, width, channels)
            #in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float).cuda()


            # Print some stuff for the UI:
            y = int(height * 0.8)
            x = int(width * 0.1)
            color = (1.0, 1.0, 1.0)
            thickness = 3.0
            cv2.putText(img=np_frame, text="Frame: {}".format(cur_frame), org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=8)

            cv2.imshow("arm video", np_frame)
        else:
            # We reached the end of the video before reaching the desired end frame somehow.
            input_process.wait()
            print("Reached the end of the video.")
            return

        # Get the current timestamp and then interpolate to find the position, velocity, and effort
        # values that correspond to it.
        cur_time = video_timestamps[cur_frame]

        # arm_records has values for the position, velocity, and effort keys
        # Time is in the time field
        # Find the first record with a timestamp greater than the image time
        print("record is {}".format(arm_records[cur_record]))
        while (cur_record < len(arm_records) and arm_records[cur_record]['timestamp'] < cur_time):
            cur_record += 1
        if cur_record >= len(arm_records):
            print("Ran out of records to label this video.")
            return

    # Remove the window
    cv2.destroyAllWindows()
    
    # Return
    return


if __name__ == '__main__':
    main()
