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


def readArmRecords(bag_path, arm_topic):
    reader = Reader(filepath=bag_path, topics=[arm_topic])

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
    # TODO FIXME Extract the desired records and store them by timestamp. The records look like
    # this:
    # {'topic': '/arm2/joint_states', 'time_ns': 1689788069723452912, 'type': 'sensor_msgs/msg/JointState', 'header': OrderedDict([('stamp', OrderedDict([('sec', 1689788069), ('nanosec', 723357753)])), ('frame_id', '')]), 'name': ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate', 'gripper', 'left_finger', 'right_finger'], 'position': [0.015339808538556099, -1.7180585861206055, 1.7226604223251343, -1.7548741102218628, -0.023009711876511574, -0.5875146389007568, 0.013221289031207561, -0.013221289031207561], 'velocity': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'effort': [0.0, 0.0, 0.0, 5.380000114440918, -2.690000057220459, 0.0, 0.0, 0.0]}
    arm_records = []
    for record in reader:
        if record['topic'] == arm_topic:
            time_sec = int(record['header']['stamp']['sec'])
            time_ns = int(record['header']['stamp']['nanosec'])
            # TODO FIXME Correct the positions using the calibration values for this arm
            data = {
                'timestamp': time_sec * 10**9 + time_ns,
                'name': record['name'],
                'position': record['position'],
                'velocity': record['velocity'],
                'effort': record['effort'],
            }
            arm_records.append(data)
    print("Done!")
    return arm_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bag_path', type=str,
        help="The path to the bag directory, with sql db, video, and timestamps csv.")
    p.add_argument('--train_robot', default='arm2')
    p.add_argument('--video_scale', required=False, default=1.0, type=float,
        help="The default video scale during labelling.")
    p.add_argument('--frame_history', required=False, default=60.0, type=float,
        help="The number of frames that the user can easily go backwards.")
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

    # Open the rosbag db file and read the arm topic
    arm_topic = f"/{args.train_robot}/joint_states"
    arm_records = readArmRecords(args.bag_path, arm_topic)


    # Now go through video frames.
    width, height, total_frames = getVideoInfo(vid_path)
    print("Found {} frames in the video.".format(total_frames))

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

    cur_frame = -1
    # next_frame is a relative position to the cur_frame variable.
    # Initialize it to advance to frame 0.
    next_frame = 1
    cur_record = 0

    finished = False

    # Keep a limited buffer of past frames so that we can go backwards
    past_frame_buffer = []

    while cur_frame < total_frames and not finished:
        # TODO FIXME Actually look at the "next_frame" variable. Somehow support going backwards,
        # which will involve cacheing the past frames or figuring out how the ffmpeg-python bindings
        # support seeking.

        # Check if we should look into the past frame buffer and if that past frame is present.
        if 0 > next_frame:
            # Don't try to go back past what is in the history.
            next_frame = max(-len(past_frame_buffer)+1, next_frame)
            # Note that as long as we've gone through the loop at least once then there should be at
            # least one frame in the buffer.
            np_frame = numpy.copy(past_frame_buffer[next_frame])
        elif 0 < next_frame:
            # Check if the frame needs to advance.
            while 0 < next_frame:
                # Fetch the next frame
                in_bytes = input_process.stdout.read(int(width * height * channels * (args.video_scale**2)))
                #in_bytes = input_process.stdout.read(None)
                if in_bytes:
                    # Convert to numpy, and then to a display image
                    np_frame = numpy.frombuffer(in_bytes, numpy.uint8).reshape(height, width, channels)
                    #in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float).cuda()
                    # Adjust the frame position. The next_frame variable is relative to the cur_frame.
                    cur_frame += 1
                    next_frame -= 1

                    # Buffer a copy of this frame and maintain the maximum buffer size.
                    past_frame_buffer.append(numpy.copy(np_frame))
                    if (len(past_frame_buffer) > args.frame_history):
                        past_frame_buffer = past_frame_buffer[1:]

                else:
                    # We reached the end of the video before reaching the desired end frame somehow.
                    input_process.wait()
                    print("Reached the end of the video.")
                    return
        else:
            # Otherise refresh the frame buffer with the last fetched frame
            np_frame = numpy.copy(past_frame_buffer[-1])


        # Print some stuff for the UI:
        y = int(height * 0.8)
        x = int(width * 0.1)
        color = (1.0, 1.0, 1.0)
        thickness = 3.0
        frame_str = "Frame: {}".format(cur_frame)
        if 0 > next_frame:
            frame_str = "{} {}".format(frame_str, next_frame)
        frame_str = "{} / {}".format(frame_str, total_frames)
        cv2.putText(img=np_frame, text=frame_str, org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=6)

        # Display the frame.
        cv2.imshow("arm video", np_frame)

        # Get the current timestamp and then interpolate to find the position, velocity, and effort
        # values that correspond to it.
        cur_time = video_timestamps[cur_frame + next_frame]

        # arm_records has values for the position, velocity, and effort keys
        # Time is in the time field
        # Find the first record with a timestamp greater than the image time
        print("record is {}".format(arm_records[cur_record]))
        while (0 <= cur_record and arm_records[cur_record]['timestamp'] > cur_time):
            cur_record -= 1
        while (cur_record < len(arm_records) and arm_records[cur_record]['timestamp'] < cur_time):
            cur_record += 1
        if cur_record >= len(arm_records):
            print("Ran out of records to label this video.")
            return
        elif cur_record < 0:
            print("Video begins before arm records.")
            return
        else:
            # Interpolate this arm record and the previous record to get the sensor values that
            # correspond to the current frame.
            # TODO
            pass

        # Keep processing user input until they change the frame or want to exit the program.
        action = False
        while not finished and not action:
                inkey = cv2.waitKey(0)
                # Finish if the user inputs q
                finished = isKey(inkey, ord('q'))
                if isKey(inkey, arrow_right):
                    next_frame += 1
                    action = True
                elif isKey(inkey, arrow_left):
                    next_frame -= 1
                    action = True
                elif isKey(inkey, arrow_up):
                    next_frame += 30
                    action = True
                elif isKey(inkey, arrow_down):
                    next_frame -= 30
                    action = True
                # TODO Use the returned key value for labelling.
                # Save the per-frame segment labels into the file located at args.label_file
                # TODO Add function to save
                # TODO Add function to toggle labelling as "use" or "do not use"
                # TODO Add function to mark the end of a maneuver

    # Remove the window
    cv2.destroyAllWindows()
    
    # Return
    return


if __name__ == '__main__':
    main()
