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

from arm_utility import (getDistance, getGripperPosition)
# These are the robot descriptions to match function calls in the modern robotics package.
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from nml_bag import Reader
# For annotation drawing
from PIL import ImageDraw, ImageFont, ImageOps
from data_utility import (readArmRecords, readVideoTimestamps, readLabels, writeLabels)
from bee_analysis.utility.video_utility import getVideoInfo


def isKey(cvkey, key):
    # OpenCV's waitKey function returns not just the key, but also any keyboard modifiers. This
    # means that the returned value cannot be compared to just the key.
    return key == (cvkey & 0xFF)

arrow_left  = 81
arrow_up    = 82
arrow_right = 83
arrow_down  = 84

def findRecord(arm_records, start_idx, desired_timestamp):
    """Find the index of the first record that occurs after the desired timestamp."""
    # First go backwards to find the first record that occurs before the timestamp, then go forwards
    # to find the first record after the given timestamp.
    # This will work even if the provided start_idx is nowhere close to the correct position but
    # will be much faster if it is.
    cur_record = start_idx
    while (0 <= cur_record and arm_records[cur_record]['timestamp'] > desired_timestamp):
        cur_record -= 1
    while (cur_record < len(arm_records) and arm_records[cur_record]['timestamp'] < desired_timestamp):
        cur_record += 1
    if cur_record >= len(arm_records):
        print("Ran out of records to label this video.")
        return None
    elif cur_record < 0:
        print("Video begins before arm records.")
        return None
    return cur_record

def main():
    p = argparse.ArgumentParser()
    p.add_argument('bag_path', type=str,
        help="The path to the bag directory, with sql db, video, and timestamps csv.")
    p.add_argument('--train_robot', default='arm2')
    p.add_argument('--video_scale', required=False, default=1.0, type=float,
        help="The default video scale during labelling.")
    p.add_argument('--frame_history', required=False, default=90.0, type=float,
        help="The number of frames that the user can easily go backwards."
             "90 frames at 1920x1024 with 3 bytes per pixel is around 500MB.")
    # Path to the label file.
    p.add_argument('--label_file', default=None)
    p.add_argument(
        '--robot_model',
        type=str,
        required=False,
        default="px150",
        help='The name of the interbotix robot model.')

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

    # Load the robot model, which will be used in later distance calculations
    # Model descriptions are installed to a path like /root/interbotix_ws/install/interbotix_xs_modules/lib/python3.10/site-packages/interbotix_xs_modules/xs_robot/mr_descriptions.py
    model_generator = getattr(mrd, args.robot_model)
    robot_model = model_generator()

    ################
    # Data loading
    # Set up readers for the timestamps, video, and database
    video_timestamps = readVideoTimestamps(ts_path)

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
    
    ################
    # Labelling setup
    if args.label_file is None:
        args.label_file = os.path.join(args.bag_path, "labels.yaml")

    print(f"Labels will be saved to {args.label_file}")
    labels = readLabels(args.label_file)
    # Start with default values if the labels are empty.
    if {} == labels:
        # Segment behaviors
        labels['behavior'] = [None for i in range(total_frames)]
        # Maneuver begin and end marks
        labels['mark'] = [None for i in range(total_frames)]

    # Labelling state
    label_on = False
    cur_segment = None


    ################
    # UI Loop

    cur_frame = -1
    # next_frame is a relative position to the cur_frame variable.
    # Initialize it to advance to frame 0.
    next_frame = 1
    cur_record = 0

    finished = False

    # Keep a limited buffer of past frames so that we can go backwards
    past_frame_buffer = []

    # TODO The video could be loaded in chunks with .trim(start_frame=x, end_frame=y) rather than
    # using the history buffer. That may be slow though, so it should be tested.
    while cur_frame < total_frames and not finished:
        # Check if we should look into the past frame buffer and if that past frame is present.
        if 0 > next_frame:
            # Don't try to go back past what is in the history.
            next_frame = max(-len(past_frame_buffer)+1, next_frame)
            # Don't try to go before the first frame either
            if 0 > cur_frame + next_frame:
                next_frame = -cur_frame
            # Note that as long as we've gone through the loop at least once then there should be at
            # least one frame in the buffer.
            np_frame = numpy.copy(past_frame_buffer[next_frame])
            # Note that this behavior is slightly different from the forward behavior in that a
            # change in label name will always affect from the currently viewed frame up until the
            # current frame.
            if label_on and cur_segment is not None:
                # Label from cur_frame+next_frame:cur_frame with cur_segment
                # Don't do this, it will distribute the string elements across the labels
                # labels['behavior'][cur_frame+next_frame:cur_frame] = cur_segment
                for idx in range(cur_frame+next_frame, cur_frame+1):
                    labels['behavior'][idx] = cur_segment

        elif 0 < next_frame:
            # Check if the frame needs to advance.
            while 0 < next_frame:
                # Fetch the next frame
                in_bytes = input_process.stdout.read(int(width * height * channels * (args.video_scale**2)))
                #in_bytes = input_process.stdout.read(None)
                if in_bytes:
                    # Convert to numpy, and then to a display image
                    np_frame = numpy.frombuffer(in_bytes, numpy.uint8).reshape(height, width, channels)
                    # Swap RGB to BGR for proper display in OpenCV
                    np_frame = numpy.stack((np_frame[:,:,2], np_frame[:,:,1], np_frame[:,:,0]), 2)
                    #in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float).cuda()
                    # Adjust the frame position. The next_frame variable is relative to the cur_frame.
                    cur_frame += 1
                    next_frame -= 1

                    # If labelling, then propagate the current segment label
                    if label_on and cur_segment is not None:
                        labels['behavior'][cur_frame] = cur_segment

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
            # Otherwise refresh the frame buffer with the last fetched frame
            np_frame = numpy.copy(past_frame_buffer[-1])
            if label_on and cur_segment is not None:
                labels['behavior'][cur_frame+next_frame] = cur_segment


        # Print some stuff for the UI:
        y = int(height * 0.8)
        x = int(width * 0.1)
        color = (1.0, 0.3, 0.3)
        thickness = 3.0
        # Cur frame message
        frame_str = "Frame: {}".format(cur_frame)
        if 0 > next_frame:
            frame_str = "{} {}".format(frame_str, next_frame)
        frame_str = "{} / {}".format(frame_str, total_frames)
        cv2.putText(img=np_frame, text=frame_str, org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=6)
        # Current label message
        y = int(height * 0.9)
        label_str = "Labelling {}, cur segment: {}".format(label_on,
            labels['behavior'][cur_frame+next_frame])
        cv2.putText(img=np_frame, text=label_str, org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=6)
        # Mark message
        if labels['mark'][cur_frame+next_frame] is not None:
            y = int(height * 0.99)
            mark_str = "Current mark: {}".format(labels['mark'][cur_frame+next_frame])
            cv2.putText(img=np_frame, text=mark_str, org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=6)

        # Get the current timestamp
        # TODO interpolate to find the position, velocity, and effort values that correspond to it.
        cur_time = video_timestamps[cur_frame + next_frame]

        cur_record = findRecord(arm_records, cur_record, cur_time)
        # TODO cur_record is the first record after `cur_time`.
        # Data would be more correct if we interpolate this arm record and the previous record to
        # get the sensor values that correspond to the current frame.

        # We can't continue labelling if there is no more data.
        if cur_record is None:
            return

        # Search for a previous mark, find the distance to it from the current state, and display
        # the distance.
        # Find the first mark in list(reversed(labels['mark'][0:cur_frame+next_frame]))
        previous_mark = None
        mark_time = None

        for offset, mark in enumerate(reversed(labels['mark'][:cur_frame+next_frame])):
            if mark is not None:
                previous_mark = mark
                mark_time = video_timestamps[cur_frame+next_frame-offset]
                break

        if previous_mark is not None:
            cur_position = getGripperPosition(args.robot_model, arm_records[cur_record])
            mark_record = findRecord(arm_records, cur_record, mark_time)
            mark_position = getGripperPosition(args.robot_model, arm_records[mark_record])
            mark_distance = getDistance(mark_position, cur_position)
            x = int(width * 0.5)
            y = int(height * 0.99)
            label_str = "Distance from {} is {}m".format(previous_mark, round(mark_distance, 3))
            cv2.putText(img=np_frame, text=label_str, org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=6)
            #print("joint names {}".format(arm_records[cur_record]['name']))
            #print("m: {}".format(robot_model.M))
            #print("slist: {}".format(robot_model.Slist))
            #print("joint positions: {}".format(arm_records[cur_record]['position']))
            #print("mark position: {}".format(mark_position))
            #print("cur position: {}".format(cur_position))

        # Display the frame.
        cv2.imshow("arm video", np_frame)

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
            elif isKey(inkey, ord('l')):
                # Toggle labelling
                label_on = not label_on
                action = True
            elif isKey(inkey, ord('s')):
                writeLabels(args.label_file, labels)
                print("Labels saved to {}".format(args.label_file))
            elif isKey(inkey, ord('k')):
                cur_segment = "keep"
                print("Current label changed to {}".format(cur_segment))
                action = True
            elif isKey(inkey, ord('d')):
                cur_segment = "discard"
                print("Current label changed to {}".format(cur_segment))
                action = True
            elif isKey(inkey, ord('m')):
                # Mark an action
                mark = cv2.waitKey(0)
                mark_value = 0
                while 0 <= mark_value:
                    mark_digit = mark - ord('0')
                    if 0 <= mark_digit and mark_digit <= 9:
                        # Update the value for this mark
                        mark_value = 10 * mark_value + mark_digit
                        # Get the next key
                        mark = cv2.waitKey(0)
                    elif ord('\n') == mark or ord('\r') == mark:
                        labels['mark'][cur_frame+next_frame] = mark_value
                        print("Mark {} written for frame {}".format(mark_value, cur_frame+next_frame))
                        mark_value = -1
                    else:
                        print("Mark must be a number, not {}".format(mark))
                        mark_value = -1
                # Redraw the screen
                action = True
            elif isKey(inkey, ord('h')):
                print("s: Save labels")
                print("q: Quit (without saving)")
                print("l: Toggle labelling on or off")
                print("k: Mark segment as keep")
                print("d: Mark segment as drop")
                print("m[0-9]+: Mark action. Type a number and then the enter key.")
                # TODO Add an autoplay?

    # Remove the window
    cv2.destroyAllWindows()
    
    # Return
    return


if __name__ == '__main__':
    main()
