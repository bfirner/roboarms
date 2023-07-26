# Utilities for data handling, such as reading data from a rosbag, a video, or a yaml file with
# labels.

import csv
import ffmpeg
import io
import math
import os
import yaml

from nml_bag import Reader


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
    # Extract the desired records and store them by timestamp. The records look like
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


def readLabels(labels_path):
    """Read the labels from a yaml file at the given path.

    Arguments:
        labels_path (str): Path to a yaml file.
    Returns:
        labels (dict): Dictionary of the labels
    """
    # If the labels aren't there then return an empty table.
    if not os.path.exists(labels_path):
        return {}
    # TODO Should check to make sure that there aren't any errors during loading or parsing.
    labels = yaml.load(io.open(labels_path, "r", newline=None), Loader=yaml.SafeLoader)
    return labels


def writeLabels(labels_path, labels):
    """Write the labels into a yaml file at the given path.

    Arguments:
        labels_path (str): Path to a yaml file.
        labels     (dict): The labels
    """
    # TODO Should check to make sure that there aren't any errors during writing.
    # Open and truncate the file
    label_file = io.open(labels_path, "w", newline=None)
    label_file.write(yaml.dump(labels))
    label_file.close()


def readVideoTimestamps(csv_path):
    """Read video timestamps from a csv file.

    Arguments:
        csv_path (str): Path to the csv file.
    Returns:
        An array of timestamps at each frame, or None upon failure.
    """
    # TODO Technically this should be checked for failure.
    time_csv_file = io.open(csv_path, "r", newline='')
    time_csv = csv.reader(time_csv_file, delimiter=",")

    # Verify that this is a timestamp file of the expected format
    first_row = next(time_csv)
    expected_row = ['frame_number', 'time_sec', 'time_ns']
    if expected_row != first_row:
        print("First timestamp csv row should be {}, but found {}".format(expected_row, first_row))
        return None
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
    return video_timestamps
