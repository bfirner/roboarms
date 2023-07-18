#!/usr/bin/env python3

"""
The default ros2bag behavior is to store images from the /image_raw stream as individual
blobs in a sqlite3 table. This is beyond slow. This script will subscribe to the image stream and
convert it into an mp4 video instead. With each frame encoded an entry will be placed into a csv
file that records the timestamp for that video frame.
"""

import argparse
import csv
import datetime
import ffmpeg
import io
import os
import rclpy
import signal
import sys

from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


# Terminate data capture when the user interrupts the process
terminate = False
def handler(signum, frame):
    terminate = True


def get_topic_list():
    # Create a dummy node so that we can use get_topic_names_and_types
    node_dummy = Node("topic_list_fetcher")
    topic_list = node_dummy.get_topic_names_and_types()
    # Destroy the dummy node before returning the available topics.
    node_dummy.destroy_node()
    return topic_list


class VideoInfoFetcher(Node):
    """Get video information."""

    def __init__(self):
        super().__init__('video_info_fetcher')

        self.video_init = False

        self.info_subscription = self.create_subscription(
            msg_type=CameraInfo,
            topic="/camera_info",
            callback=self.video_info_callback,
            qos_profile=10,
        )

    def video_info_callback(self, msg):
        """Initialize video information."""
        self.height = int(msg.height)
        self.width = int(msg.width)
        self.video_init = True


class ImageEncoder(Node):
    """Encode the given image messages into a video."""

    def __init__(self, topic_string, video_height, video_width, scale=1.0,
            crop_width=None, crop_height=None, crop_x=0, crop_y=0):
        super().__init__('image_encoder')

        self.scale = scale
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_x = crop_x
        self.crop_y = crop_y

        self.height = video_height
        self.width = video_width

        # Create the csv file to store frame timestamps and the ffmpeg stream for video writing.
        self.initialize_outputs()

        self.vid_subscription = self.create_subscription(
            msg_type=Image,
            topic=topic_string,
            callback=self.handle_frame,
            qos_profile=1,
        )

    def __del__(self):
        print("Cleaning up.")
        # Close the video writing process
        self.output_process.stdin.close()
        self.output_process.wait()
        # Close the csv file
        self.csv_file.close()
 

    def handle_frame(self, msg):
        # Write the frame
        self.output_process.stdin.write(
            bytes(msg.data)
        )
        self.timestamps.writerow([self.frame, msg.header.stamp._sec, msg.header.stamp._nanosec])
        self.frame += 1

    def initialize_outputs(self):
        """Initialize video information."""

        if self.crop_width is None:
            self.crop_width = self.width

        if self.crop_height is None:
            self.crop_height = self.height

        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_video = "robo_video_{}.mp4".format(date_str)
        out_timestamp = "robo_video_{}.csv".format(date_str)
        print("Writing data to robo_video_{}.mp4/csv".format(date_str))

        # Begin an output process from ffmpeg.
        self.output_process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.width}x{self.height}')
            # Scale
            .filter('scale', self.scale*self.width, -1)
            # The crop is automatically centered if the x and y parameters are not used.
            .filter('crop', out_w=self.crop_width, out_h=self.crop_height, x=self.crop_x, y=self.crop_y)
            .output(out_video, pix_fmt='rgb24')
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        # TODO Technically this should be checked for failure.
        self.csv_file = io.open(out_timestamp, "w")
        self.timestamps = csv.writer(self.csv_file, delimiter=",")
        self.timestamps.writerow(["frame_number", "time_sec", "time_ns"])

        self.frame = 0


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Save a video from /image_raw messages.")
    parser.add_argument(
        '--datalist',
        type=str,
        help=('A csv file with one or more video files and their labels.'
              ' Columns should be "file," "label," "begin," and "end".'
              ' The "begin" and "end" columns are in frames.'
              ' If the "begin" and "end" columns are not present,'
              ' then the label will be applied to the entire video.'))
    parser.add_argument(
        '--scale',
        type=float,
        required=False,
        default=1.0,
        help='Scaling to apply to each dimension (before cropping). A value of 0.5 will yield 0.25 resolution.')
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
        '--crop_width',
        type=int,
        required=False,
        help='The width of the output image.')
    parser.add_argument(
        '--crop_height',
        type=int,
        required=False,
        help='The height of the output image.')
    parser.add_argument(
        '--image_topic',
        type=str,
        required=False,
        default="/image_raw",
        help='The name of the image topic.')

    args = parser.parse_args()

    rclpy.init()

    # TODO Double check that the desired topics are available.
    topic_list = get_topic_list()
    #assert("/camera_info" in topic_list)
    #assert(args.image_topic in topic_list)

    # Get video info and then destroy this node.
    info = VideoInfoFetcher()
    while not info.video_init:
        rclpy.spin_once(info)
    info.destroy_node()

    print("Reading video with size {}x{}".format(info.width, info.height))

    # Make the image encoder
    encoder = ImageEncoder(args.image_topic, info.height, info.width)

    # Wait for messages until an interrupt is received
    while not terminate:
        rclpy.spin_once(encoder)

    # Clean up
    encoder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

