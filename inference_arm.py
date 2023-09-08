#!/usr/bin/env python3

# This script runs a trained neural network to control a robot arm..
# Copyright 2023 Bernhard Firner


import argparse
import ffmpeg
import numpy
import pathlib
import rclpy
import sys
import time
import torch

from queue import Queue
from threading import Event, Lock, Thread

# Ros includes
from rclpy.node import Node
from sensor_msgs.msg import (Image, JointState)

# Includes from this project
from arm_utility import (ArmReplay, getCalibrationDiff)
from data_utility import vidSamplingCommonCrop
from bee_analysis.utility.model_utility import (createModel, restoreModel)


# Terminate inference when the user kills the program
# TODO Use an event to terminate instead of a variable
exit_event = Event()
def handler(signum, frame):
    exit_event.set()


class RosHandler(Node):
    """Subscribe to camera images."""

    def __init__(self, data_lock, arm_topic_string, video_topic_string, scale=1.0,
            crop_width=None, crop_height=None, crop_x=0, crop_y=0, frames_per_sample=1,
            channels_per_frame=3, arm_name="arm2", desired_joints=[]):
        super().__init__('image_handler')

        self.data_lock = data_lock

        # There is no data available until the relevant messages are received
        self.frame = None
        self.frame_nsec = None
        self.position = None
        self.joint_names = None

        self.desired_joints = desired_joints

        self.scale = scale
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.crop_x = crop_x
        self.crop_y = crop_y

        # Read these from the message the first time the handle_frame function is called.
        self.height = None
        self.width = None

        self.vid_subscription = self.create_subscription(
            msg_type=Image,
            topic=video_topic_string,
            callback=self.handle_frame,
            qos_profile=1,
        )

        self.joint_subscription = self.create_subscription(
            msg_type=JointState,
            topic=arm_topic_string,
            callback=self.handle_joints,
            qos_profile=1,
        )

        self.channels_per_frame = channels_per_frame

        self.frames_per_sample = frames_per_sample
        if frames_per_sample > 1:
            self.frame_buffer = []


    def handle_frame(self, msg):
        # Set the height and width if they haven't been set. Also set the crop parameters and the
        # ffmpeg streams.
        if self.height is None:
            self.in_height = msg.height
            self.in_width = msg.width
            self.crop_width, self.crop_height, self.post_scale_crop_x, self.post_scale_crop_y = vidSamplingCommonCrop(self.in_height, self.in_width, self.crop_height, self.crop_width, self.scale, self.crop_x, self.crop_y)
            self.input_stream = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(self.in_width,
                    self.in_height))
                # Scale
                .filter('scale', self.scale*self.in_width, -1)
                # The crop is automatically centered if the x and y parameters are not used.
                .filter('crop', out_w=self.crop_width, out_h=self.crop_height,
                    x=self.post_scale_crop_x, y=self.post_scale_crop_y)
            )
            if self.channels_per_frame == 1:
                self.input_stream = (self.input_stream
                    # Output to another pipe so that the data can be read back in
                    .output('pipe:', format='rawvideo', pix_fmt='rgb8')
                    .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
                )
            else:
                self.input_stream = (self.input_stream
                    # Output to another pipe so that the data can be read back in
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
                )

        # Process the frame
        self.input_stream.stdin.write(numpy.uint8(msg.data).tobytes())
        crop_bytes = self.input_stream.stdout.read(self.crop_width * self.crop_height * self.channels_per_frame)
        # Reshape the frame and put the channels before the other dimensions
        np_frame = numpy.frombuffer(crop_bytes, numpy.uint8).reshape([1, self.crop_height, self.crop_width,
            self.channels_per_frame])
        # TODO The exact frame preprocessing must match what was done in dataprep, but currently the
        # image formats are different (e.g. in dataprep images are saved as pngs whereas here they
        # are coming directly from the video)
        out_img = torch.tensor(data=np_frame, dtype=torch.uint8).permute(0, 3, 1, 2) / 255.0

        # TODO Check frames_per_sample.
        # TODO Check channel_per_sample


        with self.data_lock:
            # Store this frame for handling by the consumer
            self.frame_nsec = msg.header.stamp._sec * 10**9 + msg.header.stamp._nanosec
            # TODO This should probably be a blit operation into a pinned GPU tensor to save memory
            self.frame = out_img

    def handle_joints(self, msg: JointState) -> None:
        if self.joint_names is None:
            self.joint_names = msg.name

        with self.data_lock:
            # TODO FIXME Whoops, looks like training used all of the joints
            #self.position = [msg.position[self.joint_names.index(name)] for name in self.desired_joints]
            #self.velocity = [msg.velocity[self.joint_names.index(name)] for name in self.desired_joints]
            #self.effort = [msg.effort[self.joint_names.index(name)] for name in self.desired_joints]
            self.position = msg.position
            self.velocity = msg.velocity
            self.effort = msg.effort


def dnn_inference_thread(robot_joint_names, position_queue, model_checkpoint, dnn_args, dnn_outputs,
        vector_inputs, goal_sequence, video_args, update_delay_s, exit_event):
    """Puts positions into the position queue.

    Arguments:
        robot_joint_names (list[str]): Joint names in the robot arm.
        position_queue    (Queue): Input queue to the robot
        dnn_args           (dict): The dnn arguments
        dnn_outputs       (list[str]): Names of the DNN outputs
        vector_inputs     (list[str]): Names of the DNN vector inputs
        goal_sequence     (list[int]): Sequence of goals
        video_args             (dict): Video arguments
        update_delay_s    (float): Update delay in seconds
        exit_event        (Event): Exit when is_set is True.
    """
    print("DNN inferencing thread started.")

    # Create the model and load the weights from the given checkpoint.
    net = createModel(**dnn_args)
    restoreModel(model_checkpoint, net)
    net = net.eval().cuda()

    # Initialize ROS2 nodes
    #rclpy.init()

    data_read_lock = Lock()

    # Make the image handler
    image_processor = RosHandler(data_read_lock, desired_joints=robot_joint_names, **video_args)
    data_spin_thread = Thread(target=rclpy.spin, args=(image_processor,))
    data_spin_thread.start()


    # Initialize the vector inputs tensor
    goal_idx = 0
    vector_input_locations = {}
    out_idx = 0
    for input_name in vector_inputs:
        # Vector inputs are size 1 unless they are the current robot position
        if input_name == 'current_position':
            # TODO FIXME Whoops, looks like training used all of the joints
            vector_input_locations[input_name] = slice(out_idx, out_idx + len(robot_joint_names) + 3)
            out_idx += len(robot_joint_names) + 3
        elif input_name == 'initial_mark':
            vector_input_locations[input_name] = out_idx
            out_idx += 1
        else:
            Exception("Unknown vector input: {}".format(input_name))

    # Create tensor inputs
    vector_inputs = torch.zeros([1, out_idx]).float().cuda()

    # Mark the locations of the model outputs
    output_locations = {}
    out_idx = 0
    for output_name in dnn_outputs:
        if output_name == 'target_position':
            output_locations[output_name] = slice(out_idx, out_idx+len(robot_joint_names))
            out_idx += len(robot_joint_names)
        else:
            output_locations[output_name] = out_idx
            out_idx +=1

    # Wait until there is some data available
    ready = False
    while not exit_event.is_set() and not ready:
        with data_read_lock:
            ready = image_processor.frame is not None and image_processor.position is not None

    # Wait for messages until an interrupt is received
    while not exit_event.is_set():
        # Forward the frame through the DNN model
        # This always uses the most recent frame, so if the update rate is much faster than the
        # frame rate the only thing that will update between commands is the robot's current
        # position.
        with data_read_lock:
            new_frame = image_processor.frame.float().cuda()
            joint_positions = image_processor.position

        # Set up vector inputs
        out_idx = 0
        for input_name in vector_inputs:
            # Vector inputs are size 1 unless they are the current robot position
            vector_input_locations[input_name] = out_idx
            if input_name == 'current_position':
                vector_inputs[0, out_idx:out_idx+len(robot_joint_names)].copy_(torch.tensor(joint_positions))
                out_idx += len(robot_joint_names)
            elif input_name == 'initial_mark':
                vector_inputs[0, out_idx] = goal_sequence[goal_idx]
                out_idx += 1

        # goal_mark determines the state to feed back to the network via the vector inputs
        # The initial_mark status input (if present) and
        # goal_distance is used to determine when to switch goals.
        # The target_position output contains the next set of joint positions.

        net_out = net.forward(new_frame, vector_inputs)
        predicted_distance = net_out[0, output_locations['goal_distance']].item()
        next_position = net_out[0, output_locations['target_position']].tolist()
        print("Requesting position {}".format(next_position))
        # TODO This shouldn't be a magic variable
        if predicted_distance < 0.1:
            goal_idx = (goal_idx + 1) % len(goal_sequence)
            print("Switching to goal {}".format(goal_sequence[goal_idx]))
        # Extract the next position from the model outputs and send it to the robot.
        position_queue.put(next_position, update_delay_s)
        # TODO The sleep should be slightly less than the movement time to make movement appear
        # smooth and should also consider things like inference time. Just multiplying by 0.95 here
        # is a hack.
        time.sleep(0.95*update_delay_s)

    # Tell the arm that we are done with actuation.
    position_queue.put((None, None))

    # Clean up
    image_processor.destroy_node()
    rclpy.shutdown()

    print("DNN control ending.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--puppet_model', default='px150')
    parser.add_argument('--puppet_name', default='arm2')
    parser.add_argument('--control_calibration', default='configs/arm2_calibration.yaml',
        help="If control_calibration and puppet_calibration differ, correct during replay.")
    parser.add_argument('--puppet_calibration', default='configs/arm2_calibration.yaml',
        help="If control_calibration and puppet_calibration differ, correct during replay.")
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
        choices=["alexnet", "resnet18", "resnet34", "bennet", "resnext50", "resnext34", "resnext18",
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
        '--goal_sequence',
        type=int,
        # Goals to give the robot arm.
        nargs='+',
        required=False,
        default=[],
        help='Goals to give the robot arm.')
    parser.add_argument(
        '--cps',
        type=int,
        default=30,
        help="Commands per second. It is possible that inference may not achieve this rate..")

    # Parse the arguments
    args = parser.parse_args(sys.argv[1:])

    # Set up the queue for communication from the neural network to the robot
    position_commands = Queue()

    # Get calibration corrections and start the robot
    corrections = getCalibrationDiff(args.control_calibration, args.puppet_calibration)

    # Create the robot
    bot = ArmReplay(args.puppet_model, args.puppet_name, corrections, position_commands)

    # Find the size of the model image inputs, vector inputs, and outputs
    vector_input_size = 0
    for input_name in args.vector_inputs:
        # Vector inputs are size 1 unless they are the current robot position
        if input_name == 'current_position':
            # TODO FIXME Whoops, looks like training used all of the joints
            vector_input_size += len(bot.arm.group_info.joint_names) + 3
        else:
            vector_input_size += 1

    dnn_output_size = 0
    for output_name in args.dnn_outputs:
        # Outputs are size 1 unless they are the predicted robot position
        if output_name == 'target_position':
            # TODO FIXME Whoops, looks like training used all of the joints
            dnn_output_size += len(bot.arm.group_info.joint_names) + 3
        else:
            dnn_output_size += 1

    # Arguments for the createModel function
    model_args = {
        'model_type': args.modeltype,
        'in_channels': args.out_channels*args.frames_per_sample,
        'frame_height': args.height,
        'frame_width': args.width,
        'vector_input_size': vector_input_size,
        'output_size': dnn_output_size
    }

    # Arguments for RosHandler constructor
    video_args = {
        'arm_topic_string': "/{}/joint_states".format(args.puppet_name),
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
    inferencer = Thread(target=dnn_inference_thread, args=(bot.arm.group_info.joint_names,
        bot.position_commands, args.model_checkpoint, model_args, args.dnn_outputs,
        args.vector_inputs, args.goal_sequence, video_args, (1. / args.cps), exit_event)).start()

    # This will block until interrupted
    bot.start_robot()

    # TODO FIXME: Go to a home position?
    print("Finished.")


if __name__ == '__main__':
    main()
