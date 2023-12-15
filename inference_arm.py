#!/usr/bin/env python3

# This script runs a trained neural network to control a robot arm..
# Copyright 2023 Bernhard Firner


import argparse
import cv2
import ffmpeg
import numpy
import pathlib
import rclpy
import sys
import time
import torch
import yaml

from queue import Queue
from threading import Event, Lock, Thread

# Ros includes
from rclpy.node import Node
from sensor_msgs.msg import (Image, JointState)

# Includes from this project
from arm_utility import (ArmReplay, computeGripperPosition, getDistance, interpretRTZPrediction, rSolver, RThetaZtoXYZ, XYZToRThetaZ)
from data_utility import vidSamplingCommonCrop


# Insert the bee analysis repository into the path so that the python modules of the git submodule
# can be used properly.
import sys
sys.path.append('bee_analysis')

from bee_analysis.utility.model_utility import (createModel2, hasNormalizers, restoreModel,
        restoreNormalizers)


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

        # This should be the calibrated joint state message created by the ArmReplay class, not the
        # default one found in the Interbotix nodes
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
                .filter('normalize', independence=1.0)
            )
            if 3 == self.channels_per_frame:
                pix_fmt='rgb24'
            else:
                pix_fmt='gray'
            self.input_stream = (self.input_stream
                # Output to another pipe so that the data can be read back in
                .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
            )

        # Process the frame
        self.input_stream.stdin.write(numpy.uint8(msg.data).tobytes())
        crop_bytes = self.input_stream.stdout.read(self.crop_width * self.crop_height * self.channels_per_frame)
        # Reshape the frame and put the channels before the other dimensions
        np_frame = numpy.frombuffer(crop_bytes, numpy.uint8).reshape([1, self.crop_height, self.crop_width,
            self.channels_per_frame])

        # Display the cropped image that will be used for inference
        cv2.imshow("inference patch", np_frame[0])
        # waitKey must be called for open CV to actually display the frame.
        cv2.waitKey(1)

        # TODO The exact frame preprocessing must match what was done in dataprep, but currently the
        # image formats are different (e.g. in dataprep images are saved as pngs whereas here they
        # are coming directly from the video)
        out_img = torch.tensor(
            data=np_frame, dtype=torch.uint8).permute(0, 3, 1, 2).to(dtype=torch.float) / 255.0

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
            # TODO FIXME Correct msg.position with the calibration from the robot arm
            self.position = msg.position
            self.velocity = msg.velocity
            self.effort = msg.effort

    def __del__(self):
        # Remove the patch display window
        cv2.destroyAllWindows()
        super(RosHandler, self).__del__()


def dnn_inference_thread(robot_joint_names, position_queue, model_checkpoint, dnn_outputs,
        vector_inputs, goal_sequence, video_args, normalize_video, update_delay_s, exit_event):
    """Puts positions into the position queue.

    Arguments:
        robot_joint_names (list[str]): Joint names in the robot arm.
        position_queue    (Queue): Input queue to the robot
        model_checkpoint  (str): The path to the model checkpoint.
        dnn_outputs       (list[str]): Names of the DNN outputs
        vector_inputs     (list[str]): Names of the DNN vector inputs
        goal_sequence     (list[int]): Sequence of goals
        video_args             (dict): Video arguments
        normalize_video        (bool): Normalize frames: frame = (frame - mean)/stddev
        update_delay_s    (float): Update delay in seconds
        exit_event        (Event): Exit when is_set is True.
    """
    print("DNN inferencing thread started.")

    # Create the model and load the weights from the given checkpoint.
    checkpoint = torch.load(model_checkpoint)
    # Get the model arguments from the training metadata stored in the checkpoint
    net = createModel2(checkpoint['metadata']['modeltype'], checkpoint['metadata']['model_args'])
    restoreModel(model_checkpoint, net)
    # Restore the denormalization network, if it was used.
    if hasNormalizers(model_checkpoint):
        _, denormalizer = restoreNormalizers(model_checkpoint)
        denormalizer.eval().cuda()
    else:
        denormalizer = None
    net.eval().cuda()

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
    vector_size = 0
    # History goal distance is initialized to 10cm
    prev_goal_distance = 0.1
    for input_name in vector_inputs:
        # Vector inputs are size 1 unless they are the current robot position
        if input_name == 'current_position':
            # This uses all joints. 'current_arm_position' has only the arm ones
            vector_input_locations[input_name] = slice(vector_size, vector_size + len(robot_joint_names) + 3)
            vector_size += len(robot_joint_names) + 3
        elif input_name == 'current_arm_position':
            vector_input_locations[input_name] = slice(vector_size, vector_size + len(robot_joint_names))
            vector_size += len(robot_joint_names)
        elif input_name == 'current_rtz_position':
            vector_input_locations[input_name] = slice(vector_size, vector_size + 3)
            vector_size += 3
        elif input_name == 'goal_mark':
            vector_input_locations[input_name] = vector_size
            vector_size += 1
        elif input_name[:len("goal_distance_prev_")] == "goal_distance_prev_":
            vector_input_locations[input_name] = vector_size
            vector_size += 1
            # TODO We will assume that the prediction_distance and the previous goal distance are
            # the same, so we only need to buffer a single previous goal distance prediction.
            goal_distance_history = int(input_name[len("goal_distance_prev_"):-2])
        else:
            Exception("Unknown vector input: {}".format(input_name))

    # Create tensor inputs
    vector_input_buffer = torch.zeros([1, vector_size]).float().cuda()

    # Mark the locations of the model outputs
    output_locations = {}
    out_idx = 0
    for output_name in dnn_outputs:
        if output_name == 'target_position':
            output_locations[output_name] = slice(out_idx, out_idx+len(robot_joint_names) + 3)
            out_idx += len(robot_joint_names) + 3
        elif output_name == 'target_arm_position':
            output_locations[output_name] = slice(out_idx, out_idx+len(robot_joint_names))
            out_idx += len(robot_joint_names)
        elif output_name == 'target_rtz_position':
            output_locations[output_name] = slice(out_idx, out_idx+3)
            out_idx += len(robot_joint_names)
        elif output_name == 'rtz_classifier':
            rtz_classifier_size = 10
            output_locations[output_name] = slice(out_idx, out_idx+rtz_classifier_size)
            out_idx += rtz_classifier_size
        else:
            output_locations[output_name] = out_idx
            out_idx +=1

    # Wait until there is some data available
    ready = False
    while not exit_event.is_set() and not ready:
        with data_read_lock:
            ready = image_processor.frame is not None and image_processor.position is not None

    first_movement = True

    with torch.no_grad():
        # Wait for messages until an interrupt is received
        while not exit_event.is_set():
            # Forward the frame through the DNN model
            # This always uses the most recent frame, so if the update rate is much faster than the
            # frame rate the only thing that will update between commands is the robot's current
            # position.
            with data_read_lock:
                new_frame = image_processor.frame.float()
                joint_positions = image_processor.position

            new_frame = new_frame.cuda()
            # Normalize inputs: input = (input - mean)/stddev
            if checkpoint['metadata']['normalize_images']:
                v, m = torch.var_mean(new_frame)
                new_frame = (new_frame - m) / v

            # Set up vector inputs
            for input_name in vector_inputs:
                # Vector inputs are size 1 unless they are the current robot position
                outslice = vector_input_locations[input_name]
                if input_name == 'current_position':
                    vector_input_buffer[0, outslice].copy_(torch.tensor(joint_positions))
                elif input_name == 'current_arm_position':
                    vector_input_buffer[0, outslice].copy_(torch.tensor(joint_positions))
                elif input_name == 'current_rtz_position':
                    current_rtz_position = XYZToRThetaZ(*computeGripperPosition(joint_positions, segment_lengths=[0.104, 0.158, 0.147, 0.175]))
                    print("Writing location {} into slice {}".format(current_rtz_position, outslice))
                    vector_input_buffer[0, outslice].copy_(torch.tensor(current_rtz_position))
                elif input_name == 'goal_mark':
                    vector_input_buffer[0, outslice.start] = goal_sequence[goal_idx]
                elif input_name[:len("goal_distance_prev_")] == "goal_distance_prev_":
                    vector_input_buffer[0, outslice.start] = prev_goal_distance

            # goal_mark determines the state to feed back to the network via the vector inputs
            # goal_distance is used to determine when to switch goals.
            # The target_position output contains the next set of joint positions.

            if 0 == len(vector_inputs):
                net_out = net(new_frame)
            else:
                net_out = net(new_frame, vector_input_buffer)
            if denormalizer is not None:
                net_out = denormalizer(net_out)
            predicted_distance = 1.0
            if 'goal_distance' in dnn_outputs:
                net_out[0, output_locations['goal_distance']].item()
            if 'target_arm_position' in output_locations:
                next_position = net_out[0, output_locations['target_arm_position']].tolist()
            elif 'rtz_classifier' in output_locations:
                # TODO This was a training parameter, it should be pulled from the dataset
                # Setting the threshold to 1cm here
                threshold = 0.01
                current_xyz = computeGripperPosition(joint_positions)
                predictions = net_out[0, output_locations['rtz_classifier']].tolist()
                next_rtz_position = interpretRTZPrediction(*XYZToRThetaZ(*current_xyz), threshold, predictions)
                next_xyz_position = RThetaZtoXYZ(*next_rtz_position)
                # Solve for the joint positions
                print("Trying to move from {} to {}".format(XYZToRThetaZ(*current_xyz), next_rtz_position))
                middle_joints = rSolver(next_rtz_position[0], next_rtz_position[2], segment_lengths=[0.104, 0.158, 0.147, 0.175])
                next_position = [
                    # Waist
                    next_rtz_position[1],
                    # Shoulder
                    middle_joints[0],
                    # Elbow
                    middle_joints[1],
                    # Wrist angle
                    middle_joints[2],
                    # Wrist rotate
                    0.0,
                ]
            else:
                raise RuntimeError("No target position for robot in DNN outputs!")
            print("Requesting position {}".format(next_position))
            # TODO This shouldn't be a magic variable
            if predicted_distance < 0.01:
                goal_idx = (goal_idx + 1) % len(goal_sequence)
                print("Switching to goal {}".format(goal_sequence[goal_idx]))
                # History goal distance is initialized to 10cm
                prev_goal_distance = 0.1
            else:
                # Save the current goal distance to be used as a status input.
                prev_goal_distance = predicted_distance
            delay = update_delay_s
            # Make the first movement slow
            if first_movement:
                delay += 1
            # Extract the next position from the model outputs and send it to the robot.
            position_queue.put((next_position, delay))
            # TODO The sleep should be slightly less than the movement time to make movement appear
            # smooth and should also consider things like inference time. Just subtracting 10 ms
            # here is a hack
            time.sleep(delay - 0.01)

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
    parser.add_argument('--puppet_calibration', default='configs/arm2_calibration2.yaml',
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
        choices=["alexnet", "resnet18", "resnet34", "bennet", "compactingbennet", "dragonfly", "resnext50", "resnext34", "resnext18",
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
    parser.add_argument(
        '--normalize_video',
        required=False,
        default=False,
        action="store_true",
        help=("Normalize video: input = (input - mean) / stddev. "
            "Some normalization is already done through ffmpeg, but png and jpeg differences could be fixed with this."))

    # Parse the arguments
    args = parser.parse_args(sys.argv[1:])

    # Set up the queue for communication from the neural network to the robot
    position_commands = Queue()

    # Get calibration corrections for the robot
    with open(args.puppet_calibration, 'r') as data:
        puppet_calibration = yaml.safe_load(data)

    # Create the robot
    bot = ArmReplay(args.puppet_model, args.puppet_name, puppet_calibration, position_commands)

    # Find the size of the model image inputs, vector inputs, and outputs
    vector_input_size = 0
    for input_name in args.vector_inputs:
        # Vector inputs are size 1 unless they are the current robot position
        if input_name == 'current_position':
            # Generally, current_arm_position is the correct output to use
            vector_input_size += len(bot.arm.group_info.joint_names) + 3
        if input_name == 'current_arm_position':
            vector_input_size += len(bot.arm.group_info.joint_names)
        else:
            vector_input_size += 1

    dnn_output_size = 0
    for output_name in args.dnn_outputs:
        # Outputs are size 1 unless they are the predicted robot position
        if output_name == 'target_position':
            # Generally, target_arm_position is the correct output to use
            dnn_output_size += len(bot.arm.group_info.joint_names) + 3
        elif output_name == 'target_arm_position':
            dnn_output_size += len(bot.arm.group_info.joint_names)
        else:
            dnn_output_size += 1

    # Arguments for RosHandler constructor
    video_args = {
        # Subscribe to the calibrated joint positions so that processing is robot agnostic
        'arm_topic_string': "/{}/calibrated_joint_states".format(args.puppet_name),
        'video_topic_string': args.image_topic,
        'scale': args.video_scale,
        'crop_width': args.width,
        'crop_height': args.height,
        'crop_x': args.crop_x_offset,
        'crop_y': args.crop_y_offset,
        'frames_per_sample': args.frames_per_sample,
        'channels_per_frame': args.out_channels
    }

    # TODO FIXME This needs the calibration so that it can correct the ROS joint position messages
    # Start the DNN inference thread before calling the blocking start_robot() call.
    inferencer = Thread(target=dnn_inference_thread, args=(bot.arm.group_info.joint_names,
        bot.position_commands, args.model_checkpoint, args.dnn_outputs,
        args.vector_inputs, args.goal_sequence, video_args, args.normalize_video, (1. / args.cps), exit_event)).start()

    # This will block until interrupted
    bot.start_robot()

    # TODO FIXME: Go to a home position?
    print("Finished.")


if __name__ == '__main__':
    main()
