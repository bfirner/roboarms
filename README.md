# What is this?

This repository has a collection of code and scripts to do data collection on a robot arm, label
that data for neural network training, train a model, and then control the robot arm with that
model. It depends upon another repository, <https://github.com/bfirner/bee_analysis>, so be sure to
grab it as well, either using the `--recursive` flag when cloning this repository, or by running
`git submodule update --init --recursive` afterwards.


# Getting Started

## Recording Data

### First, launch of the robo_arms launch file
> ros2 launch robo_arms_launch.yaml

The launch file starts the control for the two arms, runs v4l2 data collection, and opens a
display window for the operator to view camera inputs. It is the equivalent of the following
commands:
> ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150 mode_configs:=configs/modes_1.yaml motor_configs:=configs/px150_motor.yaml robot_name:=arm1 use_gripper:=true load_configs:=false use_rviz:=false &
> ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px150 mode_configs:=configs/modes_2.yaml motor_configs:=configs/px150_motor.yaml robot_name:=arm2 use_gripper:=true load_configs:=false use_rviz:=true &
> ros2 run v4l2_camera v4l2_camera_node --ros-args --params-file cam_params.yaml
> ros2 run rqt_image_view rqt_image_view

### Second and third, launch of the manipulator and the arm puppet.

Run the pupper controller with 
> python3 manipulator.py

Run the puppet with
> python3 arm_puppet.py

### Finally, start the recorder

Collect a ros bag with
> ros2 bag record /arm1/joint_states /arm2/joint_states /arm1/robot_description /arm2/robot_description /camera_info

Collect video with
> python3 videorecorder.py

The video is collected as a separate action because ros bags store video as invidual frames, which
is space inefficient and a waste of time.

## Replay

Collected data can be replayed. By default the robot named 'arm2' is used, but that can be
overriden in the arguments. To control 'arm2' using 'arm1' data, for example:
> python3 replay_arm.py <rosbag> 60 --src_robot arm1 --control_calibration configs/arm1_calibration.yaml

## Labelling Data

Running the labeller is simple.
> python3 datalabeller.py <bag path>  --train_robot arm2

Press 'h' on the gui window to see a list of commands (displayed on the terminal)

## Data Preparation

Data needs to be prepared for training. This project will put it into a webdataset. The command
looks something like this:
> python3 dataprep.py --crop_x_offset 200 --video_scale 0.5 --sample_prob 0.01 --crop_noise 10 --goals <used positions> --prediction_distance 0.05 --train_robot arm1 <something.tar> <rosbag directories>

The `dataprep.py` with the `--help` flag for a full list of options.

## Training

A trained model will predict the distance to the current goal, the class of the current goal, and
the position vector to either move to the desired prediction distance (controlled by the
`--prediction_distance` option in `dataprep.py`) or to move to the next goal.

Training used the bee_analysis code like this:
> python3 bee_analysis/VidActRecTrain.py --not_deterministic <dataset.tar> --outname <name.pth> --labels goal_mark goal_distance target_position --skip_metadata --convert_idx_to_classes 0 --loss_fun MSELoss --modeltype alexnet --vector_inputs initial_mark current_position --epochs 20

The `--labels` option specifies the DNN outputs. The `goal_mark` refers to the marks used during
labelling and is the mark that the arm is moving towards. During inference this will be used to
identify that the arm has reached one goal and has transitioned to the next one. The `goal_distance`
is the distance from the gripper to the target and the `target_position` is the position to predict.
That refers to either the position of the next mark or the position at `--prediction_distance`
specified during dataprep, whichever is lesser.

The `--vector_inputs` option specifies non-image DNN inputs. `initial_mark` is the mark that the arm
is leaving from (and is necessary to control the robot's behavior) and `current_position` is the
pose of the robot.

TODO: The `goal_mark` should be treated as a classification target while the other two are
regression, but currently all outputs will be treated as regression.

Important note! The current version of the training code *always* converts images to a single
channel. This means that the inference code must always use the `--out_channels 1` option to
compensate.

## Inference

> python3 inference_arm.py --crop_x_offset 200 --video_scale 0.5 --modeltype alexnet --model_checkpoint <name.pth> --goal_sequence 0 1 --cps 5 --vector_inputs initial_mark current_position --out_channels 1 --dnn_outputs goal_mark goal_distance target_position

TODO: During inference, the controlling system will also need to track the
current state and feed it back using the model's prediction of goals.
