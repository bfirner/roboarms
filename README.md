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

## Labelling Data

Running the labeller is simple.
> python3 datalabeller.py <bag path>  --train_robot arm2

Press 'h' on the gui window to see a list of commands (displayed on the terminal)

## Data Preparation

Data needs to be prepared for training. This project will put it into a webdataset. The command
looks something like this:
> python3 dataprep.py --crop_x_offset 200 --video_scale 0.5 --sample_prob 0.01 --crop_noise 10 <something.tar> <rosbag directories>

The `dataprep.py` with the `--help` flag for a full list of options.

## Training

A trained model will predict the distance to the current goal, the class of the current goal, and
the position vector to either move to the desired prediction distance (controlled by the
`--prediction_distance` option in `dataprep.py`) or to move to the next goal.

Training used the bee_analysis code like this:
> python3 bee_analysis/VidActRecTrain.py --not_deterministic <something.tar> --outname <output path/name> --labels goal_mark goal_distance target_position --skip_metadata

TODO: The `goal_mark` should be treated as a classification target while the other two are
regression, but current they will all be treated as regression.

TODO: As described, this model will not function. It will also require the current state, as
provided by the `initial_mark`. During inference, the controlling system will also need to track the
current state and feed it back using the model's prediction of goals.
