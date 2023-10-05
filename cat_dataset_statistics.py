#! /usr/bin/python3
"""
Open a webdataset file and print out the distances from the current_position and target_position for
each entry.
"""

import modern_robotics
# These are the robot descriptions to match function calls in the modern robotics package.
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd

import argparse
import math
import torch
import webdataset as wds

from arm_utility import getDistance

# Insert the bee analysis repository into the path so that the python modules of the git submodule
# can be used properly.
import sys
sys.path.append('bee_analysis')

from bee_analysis.utility.dataset_utility import decodeUTF8Strings
from bee_analysis.utility.model_utility import (createModel2, hasNormalizers, restoreModel, restoreNormalizers)


def computeGripperPosition(positions):
    """Get the x,y,z position of the gripper relative to the point under the waist in meters.
    
    Works for the px150 robot arm.

    Arguments:
        positions      (List[float]): Positions of the joints
    Returns:
        x,y,z tuple of the gripper location, in meters
    """
    # TODO These are specific values for the px150 Interbotix robot arm. They should be
    # placed into a library.
    # TODO FIXME These should be calibrated to a known 0 position or the distances will be
    # slightly off (or extremely off, depending upon the calibration).
    # The four joint positions that influence end effector position
    theta0 = positions[0]
    theta1 = positions[1]
    theta2 = positions[2]
    theta3 = positions[3]
    # The lengths of segments (or effective segments) that are moved by the previous joints,
    # in mm
    segment_G = 104    # Height of the pedestal upon which theta1 rotates
    segment_C = 158    # Effective length from theta1 to theta2
    segment_D = 150    # Length from theta2 to theta3
    segment_H = 170    # Length of the grasper from theta4
    arm_x = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta3 + theta2 + theta1)*segment_H)*math.cos(theta0)
    arm_y = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta3 + theta2 + theta1)*segment_H)*math.sin(theta0)
    arm_z = segment_G + math.cos(-theta1)*segment_C + math.sin(-theta1 - theta2)*segment_D + math.sin(-theta1 - theta2 - theta3)*segment_H
    # Return the x,y,z end effector coordinates in meters
    return (arm_x/1000., arm_y/1000., arm_z/1000.)

    #model_generator = getattr(mrd, 'px150')
    #robot_model = model_generator()
    ## Below is "correct" but gets incorrect results. Instead we will use geometry and hand-measured
    ## values for the arm segments. There must be something wrong in the M or Slist matrices
    ## (probably the Slist) but the problem isn't immediately apparent.
    ## 'M' is the home configuration of the robot, Slist has the joint screw axes at the home
    ## position. This should return the end effector position.
    #T = modern_robotics.FKinSpace(robot_model.M, robot_model.Slist, positions)
    ## Return the x,y,z components of the translation matrix.
    #return (T[0][-1], T[1][-1], T[2][-2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str,
        help='Path for the WebDataset archive.')
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default=None,
        help='Path for a model to test.')

    args = parser.parse_args()

    if args.model is None:
        decode_strs = ["current_arm_position", "target_arm_position"]
        label_dataset = (
            wds.WebDataset(args.dataset)
            .to_tuple(*decode_strs)
        )
    else:
        # TODO If using a model, the decode strings should really come from
        # checkpoint['metadata']['labels'] and checkpoint['metadata']['vector_inputs']
        decode_strs = ["0.png", "current_arm_position", "target_arm_position"]
        label_dataset = (
            wds.WebDataset(args.dataset)
            .decode("l")
            .to_tuple(*decode_strs)
        )
    # Loop through the dataset and compile label statistics
    label_dataloader = torch.utils.data.DataLoader(label_dataset, num_workers=0, batch_size=1)

    # Check if there is model stuff to do
    if args.model is not None:
        checkpoint = torch.load(args.model)
        # Create the model and load the weights from the given checkpoint.
        # Get the model arguments from the training metadata stored in the checkpoint
        net = createModel2(checkpoint['metadata']['modeltype'], checkpoint['metadata']['model_args'])
        restoreModel(args.model, net)
        # Restore the denormalization network, if it was used.
        if hasNormalizers(args.model):
            _, denormalizer = restoreNormalizers(args.model)
            denormalizer.eval().cuda()
        else:
            denormalizer = None
        net = net.eval().cuda()

    for i, data in enumerate(label_dataloader):
        if args.model is None:
            tensor_data = decodeUTF8Strings(data)
        else:
            image = data[0].unsqueeze(1).cuda()
            tensor_data = decodeUTF8Strings(data[1:])
        current = tensor_data[0][0].tolist()
        target = tensor_data[1][0].tolist()

        current_position = computeGripperPosition(current)
        target_position = computeGripperPosition(target)
        distance = getDistance(current_position, target_position)
        if args.model is None:
            print("{}, {}, {}, {}, {}, {}, {}, {}".format(i, distance, *list(current_position), *list(target_position)))
        else:
            # Normalize inputs: input = (input - mean)/stddev
            normalize_video = True
            if normalize_video:
                v, m = torch.var_mean(image)
                image = (image - m) / v
            output = net.forward(image, tensor_data[0].cuda())
            if denormalizer is not None:
                output = denormalizer(output)
            dnn_position = computeGripperPosition(output[0].tolist())
            print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(i, distance, *list(current_position), *list(target_position), *list(dnn_position)))

            


if __name__ == '__main__':
    main()
