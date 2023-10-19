#! /usr/bin/python3
"""
Open a webdataset file and print out the distances from the current_position and target_position for
each entry.
"""

import argparse
import math
import torch
import webdataset as wds

from arm_utility import getDistance

# Insert the bee analysis repository into the path so that the python modules of the git submodule
# can be used properly.
import sys
sys.path.append('bee_analysis')

from arm_utility import computeGripperPosition
from bee_analysis.utility.dataset_utility import decodeUTF8Strings
from bee_analysis.utility.model_utility import (createModel2, hasNormalizers, restoreModel, restoreNormalizers)


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
        decode_strs = ["current_arm_position", "target_arm_position", "current_xyz_position", "target_xyz_position"]
        label_dataset = (
            wds.WebDataset(args.dataset)
            .to_tuple(*decode_strs)
        )
    else:
        # Check if there is model stuff to do
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

        vector_names = checkpoint['metadata']['vector_inputs']
        labels = checkpoint['metadata']['labels']
        decode_strs = ["0.png", *vector_names, *labels]

        # The current arm position must be decoded so that it can be in the output data.
        if 'current_arm_position' not in decode_strs:
            decode_strs.append('current_arm_position')
        if 'current_xyz_position' not in decode_strs:
            decode_strs.append('current_xyz_position')
        if 'target_arm_position' not in decode_strs:
            decode_strs.append('target_arm_position')
        if 'target_xyz_position' not in decode_strs:
            decode_strs.append('target_xyz_position')

        label_dataset = (
            wds.WebDataset(args.dataset)
            .decode("l")
            .to_tuple(*decode_strs)
        )


    # Loop through the dataset and compile label statistics
    label_dataloader = torch.utils.data.DataLoader(label_dataset, num_workers=0, batch_size=1)

    # Set the proper index values to use for the joint positions
    if args.model is None:
        cur_idx = 0
        tar_idx = 1
    else:
        cur_idx = decode_strs.index("current_arm_position") - 1
        tar_idx = decode_strs.index("target_arm_position") - 1
        nn_joint_slice = slice(labels.index('target_arm_position'), labels.index('target_arm_position')+5)
    cur_xyz_idx = decode_strs.index("current_xyz_position") - 1
    tar_xyz_idx = decode_strs.index("target_xyz_position") - 1

    for i, data in enumerate(label_dataloader):
        if args.model is None:
            tensor_data = decodeUTF8Strings(data)
            vector_inputs = None
        else:
            image = data[0].unsqueeze(1).cuda()
            tensor_data = decodeUTF8Strings(data[1:])
            if 0 < len(vector_names):
                vector_inputs = torch.cat(tensor_data[1:1+len(vector_names)], 1)
            else:
                vector_inputs = []

        current = tensor_data[cur_idx][0].tolist()
        target = tensor_data[tar_idx][0].tolist()
        current_xyz = tensor_data[cur_xyz_idx][0].tolist()
        target_xyz = tensor_data[tar_xyz_idx][0].tolist()

        current_position = computeGripperPosition(current)
        target_position = computeGripperPosition(target)
        distance = getDistance(current_position, target_position)
        if args.model is None:
            print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(i, distance,
                *list(current_position), *list(target_position), *current_xyz, *target_xyz))
        else:
            # Normalize inputs: input = (input - mean)/stddev
            normalize_video = True
            if normalize_video:
                v, m = torch.var_mean(image)
                image = (image - m) / v
            if 0 < len(vector_inputs):
                output = net.forward(image, vector_inputs.cuda())
            else:
                output = net.forward(image)
            if denormalizer is not None:
                output = denormalizer(output)
            dnn_position = computeGripperPosition(output[0,nn_joint_slice].tolist())
            print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(i, distance, *list(current_position), *list(target_position), *list(dnn_position)))


if __name__ == '__main__':
    main()
