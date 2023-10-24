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
    parser.add_argument(
        '--save_images',
        type=int,
        required=False,
        default=0,
        help='Save the first specified number of images images as "dataset_images.png".')
    parser.add_argument(
        '--save_features',
        type=int,
        required=False,
        default=0,
        help='Save the feature maps of the specified images as "dataset_features_{image}_{layer}.png".')
    parser.add_argument(
        '--print_features',
        required=False,
        default=False,
        action='store_true',
        help='Print out features to the csv from the last feature layer.')

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
        #net = net.eval().cuda()
        net.eval()

        vector_names = checkpoint['metadata']['vector_inputs']
        labels = checkpoint['metadata']['labels']
        decode_strs = ["0.png", *vector_names, *labels]


        if "target_arm_position" in checkpoint['metadata']['labels']:
            nn_joint_slice = slice(labels.index('target_arm_position'), labels.index('target_arm_position')+5)
            dnn_output_to_xyz = lambda out: computeGripperPosition(out[0,nn_joint_slice].tolist())
        elif "target_xyz_position" in checkpoint['metadata']['labels']:
            dnn_output_to_xyz = lambda out: out[0].tolist()
        elif "current_xyz_position" in checkpoint['metadata']['labels']:
            dnn_output_to_xyz = lambda out: out[0].tolist()

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
    cur_xyz_idx = decode_strs.index("current_xyz_position") - 1
    tar_xyz_idx = decode_strs.index("target_xyz_position") - 1

    # Used if we are saving images
    if 0 < args.save_images:
        tiled_images = None
        tile_locations = []
        if args.model is not None:
            tile_predictions = []

    # Print out the header
    header = "sample, target distance, current x, current y, current z, target x, target y, "
    "target z, current waist, current shoulder, current elbow, current wrist_angle, "
    "current wrist_rotate, target waist, target shoulder, target elbow, target wrist_angle, "
    "target wrist_rotate"
    if args.model is not None:
        header += ", dnn x, dnn y, dnn z, dnn waist, dnn shoulder, dnn elbow, dnn wrist_angle, dnn wrist_rotate"
        # TODO Does the header need a number for every feature?
    print(header)

    for i, data in enumerate(label_dataloader):
        # We don't use the image unless it is being saved or forwarded through a network
        if i < args.save_images or args.model is not None:
            image = data[0].unsqueeze(1).cuda()
        if args.model is None:
            tensor_data = decodeUTF8Strings(data)
            vector_inputs = None
        else:
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
        csv_line = ("{}" + ", {}" * 17).format(i, distance, *list(current_xyz), *list(target_xyz),
                *list(current), *list(target))
        if args.model is not None:
            # Normalize inputs: input = (input - mean)/stddev
            if checkpoint['metadata']['normalize_images']:
                v, m = torch.var_mean(image)
                image = (image - m) / v
            with torch.no_grad():
                if 0 < len(vector_inputs):
                    output = net(image, vector_inputs.cuda())
                else:
                    output = net(image)

            if denormalizer is not None:
                output = denormalizer(output)
            dnn_xyz = dnn_output_to_xyz(output)
            # Add the DNN predictions to the output line
            csv_line += (", {:.8f}" * 8).format(*list(dnn_xyz), *output[0].tolist())
            if i < args.save_images:
                tile_predictions.append(list(dnn_xyz))

            if i < args.save_features or args.print_features:
                # Fetch the feature maps
                maps = net.produceFeatureMaps(image)
                # Store the flattened last layer in the output string if requested
                if args.print_features:
                    flat_features = maps[-1].flatten().tolist()
                    csv_line += ", " + ", ".join(["{:.6f}".format(feature) for feature in flat_features])
                # Save the feature maps
                if i < args.save_features:
                    for layer_i, fmap in enumerate(maps):
                        # Put all of the maps onto the same channel and save them as an image
                        tiled_shape = (1, fmap.size(1)*fmap.size(2), fmap.size(3))
                        new_view = fmap.reshape(tiled_shape)
                        img = transforms.ToPILImage()(new_view).convert('L')
                        img.save("dataset_features_{}_{}.png".format(batch_num, layer_i))
                # Delete the feature maps to force memory cleanup
                del maps

        # Print the completed line for the csv
        print(csv_line)

        if i < args.save_images:
            # Initialize the output images
            if tiled_images is None:
                tiled_shape = (image.size(1), args.save_images*image.size(2), image.size(3))
                tiled_images = torch.zeros(tiled_shape)
            y_offset = i*image.size(2)
            tiled_images[:,y_offset:y_offset+image.size(2),:] = image[0]
            tile_locations.append(current_position)

    if 0 < args.save_images:
        # Helper function to convert to images and PIL image libraries
        from torchvision import transforms
        from PIL import ImageDraw
        from PIL import ImageFont
        font = ImageFont.truetype('DejaVuSansMono.ttf', size=14)
        img = transforms.ToPILImage()(tiled_images).convert('RGB')
        draw = ImageDraw.Draw(img)
        color = (255, 100, 100, 255)
        # Write the location onto the image
        for i, location in enumerate(tile_locations):
            label_str = "{}: {:.4f}, {:.4f}, {:.4f}".format(i, *location)
            x = 20
            y = int(i * tiled_images.size(1)/args.save_images + 20)
            draw.text((x,y), label_str, fill=color, font=font)
            if args.model is not None:
                y = int(i * tiled_images.size(1)/args.save_images + 40)
                label_str = "DNN: {:.4f}, {:.4f}, {:.4f}".format(*tile_predictions[i])
                draw.text((x,y), label_str, fill=color, font=font)
        img.save("dataset_images.png")



if __name__ == '__main__':
    main()
