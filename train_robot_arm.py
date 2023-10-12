#! /usr/bin/python3

"""
This will train a model using a webdataset tar archive for data input.
If you use Cuda >= 10.2 then running in deterministic mode requires this environment variable to be
set: CUBLAS_WORKSPACE_CONFIG=:4096:8
To enable deterministic mode use the option --deterministic
There is another problem is AveragePooling that may also make it impossible to use deterministic
mode.
"""

# Not using video reading library from torchvision.
# It only works with old versions of ffmpeg.
import argparse
import csv
import datetime
import functools
import heapq
import io
import math
import numpy
import os
import random
import sys
import torch
import torch.cuda.amp
import webdataset as wds
from collections import namedtuple
# Helper function to convert to images
from torchvision import transforms

from arm_utility import (grepGripperLocationFromTensors)

# Insert the bee analysis repository into the path so that the python modules of the git submodule
# can be used properly.
import sys
sys.path.append('bee_analysis')

from bee_analysis.utility.dataset_utility import (extractVectors, getImageSize, getVectorSize)
from bee_analysis.utility.eval_utility import (OnlineStatistics, RegressionResults, WorstExamples)
from bee_analysis.utility.model_utility import (restoreModelAndState)
from bee_analysis.utility.train_utility import (LabelHandler, evalEpoch, trainEpoch)

from bee_analysis.models.alexnet import AlexLikeNet
from bee_analysis.models.bennet import BenNet
from bee_analysis.models.denormalizer import Denormalizer, Normalizer
from bee_analysis.models.resnet import (ResNet18, ResNet34)
from bee_analysis.models.resnext import (ResNext18, ResNext34, ResNext50)
from bee_analysis.models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)



# Argument parser setup for the program.
parser = argparse.ArgumentParser(
    description="Perform data preparation for DNN training on a video set.")
parser.add_argument(
    'dataset',
    nargs='+',
    type=str,
    help='Dataset for training.')
parser.add_argument(
    '--sample_frames',
    type=int,
    required=False,
    default=1,
    help='Number of frames in each sample.')
parser.add_argument(
    '--outname',
    type=str,
    required=False,
    default="model.checkpoint",
    help='Base name for model, checkpoint, and metadata saving.')
parser.add_argument(
    '--resume_from',
    type=str,
    required=False,
    help='Model weights to restore.')
parser.add_argument(
    '--epochs',
    type=int,
    required=False,
    default=15,
    help='Total epochs to train.')
parser.add_argument(
    '--seed',
    type=int,
    required=False,
    default='0',
    help="Seed to use for RNG initialization.")
parser.add_argument(
    '--normalize',
    required=False,
    default=False,
    action="store_true",
    help=("Normalize inputs: input = (input - mean) / stddev. "
        "Note that VidActRecDataprep is already normalizing so this may not be required."))
parser.add_argument(
    '--normalize_outputs',
    required=False,
    default=False,
    action="store_true",
    help=("Normalize the outputs: output = (output - mean) / stddev. "
        "This will read the entire dataset to find these values. After training the final weights will be adjusted so that no additional processing is required."))
parser.add_argument(
    '--modeltype',
    type=str,
    required=False,
    default="resnext18",
    choices=["alexnet", "resnet18", "resnet34", "bennet", "resnext50", "resnext34", "resnext18",
    "convnextxt", "convnextt", "convnexts", "convnextb"],
    help="Model to use for training.")
parser.add_argument(
    '--no_train',
    required=False,
    default=False,
    action='store_true',
    help='Set this flag to skip training. Useful to load an already trained model for evaluation.')
parser.add_argument(
    '--evaluate',
    type=str,
    required=False,
    default=None,
    help='Evaluate with the given dataset.')
parser.add_argument(
    '--save_top_n',
    type=int,
    required=False,
    default=None,
    help='Save N images for each class with the highest classification score. Works with --evaluate')
parser.add_argument(
    '--save_worst_n',
    type=int,
    required=False,
    default=None,
    help='Save N images for each class with the lowest classification score. Works with --evaluate')
parser.add_argument(
    '--deterministic',
    required=False,
    default=False,
    action='store_true',
    help='Set this to disable deterministic training.')
parser.add_argument(
    '--labels',
    type=str,
    # Support an array of strings to have multiple different label targets.
    nargs='+',
    required=False,
    default=["cls"],
    help='Files to decode from webdataset as the DNN output target labels.')
parser.add_argument(
    '--vector_inputs',
    type=str,
    # Support an array of strings to have multiple different label targets.
    nargs='+',
    required=False,
    default=[],
    help='Files to decode from webdataset as DNN vector inputs.')
parser.add_argument(
    '--skip_metadata',
    required=False,
    default=False,
    action='store_true',
    help='Set to skip loading metadata.txt from the webdataset.')
parser.add_argument(
    '--loss_fun',
    required=False,
    default='MSELoss',
    choices=['L1Loss', 'MSELoss'],
    type=str,
    help="Loss function to use during training.")

args = parser.parse_args()


torch.use_deterministic_algorithms(args.deterministic)
torch.manual_seed(args.seed)
random.seed(0)
numpy.random.seed(0)

# Later on we will need to change behavior if the loss function is regression rather than
# classification
regression_loss = ['L1Loss', 'MSELoss']

in_frames = args.sample_frames
decode_strs = []
label_decode_strs = []
# The image for a particular frame
for i in range(in_frames):
    decode_strs.append(f"{i}.png")

# The class label(s) or regression targets
label_range = slice(len(decode_strs), len(decode_strs) + len(args.labels))
for label_str in args.labels:
    decode_strs.append(label_str)
    label_decode_strs.append(label_str)

# Vector inputs (if there are none then the slice will be an empty range)
vector_range = slice(label_range.stop, label_range.stop + len(args.vector_inputs))
for vector_str in args.vector_inputs:
    decode_strs.append(vector_str)


# The loss will be calculated from the distance rather than the joint positions, but if they are not
# in the labels we use the loss directly.
loss_fn = getattr(torch.nn, args.loss_fun)()

def computeDistanceForLoss(tensor_a, tensor_b):
    # The distance is the square root of the sum of the squares of the differences in the
    # coordinates. The tensors have a batch dimension first, so sum along the second dimension.
    diff = grepGripperLocationFromTensors(tensor_b) - grepGripperLocationFromTensors(tensor_a)
    squares = torch.pow(diff, 2)
    sums = torch.sum(squares, dim=1, keepdim=True)
    distance = torch.sqrt(sums)

    #distance = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1, keepdim=True))
    return distance

def lossWithDistance(output, labels, loss_fn, joint_range):
    distance_error = computeDistanceForLoss(output[:,joint_range], labels[:,joint_range])
    if 'MSELoss' == args.loss_fun:
        distance_error = distance_error.pow(2)

    loss = distance_error.sum() + loss_fn(output, labels)

    # If there are labels before or after the joint positions then add them into the loss along with
    # the distance error
    loss = distance_error
    #if 0 < joint_range.start:
    #    before_output = output[:, 0:joint_range.start]
    #    before_labels = labels[:, 0:joint_range.start]
    #    loss = torch.cat((loss, loss_fn(before_output, before_labels)))
    #if joint_range.stop < labels.size(1):
    #    after_output = output[:, joint_range.stop:]
    #    after_labels = labels[:, joint_range.stop:]
    #    loss = torch.cat((loss, loss_fn(after_output, after_labels)))
    return torch.sum(loss)


print(f"Training with dataset {args.dataset}")
# If we are converting to a one-hot encoding output then we need to check the argument that
# specifies the number of output elements. Otherwise we can check the number of elements in the
# webdataset.
label_size = getVectorSize(args.dataset, decode_strs, label_range)

label_names = None
# See if we can deduce the label names
label_names = []
for label_idx, out_elem in enumerate(range(label_range.start, label_range.stop)):
    label_elements = getVectorSize(args.dataset, decode_strs, slice(out_elem, out_elem+1))
    # Change the loss function to accommodate calculating distance from the joints
    if 'target_arm_position' == args.labels[label_idx]:
        joint_range = slice(len(label_names), len(label_names) + label_elements)
        loss_fn = functools.partial(lossWithDistance, loss_fn=getattr(torch.nn, args.loss_fun)(), joint_range=joint_range)

    # Give this output the label name directly or add a number if multiple outputs come from
    # this label
    if 1 == label_elements:
        label_names.append(args.labels[label_idx])
    else:
        for i in range(label_elements):
            label_names.append("{}-{}".format(args.labels[label_idx], i))

# Find the values required to normalize network outputs
if not args.normalize_outputs:
    denormalizer = None
    normalizer = None
else: 
    print("Reading dataset to compute label statistics for normalization.")
    label_stats = [OnlineStatistics() for _ in range(label_size)]
    label_dataset = (
        wds.WebDataset(args.dataset)
        .to_tuple(*label_decode_strs)
    )
    # TODO Loop through the dataset and compile label statistics
    label_dataloader = torch.utils.data.DataLoader(label_dataset, num_workers=0, batch_size=1)
    for data in label_dataloader:
        for label, stat in zip(extractVectors(data, slice(0, label_size))[0].tolist(), label_stats):
            stat.sample(label)
    # Now record the statistics
    label_means = []
    label_stddevs = []
    for stat in label_stats:
        label_means.append(stat.mean())
        label_stddevs.append(math.sqrt(stat.variance()))

    print("Normalizing labels with the follow statistics:")
    for lnum, lname in enumerate(label_names):
        print("{} mean and stddev are: {} and {}".format(lname, label_means[lnum], label_stddevs[lnum]))

    # Convert label means and stddevs into tensors and send to modules for ease of application
    label_means = torch.tensor(label_means).cuda()
    label_stddevs = torch.tensor(label_stddevs).cuda()
    denormalizer = Denormalizer(means=label_means, stddevs=label_stddevs).cuda()
    normalizer = Normalizer(means=label_means, stddevs=label_stddevs).cuda()

    # If any of the standard deviations are 0 they must be adjusted to avoid mathematical errors.
    # More importantly, any labels with a standard deviation of 0 should not be used for training
    # since they are just fixed numbers.
    if (label_stddevs.abs() < 0.0001).any():
        print("Some labels have extremely low variance--they may be fixed values. Check your dataset.")
        exit(1)

label_handler = LabelHandler(label_size, label_range, label_names)

# The label value may need to be adjusted, to normalize before calculating loss.
if normalizer is not None:
    label_handler.setPreprocess(lambda labels: normalizer(labels))


# Network outputs may need to be postprocessed for evaluation if some postprocessing is being done
# automatically by the loss function.
# Use an identify function unless normalization is being used.
if denormalizer is None:
    nn_postprocess = lambda x: x
else:
    nn_postprocess = lambda labels: denormalizer(labels)


# Only check the size of the non-image input vector if it has any entries
vector_input_size = 0
if 0 < len(args.vector_inputs):
    vector_input_size = getVectorSize(args.dataset, decode_strs, vector_range)

# Decode the proper number of items for each sample from the dataloader
# The field names are just being taken from the decode strings, but they cannot begin with a digit
# or contain the '.' character, so the character "f" is prepended to each string and the '.' is
# replaced with a '_'. The is a bit egregious, but it does guarantee that the tuple being used to
# accept the output of the dataloader matches what is being used in webdataset decoding.
LoopTuple = namedtuple('LoopTuple', ' '.join(["f" + s for s in decode_strs]).replace('.', '_'))
dl_tuple = LoopTuple(*([None] * len(decode_strs)))

# TODO FIXME Deterministic shuffle only shuffles within a range. Should perhaps manipulate what is
# in the tar file by shuffling filenames after the dataset is created.
dataset = (
    wds.WebDataset(args.dataset, shardshuffle=True)
    .shuffle(20000//in_frames, initial=20000//in_frames)
    # TODO This will hardcode all images to single channel numpy float images, but that isn't clear
    # from any documentation.
    # TODO Why "l" rather than decode to torch directly with "torchl"?
    .decode("l")
    .to_tuple(*decode_strs)
)

image_size = getImageSize(args.dataset, decode_strs)
print(f"Decoding images of size {image_size}")

batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size)

if args.evaluate:
    eval_dataset = (
        wds.WebDataset(args.evaluate)
        .decode("l")
        .to_tuple(*decode_strs)
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=0, batch_size=batch_size)


lr_scheduler = None
# AMP doesn't seem to like all of the different model types, so disable it unless it has been
# verified.
use_amp = False
# Store the model arguments and save them with the model. This will simplify model loading and
# recreation later.
model_args = {
    'in_dimensions': (in_frames, image_size[1], image_size[2]),
    'out_classes': label_handler.size(),
}
if 0 < vector_input_size:
    model_args['vector_input_size'] = vector_input_size

if 'alexnet' == args.modeltype:
    # Model specific arguments
    model_args['linear_size'] = 512
    model_args['skip_last_relu'] = True
    net = AlexLikeNet(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,7], gamma=0.2)
    use_amp = True
elif 'resnet18' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    net = ResNet18(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-5)
elif 'resnet34' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    net = ResNet34(**model_args).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
elif 'bennet' == args.modeltype:
    net = BenNet(**model_args).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-5)
elif 'resnext50' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    net = ResNext50(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3], gamma=0.1)
    batch_size = 64
elif 'resnext34' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = False
    model_args['use_dropout'] = False
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext34(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,9], gamma=0.2)
elif 'resnext18' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    model_args['use_dropout'] = False
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext18(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextxt' == args.modeltype:
    net = ConvNextExtraTiny(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-4, weight_decay=10e-4, momentum=0.9,
            nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5,12], gamma=0.2)
    use_amp = True
elif 'convnextt' == args.modeltype:
    net = ConvNextTiny(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnexts' == args.modeltype:
    net = ConvNextSmall(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextb' == args.modeltype:
    net = ConvNextBase(**model_args).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=10e-2, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
print(f"Model is {net}")

# See if the model weights and optimizer state should be restored.
if args.resume_from is not None:
    restoreModelAndState(args.resume_from, net, optimizer)


# TODO(bfirner) Read class names from something instead of assigning them numbers.
# Note that we can't just use the label names since we may be getting classes by converting a
# numeric input into a one-hot vector
class_names = []
for i in range(label_size):
    class_names.append(f"{i}")

if not args.no_train:
    if args.save_worst_n is not None:
        print(f"Saving {args.save_worst_n} highest error training images to {worst_training.worstn_path}.")

    # Gradient scaler for mixed precision training
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    for epoch in range(args.epochs):
        totals = RegressionResults(size=label_handler.size(), names=label_handler.names())
        worst_training = None
        if args.save_worst_n:
            worst_training = WorstExamples(
                args.outname.split('.')[0] + "-worstN-train", class_names, save_worst_n)
            print(f"Saving {save_worst_n} highest error training images to {worst_training.worstn_path}.")
        print(f"Starting epoch {epoch}")
        trainEpoch(net=net, optimizer=optimizer, scaler=scaler, label_handler=label_handler,
                train_stats=totals, dataloader=dataloader, vector_range=vector_range, train_frames=in_frames,
                normalize_images=args.normalize, loss_fn=loss_fn, nn_postprocess=nn_postprocess,
                worst_training=worst_training, skip_metadata=True)
        # Adjust learning rate according to the learning rate schedule
        if lr_scheduler is not None:
            lr_scheduler.step()
        print(f"Finished training epoch {epoch}")

        torch.save({
            "model_dict": net.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "py_random_state": random.getstate(),
            "np_random_state": numpy.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "denormalizer_state_dict": denormalizer.state_dict() if denormalizer is not None else None,
            "normalizer_state_dict": normalizer.state_dict() if normalizer is not None else None,
            # Store some metadata to make it easier to recreate and use this model
            "metadata": {
                'modeltype': args.modeltype,
                'labels': args.labels,
                'vector_inputs': args.vector_inputs,
                'convert_idx_to_classes': False,
                'label_size': label_handler.size(),
                'model_args': model_args,
                'normalize_images': args.normalize,
                'normalize_labels': args.normalize_outputs,
                },
            }, args.outname)

        # Validation step if requested
        if args.evaluate is not None:
            print(f"Evaluating epoch {epoch}")
            eval_totals = RegressionResults(size=label_handler.size(), names=label_handler.names())
            evalEpoch(net=net, label_handler=label_handler, eval_stats=eval_totals,
                    eval_dataloader=eval_dataloader, vector_range=eval_range, train_frames=in_frames,
                    normalize_images=args.normalize, loss_fn=loss_fn, nn_postprocess=nn_postprocess)



# TODO FIXME Move this evaluation step into the evaluation function as well.

# Post-training evaluation
if args.evaluate is not None:
    top_eval = None
    worst_eval = None
    print("Evaluating model.")
    if args.save_top_n is not None:
        top_eval = WorstExamples(
            args.outname.split('.')[0] + "-topN-eval", class_names, args.save_top_n,
            worst_mode=False)
        print(f"Saving {args.save_top_n} highest error evaluation images to {top_eval.worstn_path}.")
    if args.save_worst_n is not None:
        worst_eval = WorstExamples(
            args.outname.split('.')[0] + "-worstN-eval", class_names, args.save_worst_n)
        print(f"Saving {args.save_worst_n} highest error evaluation images to {worst_eval.worstn_path}.")

    net.eval()
    with torch.no_grad():
        # Make a confusion matrix or loss statistics
        totals = RegressionResults(size=label_size)
        with open(args.outname.split('.')[0] + ".log", 'w') as logfile:
            logfile.write('video_path,frame,time,label,prediction\n')
            for batch_num, dl_tuple in enumerate(eval_dataloader):
                # Decoding only the luminance channel means that the channel dimension has gone away here.
                if 1 == in_frames:
                    net_input = dl_tuple[0].unsqueeze(1).cuda()
                else:
                    raw_input = []
                    for i in range(in_frames):
                        raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                    net_input = torch.cat(raw_input, dim=1)
                # Normalize inputs: input = (input - mean)/stddev
                if args.normalize:
                    v, m = torch.var_mean(net_input)
                    net_input = (net_input - m) / v

                vector_input=None
                if 0 < vector_input_size:
                    vector_input = extractVectors(dl_tuple, vector_range).cuda()

                # Visualization masks are not supported with all model types yet.
                if args.modeltype in ['alexnet', 'bennet', 'resnet18', 'resnet34']:
                    out, mask = net.vis_forward(net_input, vector_input)
                else:
                    out = net.forward(net_input, vector_input, vector_input)
                    mask = [None] * batch_size

                # Convert the labels to a one hot encoding to serve at the DNN target.
                # The label class is 1 based, but need to be 0-based for the one_hot function.
                labels = extractVectors(dl_tuple,label_range).cuda()

                metadata = [""] * labels.size(0)

                loss = loss_fn(out, label_handler.preprocess(labels))

                # Fill in the loss statistics and best/worst examples
                with torch.no_grad():
                    # The postprocessesing could include Softmax, denormalization, etc.
                    post_out = nn_postprocess(out)
                    # Labels may also require postprocessing, for example to convert to a one-hot
                    # encoding.
                    post_labels = label_handler.preeval(labels)

                    # Log the predictions
                    for i in range(post_labels.size(0)):
                        logfile.write(','.join((metadata[i], str(out[i]), str(post_labels[i]))))
                        logfile.write('\n')
                    if worst_eval is not None or top_eval is not None:
                        # For each item in the batch see if it requires an update to the worst examples
                        # If the DNN should have predicted this image was a member of the labelled class
                        # then see if this image should be inserted into the worst_n queue for the
                        # labelled class based upon the DNN output for this class.
                        input_images = dl_tuple[0]
                        for i in range(post_labels.size(0)):
                            label = torch.argwhere(post_labels[i])[0].item()
                            if worst_eval is not None:
                                worst_eval.test(label, out[i][label].item(), input_images[i], metadata[i])
                            if top_eval is not None:
                                top_eval.test(label, out[i][label].item(), input_images[i], metadata[i])

        # Save the worst and best examples
        if worst_eval is not None:
            worst_eval.save("evaluation")
        if top_eval is not None:
            top_eval.save("evaluation")

        # Print evaluation information
        print(f"Evaluation results:")
        print(totals.makeResults())
