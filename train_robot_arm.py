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

from arm_utility import (grepGripperLocationFromTensors, RTZClassifierNames)

from embedded_perturbation import (
        drawNewFeatures, findMultivariateParameters,
        generatePerturbedXYZ, makeSpherePoints)

# Insert the bee analysis repository into the path so that the python modules of the git submodule
# can be used properly.
import sys
sys.path.append('bee_analysis')

from bee_analysis.utility.dataset_utility import (decodeUTF8Strings, extractVectors, getImageSize, getVectorSize, makeDataset)
from bee_analysis.utility.eval_utility import (OnlineStatistics, RegressionResults, WorstExamples)
from bee_analysis.utility.flatbin_dataset import FlatbinDataset
from bee_analysis.utility.model_utility import (restoreModelAndState)
from bee_analysis.utility.train_utility import (LabelHandler, evalEpoch, trainEpoch,
        updateWithoutScalerOriginal)

from bee_analysis.models.alexnet import AlexLikeNet
from bee_analysis.models.bennet import BenNet, CompactingBenNet
from bee_analysis.models.dragonfly import DFNet
from bee_analysis.models.modules import Denormalizer, Normalizer
from bee_analysis.models.resnet import (ResNet18, ResNet34)
from bee_analysis.models.resnext import (ResNext18, ResNext34, ResNext50)
from bee_analysis.models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)

################ For memory debugging
#import linecache
#import tracemalloc
#
#def display_top(snapshot, key_type='lineno', limit=10):
#    snapshot = snapshot.filter_traces((
#        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#        tracemalloc.Filter(False, "<unknown>"),
#    ))
#    top_stats = snapshot.statistics(key_type)
#
#    print("Top %s lines" % limit)
#    for index, stat in enumerate(top_stats[:limit], 1):
#        frame = stat.traceback[0]
#        print("#%s: %s:%s: %.1f KiB"
#              % (index, frame.filename, frame.lineno, stat.size / 1024))
#        line = linecache.getline(frame.filename, frame.lineno).strip()
#        if line:
#            print('    %s' % line)
#
#    other = top_stats[limit:]
#    if other:
#        size = sum(stat.size for stat in other)
#        print("%s other: %.1f KiB" % (len(other), size / 1024))
#    total = sum(stat.size for stat in top_stats)
#    print("Total allocated size: %.1f KiB" % (total / 1024))
#
#tracemalloc.start()
################ For memory debugging

# Argument parser setup for the program.
parser = argparse.ArgumentParser(
    description="Perform data preparation for DNN training on a video set.")
parser.add_argument(
    'dataset',
    nargs='+',
    type=str,
    help='Datasets for training. All should either be .tar (webdataset) files or .bin (flatbin) files.')
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
    help=("Normalize image inputs: input = (input - mean) / stddev. "
        "Note that VidActRecDataprep is already normalizing so this may not be required."))
parser.add_argument(
    '--normalize_status',
    required=False,
    default=False,
    action="store_true",
    help=("Normalize status inputs by adjusting DNN weight and bias values (only works with some models)."))
parser.add_argument(
    '--normalize_outputs',
    required=False,
    default=False,
    action="store_true",
    help=("Normalize the outputs: output = (output - mean) / stddev. "
        "This will read the entire dataset to find these values, delaying initial training."))
parser.add_argument(
    '--modeltype',
    type=str,
    required=False,
    default="resnext18",
    choices=["alexnet", "resnet18", "resnet34", "bennet", "compactingbennet", "dragonfly", "resnext50", "resnext34", "resnext18",
    "convnextxt", "convnextt", "convnexts", "convnextb"],
    help="Model to use for training.")
parser.add_argument(
    '--lr',
    type=float,
    required=False,
    default=None,
    help="Learning rate, or the default for the model if None.")
parser.add_argument(
    '--no_train',
    required=False,
    default=False,
    action='store_true',
    help='Set this flag to skip training. Useful to load an already trained model for evaluation.')
parser.add_argument(
    '--feature_perturbations',
    required=False,
    default=False,
    action="store_true",
    help=("Enable feature-space training using feature-space perturbations."))
parser.add_argument(
    '--evaluate',
    type=str,
    required=False,
    default=None,
    help='Evaluate with the given dataset. Should be a .tar or a .bin dataset')
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
    '--batch_size',
    type=int,
    required=False,
    default=32,
    help='Batch size.')
parser.add_argument(
    '--deterministic',
    required=False,
    default=False,
    action='store_true',
    help='Set this to disable deterministic training.')
parser.add_argument(
    '--encode_position',
    required=False,
    default=False,
    action='store_true',
    help='Encode the pixel positions with another image channel.')
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
    choices=['L1Loss', 'MSELoss', 'BCEWithLogitsLoss'],
    type=str,
    help="Loss function to use during training.")
parser.add_argument(
    '--epochs_to_lr_decay',
    type=int,
    required=False,
    default=90,
    help='Epochs at the initial learning rate before learning rate decay (when using SGD). Default is 90.')

args = parser.parse_args()


torch.use_deterministic_algorithms(args.deterministic)
torch.manual_seed(args.seed)
random.seed(0)
numpy.random.seed(0)

# Later on we will need to change behavior if the loss function is regression rather than
# classification
regression_loss = ['L1Loss', 'MSELoss']
# Output normalization obviously shouldn't be used with one hot vectors and classifiers
if args.normalize_outputs and args.loss_fun not in regression_loss:
    print("Error: normalize_outputs should only be true for regression loss.")
    exit(1)

in_frames = args.sample_frames
decode_strs = []
label_decode_strs = []
vector_decode_strs = []
# The image for a particular frame
for i in range(in_frames):
    decode_strs.append(f"{i}.png")

# The class label(s) or regression targets
label_range = slice(len(decode_strs), len(decode_strs) + len(args.labels))
for label_str in args.labels:
    decode_strs.append(label_str)
    label_decode_strs.append(label_str)

# Vector inputs (if there are none then the slice will be an empty range)
vector_range = slice(len(decode_strs), len(decode_strs) + len(args.vector_inputs))
for vector_str in args.vector_inputs:
    decode_strs.append(vector_str)
    vector_decode_strs.append(vector_str)


# The loss will be calculated from the distance rather than the joint positions, but if they are not
# in the labels we use the loss directly.
loss_fn = getattr(torch.nn, args.loss_fun)(reduction='sum')

def computeDistanceForLoss(tensor_a, tensor_b):
    # The distance is the square root of the sum of the squares of the differences in the
    # coordinates. The tensors have a batch dimension first, so sum along the second dimension.
    diff = grepGripperLocationFromTensors(tensor_b) - grepGripperLocationFromTensors(tensor_a)
    squares = torch.pow(diff, 2)
    sums = torch.sum(squares, dim=1, keepdim=True)
    distance = torch.sqrt(sums)

    return distance

def lossWithDistance(output, labels, loss_fn, joint_range):
    # TODO Balancing loss with something that has multiple joints is nontrivial
    # This function will remain in case a good solution is found.
    return loss_fn(output, labels)

    # Add the distance loss values to the regular loss values.
    #distances = computeDistanceForLoss(output[:,joint_range], labels[:,joint_range])
    #distance_error = loss_fn(torch.zeros(distances.size()).cuda(), distances)
    #loss = distance_error + loss_fn(output, labels)
    #return loss

    # If there are labels before or after the joint positions then add them into the loss along with
    # the distance error
    #loss = distance_error
    #if 0 < joint_range.start:
    #    before_output = output[:, 0:joint_range.start]
    #    before_labels = labels[:, 0:joint_range.start]
    #    loss = torch.cat((loss, loss_fn(before_output, before_labels)))
    #if joint_range.stop < labels.size(1):
    #    after_output = output[:, joint_range.stop:]
    #    after_labels = labels[:, joint_range.stop:]
    #    loss = torch.cat((loss, loss_fn(after_output, after_labels)))
    #return torch.sum(loss)


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
    # TODO Not being used, not because it isn't a good idea, but because it is very difficult to
    # figure out how to balance loss across multiple joints of a robot arm
    # Change the loss function to accommodate calculating distance from the joints
    #if 'target_arm_position' == args.labels[label_idx]:
    #    joint_range = slice(len(label_names), len(label_names) + label_elements)
    #    loss_fn = functools.partial(lossWithDistance, loss_fn=getattr(torch.nn, args.loss_fun)(), joint_range=joint_range)

    # Give this output the label name directly or add a number if multiple outputs come from
    # this label
    if 1 == label_elements:
        label_names.append(args.labels[label_idx])
    else:
        if args.labels[label_idx] == 'rtz_classifier':
            classifier_names = RTZClassifierNames()
            for i in range(label_elements):
                label_names.append("{}".format(classifier_names[i]))
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
    label_dataset =  makeDataset(args.dataset, label_decode_strs)
    #label_dataset = (
    #    wds.WebDataset(args.dataset)
    #    .to_tuple(*label_decode_strs)
    #)
    # Loop through the dataset and compile label statistics
    #label_dataloader = torch.utils.data.DataLoader(label_dataset, num_workers=0, batch_size=1)
    #for data in label_dataloader:
    for data in label_dataset:
        flat_data = torch.cat([torch.tensor(datum) for datum in data], dim=0)
        for label, stat in zip(flat_data.tolist(), label_stats):
            stat.sample(label)
    # Clean up the open files from dataloading
    del label_dataset
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

label_handler = LabelHandler(label_size=label_size, label_range=label_range, label_names=label_names)

# The label value may need to be adjusted, to normalize before calculating loss.
if normalizer is not None:
    label_handler.setPreprocess(lambda labels: normalizer(labels))

# Network outputs may need to be postprocessed for evaluation if some postprocessing is being done
# automatically by the loss function.
# Use an identify function unless normalization is being used.
if 'BCEWithLogitsLoss' == args.loss_fun:
    nn_postprocess = torch.nn.Sigmoid()
    # Note that a classifier task could not have used normalization
elif denormalizer is None:
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
# TODO FIXME Yes, remove shuffling here, shuffle them outside of training. This will solve
# speed issues, memory issues, and issues with sample correlation.
# Decode directly to torch memory
# TODO Use use .map_tuple to preprocess samples during the dataloading
channels = 1
image_decode_str = "torchl" if 1 == channels else "torchrgb"
dataset = makeDataset(args.dataset, decode_strs)
#dataset = (
#    #wds.WebDataset(args.dataset, shardshuffle=True)
#    #.shuffle(20000//in_frames, initial=20000//in_frames)
#    wds.WebDataset(args.dataset, shardshuffle=False)
#    .shuffle(1000//in_frames, initial=1000//in_frames)
#    # TODO This will hardcode all images to single channel numpy float images, but that isn't clear
#    # from any documentation.
#    # TODO Why "l" rather than decode to torch directly with "torchl"?
#    .decode(image_decode_str)
#    .to_tuple(*decode_strs)
#)

image_size = getImageSize(args.dataset, decode_strs)
print(f"Decoding images of size {image_size}")

batch_size = args.batch_size
#dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=0, batch_size=None)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=batch_size,
        pin_memory=True, drop_last=False, persistent_workers=False)

if args.evaluate:
    eval_dataset = makeDataset(args.evaluate, decode_strs)
    #eval_dataset = (
    #    wds.WebDataset(args.evaluate)
    #    .decode(image_decode_str)
    #    .to_tuple(*decode_strs)
    #)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, num_workers=0, batch_size=batch_size)


lr_scheduler = None
# AMP doesn't seem to like all of the different model types, so disable it unless it has been
# verified.
use_amp = False
# Store the model arguments and save them with the model. This will simplify model loading and
# recreation later.
in_channels = in_frames
if args.encode_position:
    in_channels += 1
model_args = {
    'in_dimensions': (in_channels, image_size[1], image_size[2]),
    'out_classes': label_handler.size(),
}
# True to use a lower learning rate for the first epoch
warmup_epoch = True
if 0 < vector_input_size:
    model_args['vector_input_size'] = vector_input_size

if 'alexnet' == args.modeltype:
    # Model specific arguments
    model_args['linear_size'] = 512
    model_args['skip_last_relu'] = True
    net = AlexLikeNet(**model_args).cuda()
    lr = 10e-4 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,7], gamma=0.2)
    use_amp = True
elif 'resnet18' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    net = ResNet18(**model_args).cuda()
    lr = 10e-5 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
elif 'resnet34' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    net = ResNet34(**model_args).cuda()
    lr = 10e-5 if args.lr is None else args.lr
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
elif 'bennet' == args.modeltype:
    net = BenNet(**model_args).cuda()
    lr = 0.001 if args.lr is None else args.lr
    #optimizer = torch.optim.AdamW(net.parameters(), lr=10e-5)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=0.001, nesterov=True)
    #milestones = [args.epochs_to_lr_decay, args.epochs_to_lr_decay+20, args.epochs_to_lr_decay+60]
    milestones = [args.epochs_to_lr_decay, args.epochs_to_lr_decay+20]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
elif 'compactingbennet' == args.modeltype:
    net = CompactingBenNet(**model_args).cuda()
    #optimizer = torch.optim.AdamW(net.parameters(), lr=10e-5)
    #optimizer = torch.optim.Adam(net.parameters(), lr=10e-3)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.008, momentum=0.5, weight_decay=0.001, nesterov=True)
    # Tuned with args.normalize_outputs
    lr = 10e-5 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=0.001, nesterov=True)
    milestones = [args.epochs_to_lr_decay, args.epochs_to_lr_decay+20]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
elif 'dragonfly' == args.modeltype:
    net = DFNet(**model_args).cuda()
    #optimizer = torch.optim.SGD(net.parameters(), lr=10e-5, momentum=0.5, weight_decay=0.001, nesterov=True)
    lr = 0.008 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5, weight_decay=0.001, nesterov=True)
    milestones = [args.epochs_to_lr_decay, args.epochs_to_lr_decay+20]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
elif 'resnext50' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    net = ResNext50(**model_args).cuda()
    lr = 10e-2 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2,3], gamma=0.1)
    batch_size = 64
elif 'resnext34' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = False
    model_args['use_dropout'] = False
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext34(**model_args).cuda()
    lr = 10e-2 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,9], gamma=0.2)
elif 'resnext18' == args.modeltype:
    # Model specific arguments
    model_args['expanded_linear'] = True
    model_args['use_dropout'] = False
    # Learning parameters were tuned on a dataset with about 80,000 examples
    net = ResNext18(**model_args).cuda()
    lr = 10e-2 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextxt' == args.modeltype:
    net = ConvNextExtraTiny(**model_args).cuda()
    lr = 10e-4 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=10e-4, momentum=0.9,
            nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,5,12], gamma=0.2)
    use_amp = True
elif 'convnextt' == args.modeltype:
    net = ConvNextTiny(**model_args).cuda()
    lr = 10e-2 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnexts' == args.modeltype:
    net = ConvNextSmall(**model_args).cuda()
    lr = 10e-2 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
elif 'convnextb' == args.modeltype:
    net = ConvNextBase(**model_args).cuda()
    lr = 10e-2 if args.lr is None else args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=10e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,12], gamma=0.2)
print(f"Model is {net}")

# Normalize status vector input by adjust the weight and bias in the linear layer.
# Not all networks support this feature and there is no reason to normalize if the weights and bias
# values will be restored from a checkpoint.
if 0 < vector_input_size and args.normalize_status and not args.resume_from:
    print("Reading the dataset to compute vector input statistics for normalization.")
    vector_stats = [OnlineStatistics() for _ in range(vector_input_size)]
    vector_dataset = makeDataset(args.dataset, vector_decode_strs)
    print("The vector decode strings are {}".format(vector_decode_strs))
    #vector_dataset = (
    #    wds.WebDataset(args.dataset)
    #    .to_tuple(*vector_decode_strs)
    #)
    # TODO Loop through the dataset and compile vector inputs statistics
    #vector_dataloader = torch.utils.data.DataLoader(vector_dataset, num_workers=0, batch_size=1)
    #for data in vector_dataloader:
    for data in vector_dataset:
        flat_data = torch.cat([torch.tensor(datum) for datum in data], dim=0)
        for vector, stat in zip(flat_data.tolist(), vector_stats):
            stat.sample(vector)
    # Clean up the open files from dataloading
    del vector_dataset
    # Now record the statistics
    vector_means = []
    vector_stddevs = []
    for stat in vector_stats:
        vector_means.append(stat.mean())
        # If we've chosen to train with something that has only a single value then it blows up
        # here. That is okay, you shouldn't be training with constant values.
        vector_stddevs.append(math.sqrt(stat.variance()))

    print("Normalizing status vector inputs with means {} and stddevs {}".format(vector_means, vector_stddevs))

    if hasattr(net, "normalizeVectorInputs") and callable(getattr(net, "normalizeVectorInputs")):
        net.normalizeVectorInputs(vector_means, vector_stddevs)
else:
    print("normalize_status set to true, but chosen model architecture {} does not support that feature.".format(args.modeltype))

if warmup_epoch:
    for group in optimizer.param_groups:
        group['lr'] *= 0.1

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
    # Gradient scaler for mixed precision training
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Save models when the error is minimized
    min_err = 1000
    for epoch in range(args.epochs):
        ########## For memory debugging
        #snapshot = tracemalloc.take_snapshot()
        #display_top(snapshot)
        #top_stats = snapshot.statistics('lineno')
        #print("[ Top 10 ]")
        #for stat in top_stats[:10]:
        #    print(stat)
        ########## For memory debugging

        totals = RegressionResults(size=label_handler.size(), names=label_handler.names())
        worst_training = None
        if args.save_worst_n:
            worst_training = WorstExamples(
                args.outname.split('.')[0] + "-worstN-train", class_names, args.save_worst_n)
            print(f"Saving {args.save_worst_n} highest error training images to {worst_training.worstn_path}.")
        print(f"Starting epoch {epoch}")
        trainEpoch(net=net, optimizer=optimizer, scaler=scaler, label_handler=label_handler,
                train_stats=totals, dataloader=dataloader, vector_range=vector_range, train_frames=in_frames,
                normalize_images=args.normalize, loss_fn=loss_fn, nn_postprocess=nn_postprocess,
                encode_position=args.encode_position, worst_training=worst_training, skip_metadata=True)
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
                'encode_position': args.encode_position,
                },
            }, args.outname)

        # If we were doing a warmup epoch then revert to the original learning rate now.
        if 0 == epoch and warmup_epoch:
            for group in optimizer.param_groups:
                group['lr'] *= 10

        # Validation step if requested
        if args.evaluate is not None:
            print(f"Evaluating epoch {epoch}")
            eval_totals = RegressionResults(size=label_handler.size(), names=label_handler.names())
            evalEpoch(net=net, label_handler=label_handler, eval_stats=eval_totals,
                    eval_dataloader=eval_dataloader, vector_range=vector_range, train_frames=in_frames,
                    normalize_images=args.normalize, loss_fn=loss_fn, nn_postprocess=nn_postprocess)

        if abs(totals.mean()) < min_err:
            min_err = abs(totals.mean())
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
                        'encode_position': args.encode_position,
                    },
                }, (args.outname + ".epoch_{}".format(epoch)))


# Feature-space training
if args.feature_perturbations and 'target_xyz_position' in args.labels:
    # TODO Assuming that xyz position is the only DNN output
    if "target_xyz_position" in args.labels:
        dnn_output_to_xyz = lambda out: out[0].tolist()
    if "current_xyz_position" not in args.vector_inputs:
        decode_strs.append('current_xyz_position')

    # Loop through the entire training set and calculate the mean values and correlation
    # coefficients.
    # TODO
    net.train()

    feature_dataset = (
        wds.WebDataset(args.dataset, shardshuffle=False)
        .shuffle(20000//in_frames, initial=20000//in_frames)
        .decode("l")
        .to_tuple(*decode_strs)
    )

    image_size = getImageSize(args.dataset, decode_strs)
    print(f"Decoding images of size {image_size}")

    # Use small batches so that the feature fetching step doesn't use up all of the available memory
    dataloader = torch.utils.data.DataLoader(feature_dataset, num_workers=0, batch_size=1)

    # The features and labels that will be used to create the correlation coefficient laters
    flat_features_and_gt = []
    # The target positions will be used to calculate perturbed positions, so store them as well
    target_positions = []
    cur_pos_idx = decode_strs.index("current_xyz_position")
    target_pos_idx = decode_strs.index("target_xyz_position")
    for batch_num, dl_tuple in enumerate(dataloader):
        # The image is always the first item
        image = dl_tuple[0].unsqueeze(1).cuda()

        # The labels aren't actually used in this pass
        labels = extractVectors(dl_tuple, label_handler.range()).cuda()
        vector_inputs=None
        if vector_range.start != vector_range.stop:
            vector_inputs = extractVectors(dl_tuple, vector_range).cuda()

        # Normalize inputs: input = (input - mean)/stddev
        if args.normalize:
            v, m = torch.var_mean(image)
            image = (image - m) / v

        # Get the network output
        with torch.no_grad():
            if 0 < len(args.vector_inputs):
                output = net(image, vector_inputs.cuda())
            else:
                output = net(image)

            if denormalizer is not None:
                output = denormalizer(output)

            # Fetch the feature maps
            features = net.forwardToFeatures(image)

            # The current positions need to be stored with their features
            cur_positions = decodeUTF8Strings(dl_tuple[cur_pos_idx:cur_pos_idx+1])[0]
            batch_target_positions = decodeUTF8Strings(dl_tuple[target_pos_idx:target_pos_idx+1])[0]

            # Store the features of each item in the batch
            for b_index in range(features.size(0)):
                # TODO There's no reason to convert from tensors when these will just go back into
                # tensors
                flat_features = features[b_index].tolist()
                # Store the flattened last layer in the output string if requested
                # TODO should we use the dnn xyz prediction here, not the true xyz, with the theory
                # that these are more interpretable by the DNN itself? This would require a current
                # xyz prediction, as well the target xyz prediction.
                #dnn_xyz = output[b_index].tolist()
                target_positions.append(batch_target_positions[b_index].tolist())
                flat_features_and_gt.append(flat_features + cur_positions[b_index].tolist())
            # Aggresively clear this from memory in case it is large
            del features

    # TODO The hard-coded 3 here assumes that the xyz values are the only labels
    label_size = 3

    # Now find the correlation coefficients and the means of the different features and labels
    features_and_gt_tensor = torch.tensor(flat_features_and_gt)
    print("The features and label tensor has size {}".format(features_and_gt_tensor.size()))

    # TODO FIXME Verify that the source and target positions are correct

    # The rotated features_and_gt_tensor is needed to easily take the mean and for the corrcoef
    # function, which requires variables in rows and observations in columns
    #rot_features_and_labels = features_and_gt_tensor.rot90(k=1, dims=[1,0])
    rot_features_and_gt = features_and_gt_tensor.T
    means = rot_features_and_gt.mean(dim=1)
    feature_means = means[:-label_size]
    gt_means = means[-label_size:]
    #correlations = rot_features_and_gt.corrcoef()
    
    # Compute the sample covariance matrix
    sample_cov_matrix = rot_features_and_gt.cov()
    print("The covariance matrix is:")
    for row in range(sample_cov_matrix.size(0)):
        print(sample_cov_matrix[row])

    # Treat the target positions similary to the rest of the data, placing variables in rows and
    # observations in columns
    target_positions_tensor = torch.tensor(target_positions)

    pert_optimizer = torch.optim.SGD(net.classifier.parameters(), lr=0.002, momentum=0.0, weight_decay=0.000, nesterov=False)

    # With the means and sample covariance matrix, go back through another training loop
    # This time we won't load any images, just the source state and the target state.
    # For each batch, train with some of the original values and add in perturbations as well.
    # The features and labels are already loaded, so simply go through go through the columns of
    # data as batches.
    batch_size = 64
    for epoch in range(1):
        for observation_begin in range(0, features_and_gt_tensor.size(0), batch_size):
            observation_end = min(features_and_gt_tensor.size(0), observation_begin + batch_size)

            # The original training features
            original_features = features_and_gt_tensor[observation_begin:observation_end, :-label_size]
            # Call generatePerturbedXYZ to generate new training points that are in an orthogonal
            # direction from the begin->end vector
            begin_xyzs = features_and_gt_tensor[observation_begin:observation_end, -label_size:]
            end_xyzs = target_positions_tensor[observation_begin:observation_end]

            distances = torch.pow(end_xyzs - begin_xyzs, 2).sum(dim=1, keepdim=False).sqrt()

            # Create new training targets
            # Then use findMultivariateParameters to find the distribution for features with the new
            # training targets
            new_end_xyzs = []
            new_features = []
            # The number of draws to perform for each generates xyz location
            new_data_draws = 1
            new_begin_xyzs = []
            for idx in range(begin_xyzs.size(0)):
                # Generate a new xyz that maintains the same distance to the endpoint.
                #step_size = distances[idx].item()/math.sqrt(2)
                # TODO experimenting with a really large step
                #step_size = 3*distances[idx].item() + 0.001
                step_size = random.uniform(1,3)*distances[idx].item() + 0.001
                new_begin_xyz = torch.tensor(generatePerturbedXYZ(begin_xyzs[idx], end_xyzs[idx], step_size))
                # TODO Testing with spheres
                #new_begin_xyz = makeSpherePoints(point_means=gt_means,
                #    sample_points=begin_xyzs[idx:idx+1], start_radius=0.015)[0]
                # Calculate the mean vector and covariance matrix of the features dependent upon this
                # new begin location
                mean_vector, cov_matrix = findMultivariateParameters(
                    new_begin_xyz, sample_cov_matrix, feature_means, gt_means)

                # Generate new_data_draws random samples from the distribution
                # TODO FIXME Just trying the means for now
                drawn_features = drawNewFeatures(mean_vector, cov_matrix, num_draws=new_data_draws)
                for draw in range(new_data_draws):
                    new_features.append(drawn_features[draw:draw+1])
                    #new_features.append(feature_means.expand(1, -1))
                    new_begin_xyzs.append(new_begin_xyz.expand(1, -1))
                    # The end target remains the same
                    new_end_xyzs.append(end_xyzs[idx].expand(1, -1))

            # Combine the original and perturbed features and targets to create the training batch
            training_labels = torch.cat((end_xyzs, *new_end_xyzs), dim=0)
            training_start_xyzs = torch.cat((begin_xyzs, *new_begin_xyzs), dim=0)
            new_features = torch.cat(new_features, dim=0)
            training_features = torch.cat((original_features, new_features), dim=0)

            # For debugging
            for label_idx in range(training_labels.size(0)):
                print("pcoords" + (", {}"*6).format(*training_start_xyzs[label_idx].tolist(),
                    *training_labels[label_idx].tolist()))

            # TODO FIXME Vector inputs

            # If the current xyz position is also a training target, add it into the labels tensor.
            if "current_xyz_position" in args.labels:
                # repeat_interleave the current positions into the labels as well.
                training_labels = torch.cat(
                    (training_labels,
                        torch.cat((begin_xyzs, *new_begin_xyzs), dim=0)), dim=1)

            # Send the new batches forward and backward.
            out, loss = updateWithoutScalerOriginal(loss_fn=loss_fn, net=net.classifier,
                    image_input=training_features.cuda(), vector_input=None,
                    labels=label_handler.preprocess(training_labels.cuda()), optimizer=pert_optimizer)
            # TODO FIXME record loss, etc. Make sure that things are stable.
            print("loss at observation {} is {}".format(observation_begin, loss))
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
        }, args.outname + ".perturbed")


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
                    # Normalize per channel, so compute over height and width
                    v, m = torch.var_mean(net_input, dim=(2,3), keepdim=True)
                    net_input = (net_input - m) / v

                vector_input=None
                if 0 < vector_input_size:
                    vector_input = extractVectors(dl_tuple, vector_range).cuda()

                # Visualization masks are not supported with all model types yet.
                if args.modeltype in ['alexnet', 'bennet', 'compactingbennet', 'resnet18', 'resnet34']:
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

