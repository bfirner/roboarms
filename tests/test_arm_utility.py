#!/usr/bin/env python3

# Test the arm_utility
# In particular, algorithms as in rSolver that are integral to correct robot operation

import pytest

import arm_utility
import math


def test_rSolver():
    # Attempt to generate a pose for a nearby target
    target = [0.23115182452192629, 0.0, 0.03520485992485018]
    segment_lengths = [0.104, 0.158, 0.147, 0.175]
    target_rtz = arm_utility.XYZToRThetaZ(*target)
    new_joints = arm_utility.rSolver(target_rtz[0], target_rtz[2], segment_lengths=segment_lengths)

    assert abs(new_joints[0]) < math.pi/3.
    assert abs(new_joints[1]) < math.pi/3.
    assert abs(new_joints[2]) < math.pi/3.

    new_coords = arm_utility.computeGripperPosition([0.] + new_joints, segment_lengths)
    distance = arm_utility.getDistance(new_coords, target)
    assert distance < 0.001

def test_rSolver():
    # Attempt to generate a pose for a distant target
    target = [0.3, 0.0, 0.03520485992485018]
    segment_lengths = [0.104, 0.158, 0.147, 0.175]
    target_rtz = arm_utility.XYZToRThetaZ(*target)
    new_joints = arm_utility.rSolver(target_rtz[0], target_rtz[2], segment_lengths=segment_lengths)

    assert abs(new_joints[0]) < math.pi/3.
    assert abs(new_joints[1]) < math.pi/3.
    assert abs(new_joints[2]) < math.pi/3.

    new_coords = arm_utility.computeGripperPosition([0.] + new_joints, segment_lengths)
    distance = arm_utility.getDistance(new_coords, target)
    assert distance < 0.001

def test_rSolver():
    # Attempt to generate a pose for a high target
    target = [0.2, 0.0, 0.3]
    segment_lengths = [0.104, 0.158, 0.147, 0.175]
    target_rtz = arm_utility.XYZToRThetaZ(*target)
    new_joints = arm_utility.rSolver(target_rtz[0], target_rtz[2], segment_lengths=segment_lengths)

    assert abs(new_joints[0]) < math.pi/3.
    assert abs(new_joints[1]) < math.pi/3.
    assert abs(new_joints[2]) < math.pi/3.

    new_coords = arm_utility.computeGripperPosition([0.] + new_joints, segment_lengths=segment_lengths)
    distance = arm_utility.getDistance(new_coords, target)
    assert distance < 0.001

def test_rSolver():
    # Attempt to generate a pose for a near target with a longer gripper
    target = [0.25, 0.0, 0.05]
    segment_lengths = [0.104, 0.158, 0.147, 0.3]
    target_rtz = arm_utility.XYZToRThetaZ(*target)
    new_joints = arm_utility.rSolver(target_rtz[0], target_rtz[2], segment_lengths=segment_lengths)

    assert abs(new_joints[0]) < math.pi/3.
    assert abs(new_joints[1]) < math.pi/3.
    assert abs(new_joints[2]) < math.pi/3.

    new_coords = arm_utility.computeGripperPosition([0.] + new_joints, segment_lengths=segment_lengths)
    distance = arm_utility.getDistance(new_coords, target)
    assert distance < 0.001
