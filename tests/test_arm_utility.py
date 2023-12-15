#!/usr/bin/env python3

# Test the arm_utility
# In particular, algorithms as in rSolver that are integral to correct robot operation

import pytest

import arm_utility
import math

testdata = [
    ([0.23115182452192629, 0.0, 0.03520485992485018], 0.175),
    ([0.3, 0.0, 0.03520485992485018], 0.175),
    ([0.2, 0.0, 0.3], 0.175),
    ([0.25, 0.0, 0.05],  0.3),
    ([0.05249901188844186, 0.002786401510238651, 0.3881603860167609], 0.265), # this lead to [0.002786401510238651, -1.4469024488034017, -0.71, 1.6765656382911949, 0.004601942375302315], which is super high
    ([0.25321934895550047, 0.0023725500898967374, 0.07310723154126952], 0.265), # this fails
    ([0.19900299276422714, 0.006135923322290182, -0.0409016373742874], 0.265),
]


@pytest.mark.parametrize("target, hand_length", testdata, ids=["near", "far", "high", "long hand", "back failure", "near failure", "solver failure"])
def test_rSolver(target, hand_length):
    # Attempt to generate a pose for a nearby target
    segment_lengths = [0.104, 0.158, 0.147, hand_length]
    target_rtz = arm_utility.XYZToRThetaZ(*target)
    new_joints = arm_utility.rSolver(target_rtz[0], target_rtz[2], segment_lengths=segment_lengths)

    assert abs(new_joints[0]) < math.pi/3.
    assert abs(new_joints[1]) < math.pi/3.
    assert abs(new_joints[2]) < math.pi/3.

    new_coords = arm_utility.computeGripperPosition([0.] + new_joints, segment_lengths)
    distance = arm_utility.getDistance(new_coords, target)
    assert distance < 0.001
