#!/usr/bin/env python3

# Test the arm_utility
# In particular, algorithms as in rSolver that are integral to correct robot operation

import pytest

import math
import random
import torch

import sim_utility

unit_vector_data = [
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [0.2, 0.8, 1.5],
    [-1.5, 0.8, 0.0],
]

@pytest.mark.parametrize("basis", unit_vector_data)
def test_toUnitVector(basis):
    # Verify that the vectors are unit vectors
    unit_vector = sim_utility.toUnitVector(basis)
    magnitude = math.sqrt(sum([coord**2 for coord in unit_vector]))
    assert magnitude == pytest.approx(1.0, 0.0001)

transform_test_data = [
    # From the world viewpoint and back into the world viewpoint
    ([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
     [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
     [12., 15., 56.], [12., 15., 56.]),
    # From one point to another
    ([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
     [[0.5, 0.5, 0.], [0.5, 0., 0.5], [0., 0.5, 0.5]],
     [12., 15., 56.], [math.sqrt(0.5)*27, math.sqrt(0.5)*68., math.sqrt(0.5)*71]),
]

@pytest.mark.parametrize("source, destination, given_input, expected_output", transform_test_data)
def test_makeTransformationMatrix(source, destination, given_input, expected_output):
    # Test with several origins
    for _ in range(3):
        dest_origin = [random.randrange(0, 10), random.randrange(0, 10), random.randrange(0, 10)]
        source_origin = [random.randrange(0, 10), random.randrange(0, 10), random.randrange(0, 10)]
        print("Origins are {} and {}".format(dest_origin, source_origin))

        # Verify that the transformation matrix has correct properties
        tx_matrix = sim_utility.makeTransformationMatrix(dest_origin, destination, source_origin, source)

        print("tx matrix is {}".format(tx_matrix))

        # Verify that the translation values are correct
        for dimension in range(3):
            assert tx_matrix[dimension,-1] == source_origin[dimension] - dest_origin[dimension]

        # Subtract the input origin (in world coordinates) so that the random origin is cancelled out before calculating the coordinate transform.
        #input_coords = [coord - origin for coord, origin in zip(given_input, source_origin)]
        #output = tx_matrix.matmul(torch.tensor(input_coords + [1]))[:-1]

        #expected_output_coords = [coord - origin for coord, origin in zip(expected_output, dest_origin)]

        #assert output == pytest.approx(expected_output_coords, 0.0001)

        # Verify that we can transform back to the original coordinates
        output = tx_matrix.matmul(torch.tensor(given_input + [1]))[:-1]
        return_matrix = sim_utility.makeTransformationMatrix(source_origin, source, dest_origin, destination)
        reconstructed_input = return_matrix.matmul(torch.cat((output, torch.ones(1))))[:-1]
        assert reconstructed_input == pytest.approx(given_input, 0.0001)

def test_jointStatesToImage():
    # Verify simple coordinate transformations
    pass

