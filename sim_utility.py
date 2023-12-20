# Utilities for arm simulation.


import math
import torch
import yaml

from arm_utility import computeAllJointPositions


def toUnitVector(list_vector):
    """Returns the list of numbers as a torch unit vector."""
    magnitude = math.sqrt(sum([coord**2 for coord in list_vector]))
    return [component / magnitude for component in list_vector]

def makeTransformationMatrix(destination_origin, destination_bases, source_origin, source_bases):
    """Make a transformation matrix from source (a 3x3 matrix) to destination (another 3x3 matrix)"""
    # Change in the cartesian coordinate systems will consist of a translation and a rotation
    # The origin arguments tell of the translation required to go from one to the other.
    # Transforming from one coordinate system to the other will require a rotation.
    # Basically, if one coordinate system has a different basis than the other, movement along an axis in one system could be spread across all three
    # axes of the other system, so a transform must be used to determine the projection.
    # From https://pkel015.connect.amazon.auckland.ac.nz/SolidMechanicsBooks/Part_III/Chapter_1_Vectors_Tensors/Vectors_Tensors_05_Coordinate_Transformation_Vectors.pdf
    # the components of the 3x3 transformation matrix are Q_{ij} = cos(x_i, x'_j) = e_i \dot e'_j
    # where e_i is the basis vector in the i'th axis for the target coordinate system and e'_j is the same in the source coordinate system.

    # First, let's make all of the basis vectors be unit vectors
    unit_source_bases = torch.tensor([toUnitVector(vector) for vector in source_bases])
    unit_dest_bases = torch.tensor([toUnitVector(vector) for vector in destination_bases])

    # Now find the translations to go from source_origin to destination_origin
    # E.g. if the source origin was 1,2,3 and the destination origin was 2,2,2 then all points are corrected by adding -1,0,1
    translation = [source - dest for dest, source in zip(destination_origin, source_origin)]

    # The dot products will have the relative movement from one basis to the other
    # Add the translations as the fourth column.
    source_to_dest = torch.tensor(
        [
            [unit_dest_bases[0].dot(unit_source_bases[0]), unit_dest_bases[0].dot(unit_source_bases[1]), unit_dest_bases[0].dot(unit_source_bases[2]), translation[0]],
            [unit_dest_bases[1].dot(unit_source_bases[0]), unit_dest_bases[1].dot(unit_source_bases[1]), unit_dest_bases[1].dot(unit_source_bases[2]), translation[1]],
            [unit_dest_bases[2].dot(unit_source_bases[0]), unit_dest_bases[2].dot(unit_source_bases[1]), unit_dest_bases[2].dot(unit_source_bases[2]), translation[2]],
            [0., 0., 0., 1.],
        ]
    )
    # ∎

    return source_to_dest

def jointStatesToImage(joint_states, segment_lengths, arm_origin, arm_bases, camera_fovs, camera_origin, camera_bases, resolution):
    """Render a simple grayscale image of an arm on a white background with a pinhole view.


    Arguments:
        joint_states       list([float]): List of the five joint positions (waist, shoulder, elbow, wrist angle, wrist rotate)
        segment_lengths    list([float]): Lengths of each arm segment
        arm_origin         list([float]): World coordinates of the arm (x, y, z)
        arm_bases          list(list[float]): Basis vectors that defines the arm coordinate system
        camera_fovs        list([float]): Vertical and horizontal fields of view (for a pinhole camera)
        camera_origin      list([float]): World coordinates of the camera
        camera_bases       list(list[float]): Basis vectors that defines the camera coordinate system
        resolution           list([int]): The height and width of the image
    Returns:
        cv::image object
    """
    # Make a drawing buffer
    np_buffer = torch.zeros([1, resolution[0], resolution[1]])

    # Find the joints locations
    coordinates = computeAllJointPositions(joint_states, segment_lengths)

    # Get the transformation matrix
    arm_to_camera = makeTransformationMatrix(camera_origin, camera_bases, arm_origin, arm_bases)

    # Find the camera coordinates for the joints
    # Add the magic number (1.0) onto each one for the coordinate transform and remove them from the end coordinates.
    camera_view_joints = [arm_to_camera.matmul(torch.tensor(coords + [1.0]))[:-1] for coords in coordinates]
    # ∎

    # TODO Add tests

    return camera_view_joints

