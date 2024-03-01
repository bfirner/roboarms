# Utilities for arm simulation.


import cv2
import math
import numpy
import torch
import yaml

from arm_utility import computeAllJointPositions


def toUnitVector(list_vector):
    """Returns the list of numbers as a torch unit vector."""
    magnitude = math.sqrt(sum([coord**2 for coord in list_vector]))
    return [component / magnitude for component in list_vector]

def makeTransformationMatrix(destination_origin, destination_bases, source_origin, source_bases):
    """Make a transformation matrix from source (a 3x3 matrix) to destination (another 3x3 matrix)

    Bear in mind that the transformation matrix itself will by 4x4 rather than 3x3 to allow for
    translation as well as rotation.

    """
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

# TODO To be able to render labels onto an image, we need some utility functions
# 1. Label to original image
# sim_utility.cameraCoordinateToImage((x, y, z), fovs_in_radians, resolution) -> (y, x)
# 1.1 Requires storing image metadata along with the video
# 2. Common cropping functions to process tensors with marked labels in the same way as the training images
# 2.1 Move the process_img function (and setup code) out of inference_sim.py and into
#     bee_analysis.utility.video_utility so that it can be used for all cropping during dataprep,
#     inference, and debugging.
def cameraCoordinateToImage(camera_coordinates, camera_fovs_rads, resolution):
    """Convert a camera coordinates tuple into image coordinate space.

    The range for the image will treated as the range 0 to height or width, but coordinates can be returned that lie outside of the image.
    The returned coordinates (x offset from left side, y offset from top)
    Arguments:
        camera_coordinates (x, y, z): Coordinates of the object
        camera_fovs_rads (vfov, hfov): FOVs, in radians, of the pinhole image
        resolution list([int]): The height and width of the image
    Returns:
        (x, y) coordinates on the image
    """
    # Everything is in a right hand rule system, where x is the distance, y is a lateral offset (left/counter clockwise is positive), and z is the
    # vertical offset (up is positive).
    # This is different from the image coordinate system, where the origin is the upper left, y is the vertical offset (down is positive), and x
    # is the horizontal offset (right is positive)

    # First, convert the coordinates to angles, and from that convert them to fov offsets, and from there convert them into pixels.
    # This is the atan of z/x. Positive is up.
    y_angle = math.atan(camera_coordinates[2] / camera_coordinates[0])
    # This is the atan of y/x. Positive is left.
    x_angle = math.atan(camera_coordinates[1] / camera_coordinates[0])

    # Fraction of a half image offset based upon the angle and the FOV
    y_center_offset = y_angle / camera_fovs_rads[0]
    x_center_offset = x_angle / camera_fovs_rads[1]

    # Positive is up, so subtract the offset from the image center
    y_origin_offset = 0.5 - y_center_offset
    # Positive is left, so subtract the offset from the image center
    x_origin_offset = 0.5 - x_center_offset

    image_y = round(y_origin_offset * resolution[0])
    image_x = round(x_origin_offset * resolution[1])

    return (image_x, image_y)


class JointStatesToImage(object):

    def __init__(self, segment_lengths, arm_origin, arm_bases, camera_fovs_deg, camera_origin, camera_bases, resolution):
        """Create a renderer for simple grayscale images of an arm on a white background with a pinhole view.


        Arguments:
            segment_lengths    list([float]): Lengths of each arm segment
            arm_origin         list([float]): World coordinates of the arm (x, y, z)
            arm_bases          list(list[float]): Basis vectors that defines the arm coordinate system
            camera_fovs_deg    list([float]): Vertical and horizontal fields of view (for a pinhole camera, in degrees)
            camera_origin      list([float]): World coordinates of the camera
            camera_bases       list(list[float]): Basis vectors that defines the camera coordinate system
            resolution               list([int]): The height and width of the image
        Returns:
            cv::image object
        """
        self.segment_lengths = segment_lengths
        self.camera_fovs_degs = camera_fovs_deg
        self.camera_fovs_rads = [fov * math.pi / 180.0 for fov in camera_fovs_deg]
        self.resolution = resolution
        self.video_stream = None
        # Make a drawing buffer (height, width, channels)
        self.draw_buffer = numpy.zeros([resolution[0], resolution[1], 3], dtype=numpy.uint8)
        # We may be asked to keep a consistent background image during the simulation
        self.background = None

        # Get the transformation matrix for arm to camera
        self.arm_to_camera = makeTransformationMatrix(camera_origin, camera_bases, arm_origin, arm_bases)

        # Verify that the transformation matrix has rows that are orthogonal. This should be true; each coordinate plane should be orthogonal to the
        # others. An orthogonal matrix multiplied by itself will yield the identity matrix.
        # Allow some numeric imprecision (a difference less than 0.0001)
        if not (torch.abs(self.arm_to_camera[:3,:3].matmul(self.arm_to_camera[:3,:3].T) - torch.eye(3)) < 0.0001).all().item():
            raise RuntimeError(
                "Basis vectors provided to JointStatesToImage that are not orthogonal.  Transformation matrix is {}".format(self.arm_to_camera))

        # Get the transformation matrix for world to camera. This is used if items will be drawn in
        # the background.
        self.bg_to_camera = makeTransformationMatrix(camera_origin, camera_bases, [0., 0., 0.], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def cameraCoordinateToImage(self, camera_coordinates):
        """Convert a camera coordinates tuple into image coordinate space.

        The range for the image will treated as the range 0 to height or width, but coordinates can be returned that lie outside of the image.
        The returned coordinates (x offset from left side, y offset from top)
        """
        return cameraCoordinateToImage(camera_coordinates, self.camera_fovs_rads, self.resolution)

    def render(self, joint_states):
        """Render a simple grayscale image of an arm on a white background with a pinhole view.

        Arguments:
            joint_states       list([float]): List of the five joint positions (waist, shoulder, elbow, wrist angle, wrist rotate)
        Returns:
            cv::image object
        """

        # Find the joints locations in arm space
        arm_coordinates = computeAllJointPositions(joint_states, self.segment_lengths)

        # Find the camera coordinates for the joints
        # Add the magic number (1.0) onto each one for the coordinate transform and remove them from the end coordinates.
        joint_coordinates = [self.arm_to_camera.matmul(torch.tensor(coords + [1.0]))[:-1] for coords in arm_coordinates]

        # Convert world coordinates to camera image coordinates
        image_coordinates = [self.cameraCoordinateToImage(coord) for coord in joint_coordinates]

        # Now render the joints.
        # TODO Just drawing with lines, should render with rectangles or something
        thickness = 5
        # TODO Different color for each joint
        color = (0.1, 0.1, 0.1)
        # Fill the buffer with white unless a background already exists.
        if self.background is None:
            self.draw_buffer.fill(255.0)
        else:
            numpy.copyto(self.draw_buffer, self.background)
        for coord_a, coord_b in zip(image_coordinates[:-1], image_coordinates[1:]):
            # Draw onto the draw buffer with an anti-aliased line
            cv2.line(img=self.draw_buffer, pt1=coord_a, pt2=coord_b, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            # TODO probably use a cv2.fillPoly or cv2.polylines to draw the robot limbs with size

        return self.draw_buffer

    def addLetter(self, character, char_coordinates):
        """Adds the given character to the image at the given world space coordinates."""
        # Create a background if one does not already exist
        # It is the same size as the drawing buffer and starts as a white color (255, 255, 255)
        if self.background is None:
            self.background = 255 * numpy.ones([self.resolution[0], self.resolution[1], 3], dtype=numpy.uint8)
        # Add the letter to the background

        # # Find the size and then make a buffer with the letter
        char_size, baseline = cv2.getTextSize(character, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            thickness=4)
        char_width, char_height = char_size
        char_img_height = char_height + baseline
        # Correct value for RGB: 72, 60, 50. Open CV uses BGR.
        #transparent_taupe = (72, 60, 50, 255)
        transparent_taupe = (50, 60, 72, 255)
        text_buffer = numpy.zeros((char_img_height, char_width, 4), dtype=numpy.uint8)
        cv2.putText(img=text_buffer, text=character, org=(0, char_height),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=transparent_taupe, thickness=4,
            lineType=cv2.LINE_AA)
        src_points = numpy.float32([[0, 0], [char_width, 0], [char_width, char_img_height], [0, char_img_height]])

        # # Find the target points on the image by mapping from the provided points in world coordinates into the camera perspective.
        target_coordinates = [self.cameraCoordinateToImage(self.bg_to_camera.matmul(torch.tensor(coord + [1.0]))) for coord in char_coordinates]
        target_points = numpy.float32(target_coordinates)
        tx_matrix = cv2.getPerspectiveTransform(src_points, target_points)

        # # cv2.warpPerspective onto a buffer the same size as the background
        # # We could have rendered this onto a smaller image and then drawn into the proper
        # # position, but since this is a one-time operation we will consider this "okay" for now.
        warped_text_buffer = cv2.warpPerspective(text_buffer, tx_matrix, (self.background.shape[1], self.background.shape[0]))

        # Render onto background with the worst version of alpha support ever (meaning manual)
        text_mask = warped_text_buffer[:, :, 3]/255
        text_mask = numpy.repeat(text_mask[:, :, numpy.newaxis], 3, axis=2)
        masked_letter = text_mask * warped_text_buffer[:,:,:-1]
        masked_bg = (1.0 - text_mask) * self.background
        self.background = cv2.add(masked_letter.astype(numpy.uint8), masked_bg.astype(numpy.uint8))


    def save(self, joint_states, path):
        """Render and save the robot state into path."""
        cv2.imwrite(path, self.render(joint_states))

    def beginVideo(self, path):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # The order of height and width is not the same as expected of the buffer itself.
        height_width = (self.resolution[1], self.resolution[0])
        self.video_stream = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=30.0, frameSize=height_width, isColor=True)

    def writeFrame(self, joint_states):
        if self.video_stream is not None:
            image = self.render(joint_states)
            self.video_stream.write(image)

    def endVideo(self):
        self.video_stream.release()
        self.video_stream = None
