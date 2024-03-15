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
        camera_coordinates (x, y, z): Coordinates of the object in world space from the camera reference frame
        camera_fovs_rads (vfov, hfov): FOVs, in radians, of the pinhole image
        resolution list([int]): The height and width of the image
    Returns:
        (x, y) coordinates on the image
    """
    # Everything is in a right hand rule system, where x is the distance, y is a lateral offset (left/counter clockwise is positive), and z is the
    # vertical offset (up is positive).
    # This is different from the image coordinate system, where the origin is the upper left, y is the vertical offset (down is positive), and x
    # is the horizontal offset (right is positive)

    # If the x coordinate is negative then the image is behind the camera. The corner should map to
    # a place below the screen.
    if camera_coordinates[0] < 0:
        return (-1, -1)

    # First, convert the coordinates to angles, and from that convert them to fov offsets, and from there convert them into pixels.
    # This is the atan of z/x. Positive is up.
    y_angle = math.atan(camera_coordinates[2] / camera_coordinates[0])
    # This is the atan of y/x. Positive is left.
    x_angle = math.atan(camera_coordinates[1] / camera_coordinates[0])

    # Fraction of the image offset based upon the angle and the FOV
    y_fraction = y_angle / camera_fovs_rads[0]
    x_fraction = x_angle / camera_fovs_rads[1]

    # Positive is up, so subtract the offset from the image center. The 0 point on the image is the
    # top.
    y_origin_offset = 0.5 - y_fraction
    # Positive is left, so subtract the offset from the image center
    x_origin_offset = 0.5 - x_fraction

    image_y = round(y_origin_offset * resolution[0])
    image_x = round(x_origin_offset * resolution[1])

    return (image_x, image_y)


class RenderObject(object):

    def __init__(self, buffer, buffer_size, world_coordinates):
        """
        Arguments:
            buffer (cv2 image): The image buffer with the render object.
            buffer_size (width, height): The size of the image buffer.
            coordinates (list[float]): World space coordinates of the corners of the object,
                                       clockwise from the lower left.
        """
        self.buffer = buffer
        self.width = buffer_size[0]
        self.height = buffer_size[1]
        self.coordinates = world_coordinates
        # Clockwise from the lower left corner!!!
        self.src_points = numpy.float32([[0, 0], [0, self.height], [self.width, self.height], [self.width, 0]])

    def getBuffer(self):
        return self.buffer

    def getCoordinates(self):
        return self.coordinates

    def sourcePoints(self):
        return self.src_points


def basisFromYawPitchRoll(yaw, pitch, roll) -> torch.FloatTensor:
    """Create a basis vector from yaw, pitch, and roll."""
    yaw_matrix = torch.tensor([
        [math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
    pitch_matrix = torch.tensor([
        [math.cos(pitch), 0., math.sin(pitch)], [0., 1., 0.], [-math.sin(pitch), 0., math.cos(pitch)]])
    roll_matrix = torch.tensor([
        [1., 0., 0.], [0., math.cos(roll), -math.sin(roll)], [0., math.sin(roll), math.cos(roll)]])
    return yaw_matrix.matmul(pitch_matrix).matmul(roll_matrix)


class MobileEgoToImage(object):
    """Usage:
    1. Initialize
    2. Add objects (letters) with addLetter
    3. beginVideo
    4. Sim loop:
       1. moveEgo
       2. writeFrame
    5. endVideo
    The save() function can also be used to save individual frames rather than a video.
    """

    def __init__(self, ego_origin, ego_pose, relative_camera_origin, camera_pose, camera_fovs_deg, resolution):
        """Create a renderer for simple grayscale images of an arm on a white background with a pinhole view.

        Arguments:
            ego_origin      list([float]): World coordinates of the ego (x, y, z)
            ego_pose          list(float): Yaw, pitch, and roll (in radians) of the ego.
            relative_camera_origin list([float]): Relative coordinates of the camera, relative to ego
            camera_pose       list(float): Yaw, pitch, and roll (in radians) of the camera relative to ego
            camera_fovs_deg list([float]): Vertical and horizontal fields of view (for a pinhole camera, in degrees)
            resolution        list([int]): The height and width of the image
        """
        self.ego_position = [origin for origin in ego_origin]
        self.ego_bases = basisFromYawPitchRoll(*ego_pose)
        self.rel_camera_bases = basisFromYawPitchRoll(*camera_pose)
        self.rel_camera_offset = self.rel_camera_bases.matmul(torch.tensor(relative_camera_origin))
        self.camera_fovs_degs = camera_fovs_deg
        self.camera_fovs_rads = [fov * math.pi / 180.0 for fov in camera_fovs_deg]
        self.resolution = resolution
        self.video_stream = None
        # Colors. Recall that OpenCV images are in BGR order. The ground will be rendered as an
        # object so it needs a transparency value.
        self.sky_blue = (235, 206, 135)
        self.transparent_grass_green = (23, 154, 0, 255)
        # Objects that will be rendered
        self.render_objects = []
        # Add a ground object
        # We could work out the math for the horizon line and the ground location, but it is far
        # easier to simply add a large green square to the render objects instead.
        ground = numpy.zeros([4, 4, 4], dtype=numpy.uint8)
        ground[..., :] = self.transparent_grass_green
        # Make ground tiles rather than handling things going out of view properly
        tile_increment = 10.
        for xoffset in range(6):
            xbase = -3*tile_increment + 2*tile_increment * xoffset
            for yoffset in range(4):
                ybase = -2.5*tile_increment + 2*tile_increment * yoffset
                ground_corners = [
                        [xbase - tile_increment, ybase + tile_increment, 0.], [xbase + tile_increment, ybase + tile_increment, 0.],
                        [xbase - tile_increment, ybase - tile_increment, 0.], [xbase - tile_increment, ybase - tile_increment, 0.]]
                self.render_objects.append(RenderObject(ground, (4, 4), ground_corners))
        # To transform from ego to camera coordinates. This will not need to be updated while
        # running.
        self.ego_to_camera = makeTransformationMatrix(self.rel_camera_offset, self.rel_camera_bases, [0., 0., 0.], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.updateWorldToCamera()

    def moveEgo(self, translations, rotations):
        """Translate and rotate the ego vehicle relative to current position. Rotations are in radians."""
        # Update ego position
        self.ego_position = [position + delta for position, delta in zip(self.ego_position, translations)]
        # Update the ego basis vectors for the new orientation
        self.ego_bases = basisFromYawPitchRoll(*rotations).matmul(self.ego_bases)
        self.updateWorldToCamera()

    def setEgo(self, new_position, pose):
        """Translate and rotate the ego vehicle to an new absolute state. Rotations are in radians."""
        # Update ego position
        self.ego_position = new_position
        # Update the ego basis vectors for the new orientation
        self.ego_bases = basisFromYawPitchRoll(*pose)
        self.updateWorldToCamera()

    def updateWorldToCamera(self):
        #TODO FIXME This is a weird side effect function, maybe just call this once to make the
        # matrix when render is called? Otherwise, this should be called at __init__ and after
        # moveEgo
        """Update the world to camera transformation matrix in self.world_to_camera"""
        transformed_ego_position = self.ego_bases.matmul(torch.tensor(self.ego_position))
        self.world_to_ego = makeTransformationMatrix(transformed_ego_position.tolist(), self.ego_bases, [0., 0., 0.], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.world_to_camera = self.ego_to_camera.matmul(self.world_to_ego)
        print("World to camera is now {}".format(self.world_to_camera))

    def cameraCoordinateToImage(self, camera_coordinates):
        """Convert a camera coordinates tuple into image coordinate space.

        The range for the image will treated as the range 0 to height or width, but coordinates can be returned that lie outside of the image.
        The returned coordinates (x offset from left side, y offset from top)
        """
        return cameraCoordinateToImage(camera_coordinates, self.camera_fovs_rads, self.resolution)

    def render(self):
        """Render a simple grayscale image of an arm on a white background with a pinhole view.

        Arguments:
            joint_states       list([float]): List of the five joint positions (waist, shoulder, elbow, wrist angle, wrist rotate)
        Returns:
            cv::image object
        """
        # Make a drawing buffer (height, width, channels) and fill it with sky
        draw_buffer = numpy.zeros([self.resolution[0], self.resolution[1], 3], dtype=numpy.uint8)
        draw_buffer[..., :] = self.sky_blue

        for render_obj in self.render_objects:
            draw_buffer = self.renderObjectToBackground(render_obj, draw_buffer)

        return draw_buffer

    def addLetter(self, character, char_coordinates):
        """Adds the given character to the render objects at the given world space coordinates."""
        # # Find the size and then make a buffer with the letter
        char_size, baseline = cv2.getTextSize(character, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            thickness=4)
        char_width, char_height = char_size
        char_img_height = char_height + baseline
        # Correct value for RGB: 72, 60, 50. Open CV uses BGR.
        transparent_taupe = (50, 60, 72, 255)
        text_buffer = numpy.zeros((char_img_height, char_width, 4), dtype=numpy.uint8)
        cv2.putText(img=text_buffer, text=character, org=(0, char_height),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=transparent_taupe, thickness=4,
            lineType=cv2.LINE_AA)

        self.render_objects.append(RenderObject(text_buffer, (char_width, char_img_height), char_coordinates))

    def renderObjectToBackground(self, render_object, camera_view_buffer):
        """Render a RenderObject onto the camera view and return the modified buffer."""
        # Find the target points on the image by mapping from the provided points in world
        # coordinates of the object corners onto the camera perspective.
        target_world_coordinates = [self.world_to_camera.matmul(torch.tensor(coord + [1.0])) for coord in render_object.getCoordinates()]
        target_points = numpy.float32([self.cameraCoordinateToImage(world_coordinates) for world_coordinates in target_world_coordinates])
        tx_matrix = cv2.getPerspectiveTransform(render_object.sourcePoints(), target_points)
        #print("DEBUG: object has world coordinates {}".format(target_world_coordinates))

        # TODO FIXME Not all objects are fully visible. Check the corners of the target coordinates
        # to get the necessary image width and size for the rendering area, then draw it onto the
        # correct location, clipping if necessary.
        xmin = round(min([x for x, _ in target_points]))
        ymin = round(min([y for _, y in target_points]))
        xmax = round(max([x for x, _ in target_points]))
        ymax = round(max([y for _, y in target_points]))

        #print("Target points are: {}".format(target_points))

        # Not handling partially visible objects
        if xmin > camera_view_buffer.shape[1] or ymin > camera_view_buffer.shape[0] or 0 >= ymax or 0 >= xmax or xmin == xmax or ymin == ymax:
            # Nothing to render
            return camera_view_buffer
        # Removing things behind the camera
        if xmin < 0 or ymin < 0:
            return camera_view_buffer

        #print("DEBUG: xmin {}, xmax {}, ymin {}, ymax {}".format(xmin, xmax, ymin, ymax))
        warp_size = (xmax - xmin, ymax - ymin)

        # Target area for warped image that is within the bounds of the target image
        dest_x_begin = max(0, xmin)
        dest_y_begin = max(0, ymin)
        dest_x_end = min(camera_view_buffer.shape[1], xmax)
        dest_y_end = min(camera_view_buffer.shape[0], ymax)

        # The clipped source area from the warped patch itself
        src_x_begin = max(0, dest_x_begin - xmin)
        src_y_begin = max(0, dest_y_begin - ymin)
        src_x_end = min(xmax - xmin, dest_x_end - xmin)
        src_y_end = min(ymax - ymin, dest_y_end - ymin)

        dest_range = (slice(dest_y_begin, dest_y_end), slice(dest_x_begin, dest_x_end))
        src_range = (slice(src_y_begin, src_y_end), slice(src_x_begin, src_x_end))

        #print("DEBUG: rendering visible object with warp size {} from {} onto {}".format(warp_size, src_range, dest_range))

        ## TODO Running the code with known values, the above optimization isn't working
        warp_size = (camera_view_buffer.shape[1], camera_view_buffer.shape[0])
        dest_range = (slice(0, camera_view_buffer.shape[0]), slice(0, camera_view_buffer.shape[1]))
        src_range = (slice(0, camera_view_buffer.shape[0]), slice(0, camera_view_buffer.shape[1]))

        # Warp the object onto the buffer
        # cv.INTER_AREA is being used to avoid loss of quality assuming that most images will be
        # downscaled during the projection. This is not based upon any experiments.
        warped_buffer = cv2.warpPerspective(src=render_object.getBuffer(), M=tx_matrix,
                dsize=warp_size, flags=cv2.INTER_AREA)
        warped_buffer = warped_buffer[src_range]

        # Render onto background with the worst version of alpha support ever (meaning manual) by
        # converting the alpha channel of the warped images into a mask, broadcasting it to the
        # image size, and then multiplying by the warped image and the render target.
        # TODO Use cv2.threshold to get a mask and then cv2.bitwise_and and cv2.bitwise_not to apply
        # the mask and gets its inverse. That would save the multiplication here.
        object_mask = warped_buffer[..., -1]/255
        object_mask = numpy.repeat(object_mask[:, :, numpy.newaxis], 3, axis=2)
        masked_object = object_mask * warped_buffer[:, :, :-1]
        masked_buffer_patch = (1.0 - object_mask) * camera_view_buffer[dest_range]
        redrawn_patch = cv2.add(masked_object.astype(numpy.uint8), masked_buffer_patch.astype(numpy.uint8))
        camera_view_buffer[dest_range] = redrawn_patch

        return camera_view_buffer

    def save(self, path):
        """Render and save the robot state into path."""
        cv2.imwrite(path, self.render())

    def beginVideo(self, path):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # The order of height and width is not the same as expected of the buffer itself.
        height_width = (self.resolution[1], self.resolution[0])
        self.video_stream = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=30.0, frameSize=height_width, isColor=True)
        if not self.video_stream.isOpened():
            print("Video creation failed, possibly due to avc1 codec. Retrying with mp4v codec.")
            # OpenCV may not have been built with suppoprt for the h264 format
            #add_codec("mp4", "mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_stream = cv2.VideoWriter(filename=path, fourcc=fourcc, fps=30.0, frameSize=height_width, isColor=True)
        if self.video_stream is None:
            raise RuntimeError("No supported codec found for video writing.")

    def writeFrame(self):
        if self.video_stream is not None:
            image = self.render()
            self.video_stream.write(image)

    def endVideo(self):
        self.video_stream.release()
        self.video_stream = None


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
