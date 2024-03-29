# Utilities for arm handling, such as calibration data or controlling an arm, and to analyze robot
# joint states, such as from a rosbag.


import math
import torch
import yaml

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


def getCalibratedJoints(arm_data, ordered_joint_names, calibration):
    positions = []
    for joint_name in ordered_joint_names:
        joint_index = arm_data['name'].index(joint_name)
        positions.append(arm_data['position'][joint_index] - calibration[joint_name])
    return positions

def rSolver(r_value, z_value, segment_lengths=[0.104, 0.158, 0.147, 0.175]):
    """Solve to joint positions given a desired 'r' value in polar coordinates.

    Will stay close to the given z_value (vertical offset) for the tip of the arm.

    Works for the px150 robot arm.
    Note: The px150 with a chopstick pointer has length 0.265 for the last joint

    Arguments:
        r_value               (float): Desired radius extension
        z_value               (float): Desired z offset
        segment_lengths (List[float]): Lengths of the robot segments, in mm
    Returns:
        returns tuple of joint values
    """
    # Get within 1.0mm of the desired location (which is 0.001m)
    error_bound = 0.001
    # Don't bend a joint more than this amount
    max_bend = math.pi/3.0
    # The lengths of segments (or effective segments) that are moved by the joints, in mm
    segment_G = segment_lengths[0]    # Height of the pedestal upon which theta1 rotates
    segment_C = segment_lengths[1]    # Effective length from theta1 to theta2
    segment_D = segment_lengths[2]    # Length from theta2 to theta3
    segment_H = segment_lengths[3]    # Length of the grasper from theta3

    # Solvers for radius and z positions, in mm
    def rOffset(theta1, theta2, theta3):
        return math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta3 + theta2 + theta1)*segment_H
    def zOffset(theta1, theta2, theta3):
        return segment_G + math.cos(theta1)*segment_C - math.sin(theta1 + theta2)*segment_D - math.sin(theta1 + theta2 + theta3)*segment_H

    # First, move the shoulder joint so that the limb is perpendicular to a line going from the
    # center of the shoulder segment to the target location
    # This makes a triangle with once side equal to segment_C/2 and the hypotenuse will be
    # sqrt((segment_G-z_value)**2 + (0-r_value)**2).
    # The initial shoulder angle should thus be: math.acos((segment_C/2.) / hypotenuse)
    hypotenuse = math.sqrt((segment_G-z_value)**2 + (0-r_value)**2)
    initial_shoulder_angle = math.acos((segment_C/2.) / hypotenuse)
    theta1 = initial_shoulder_angle
    # Don't bend past the maximum
    if theta1 > max_bend:
        theta1 = max_bend - 10e-5
    elif theta1 < -max_bend:
        theta1 = -max_bend + 10e-5
    theta2 = 0.
    theta3 = 0.

    def shoulderCoord(theta1):
        shoulder_r = math.sin(theta1)*segment_C
        shoulder_z = segment_G + math.cos(theta1)*segment_C
        return (shoulder_r, shoulder_z)

    def elbowCoord(theta1, theta2):
        elbow_r = math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D
        elbow_z = segment_G + math.cos(theta1)*segment_C - math.sin(theta1 + theta2)*segment_D
        return (elbow_r, elbow_z)

    def handCoord(theta1, theta2, theta3):
        hand_r = math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta1 + theta2 + theta3)*segment_H
        hand_z = segment_G + math.cos(theta1)*segment_C - math.sin(theta1 + theta2)*segment_D - math.sin(theta1 + theta2 + theta3)*segment_H
        return (hand_r, hand_z)

    def shoulderDistance(theta1):
        shoulder_r, shoulder_z = shoulderCoord(theta1)
        return math.sqrt((shoulder_r - r_value)**2 + (shoulder_z - z_value)**2)

    def elbowDistance(theta1, theta2):
        elbow_r, elbow_z = elbowCoord(theta1, theta2)
        return math.sqrt((elbow_r - r_value)**2 + (elbow_z - z_value)**2)

    def handDistance(theta1, theta2, theta3):
        hand_r, hand_z = handCoord(theta1, theta2, theta3)
        return math.sqrt((hand_r - r_value)**2 + (hand_z - z_value)**2)

    # Search for the correct position starting with this increment
    increment = 0.1

    # Find the direction the shoulder will need to bend.
    # If the elbow and wrist segments are less than the distance the elbow will need to bend
    # forward, and if they are greater than the distance the elbow will need to bend backwards
    # Adjust theta1 until it is as the right distance so that the rest of the arm will touch the
    # point
    distance = shoulderDistance(theta1) - (segment_D + segment_H)
    shoulder_sign = math.copysign(1.0, distance)

    while abs(distance) > error_bound and abs(theta1) < max_bend and abs(increment) > 10e-5:
        # If increment will push theta1 over the maximum bend, set theta1 to the maximum bend
        # instead
        next_theta1 = theta1 + shoulder_sign*increment
        if next_theta1 > max_bend:
            next_theta1 = max_bend
        elif next_theta1 < -max_bend:
            next_theta1 = -max_bend

        new_distance = shoulderDistance(next_theta1) - (segment_D + segment_H)

        # See if we've gone too far. If not, keep the new value and continue
        new_sign = math.copysign(1.0, new_distance)
        if new_sign != shoulder_sign or abs(next_theta1) > max_bend:
            increment = increment / 10.0
        else:
            theta1 = next_theta1
            distance = new_distance

    # If the end of the arm is too close, bend theta3 in the opposite direction as theta1 to shorten
    # the effective distance from the elbow joint to the end of the hand
    if distance < -error_bound:
        # From the law of cosines, the distance is:
        # dist**2 = segment_D**2 + segment_H**2 - 2 * segment_D * segment_H * math.cos(math.pi - theta3)
        # The distance is the desired distance, which is segment_D + segment_H - distance
        abs_dist = shoulderDistance(theta1)
        theta3 = math.pi - math.acos((segment_D**2 + segment_H**2 - abs_dist**2)/(2*segment_D*segment_H))
        theta3 = -1 * math.copysign(theta3, theta1)

        # We repeat the previous search, but now changing theta2 while theta1 remains fixed.
        theta2 = max_bend - 10e-5
        distance = handDistance(theta1, theta2, theta3)
        hand_coords = handCoord(theta1, theta2, theta3)
        target_angle = math.atan(z_value / r_value)
        hand_angle = math.atan(hand_coords[1] / hand_coords[0])
        angle_sign = math.copysign(1.0, hand_angle - target_angle)
        # TODO It feels more elegant (but doesn't really matter) if we start at 0 and move in the
        # correct sign direction rather than going to the max_bend and working backwards


        # TODO FIXME If setting theta2 to the max bend still needs to go more positive to reach the
        # target, then the target is unreachable without moving theta1 again.
        if angle_sign == 1.0:
            # TODO Raise an error
            return [theta1, theta2, theta3]

        increment = 0.1
        while abs(distance) > error_bound and abs(theta2) < max_bend and abs(increment) > 10e-5:
            # TODO If increment will push theta2 over the maximum bend, set theta2 to the maximum
            # bend instead
            next_theta2 = theta2 - increment
            if next_theta2 >= max_bend:
                next_theta2 = max_bend
            elif next_theta2 <= -max_bend:
                next_theta2 = -max_bend

            new_distance = handDistance(theta1, next_theta2, theta3)
            new_coords = handCoord(theta1, next_theta2, theta3)
            new_hand_angle = math.atan(new_coords[1] / new_coords[0])
            new_angle_sign = math.copysign(1.0, new_hand_angle - target_angle)

            # See if we've gone too far. If not, keep the new value and continue
            if new_distance > distance or abs(next_theta2) >= max_bend or new_angle_sign != angle_sign:
                increment = increment / 10.0
            else:
                theta2 = next_theta2
                distance = new_distance
    # Bend the elbow (but keep the hand straight) to converge if the arm length is correct to touch
    # the target.
    elif abs(distance) < error_bound:
        # Now solve for theta2
        # The hypotenuse of the triangle is the distance. Grab the z distance as well, and plug that
        # ratio into the math.acos function to get the angle from the end of the shoulder. Then
        # remove the angles from theta1 to get the angle for theta2.
        # Subtract from math.pi/2 to be relative to the joint rather than the perpendicular line to
        # the ground from the joint.
        _, shoulder_z = shoulderCoord(theta1)
        z_distance = shoulder_z - z_value
        angle_from_shoulder_end = math.acos(z_distance / shoulderDistance(theta1))
        theta2 = math.pi/2 - angle_from_shoulder_end - theta1
        theta3 = 0
        # QED
    # Otherwise the arm is not long enough. There's nothing that we can do in this case
    else:
        # TODO Raise an error
        return [False, False, False]

    # The final distance from the target, after flexing theta3
    distance = handDistance(theta1, theta2, theta3)

    # If this failed then we are out of luck
    assert distance <= error_bound

    # Return the result
    return [theta1, theta2, theta3]

def XYZToRThetaZ(x, y, z):
    """Convert x,y,z coordinates to [r,theta,z]."""
    r = math.sqrt(x**2 + y**2)
    theta = math.atan(y/x)
    return [r, theta, z]

def RThetaZtoXYZ(r, theta, z):
    """Convert r, theta, z coordinates to [x,y,z]."""
    x = math.cos(theta) * r
    y = math.sin(theta) * r
    return [x, y, z]

def RTZClassifierNames():
    """Return the names of the rtz classifier outputs.

    At some point, if this becomes too complicated it may make sense to encapsulate all classifier
    outputs into a class with @staticmethod members
    """
    return ["r+0.25", "r-0.25", "z+0.25", "z-0.25", "theta+pi/30", "theta+pi/100", "theta+pi/300",
            "theta-pi/30", "theta-pi/100", "theta-pi/300"]

def interpretRTZPrediction(r, theta, z, threshold, prediction):
    """Interpret a 10 value prediction from a DNN.

    Arguments:
        r: The current value of r
        theta: The current value of theta
        z: The current values of z
        threshold: The movement distance during dataprep. Outputs are true if movement >=0.25*threshold.
        prediction: The 10 element prediction from a DNN.
    Returns:
        [r,theta,z]
    """
    # There are a total of 10 outputs: 2 for r, 2 for z, and 6 for theta

    out_r = r
    out_theta = theta
    out_z = z

    # The first two outputs should be 1 if:
    # Current position r is more than the movement threshold less than the target r
    # Current position r is more than the movement threshold greater than the target r
    if 0.5 < prediction[0]:
        out_r += 0.25 * threshold
    if 0.5 < prediction[1]:
        out_r -= 0.25 * threshold
    # Notice that is both prediction[0] and prediction[1] are > 0.5 then out_r will still be r.


    # The second two outputs should be true if:
    # Current position z is more than the movement threshold less than the target z
    # Current position z is more than the movement threshold greater than the target z
    if 0.5 < prediction[2]:
        out_z += 0.25 * threshold
    if 0.5 < prediction[3]:
        out_z -= 0.25 * threshold
    # The next three outputs are true if:
    # Current position theta is more than the given threshold less than the target theta
    # Current position theta is more than the given threshold greater than the target theta
    # Where theta is broken into slices of math.pi/10, math.pi/30, and math.pi/100
    if prediction[4] > 0.5:
        # 6 degrees
        out_theta += math.pi/30.
    elif prediction[5] > 0.5:
        # 1.8 degrees
        out_theta += math.pi/100.
    elif prediction[6] > 0.5:
        # 0.6 degrees
        out_theta += math.pi/300.
    if prediction[7] > 0.5:
        out_theta -= math.pi/30.
    elif prediction[8] > 0.5:
        out_theta -= math.pi/100.
    elif prediction[9] > 0.5:
        out_theta -= math.pi/300.

    return [out_r, out_theta, out_z]

def rtzDiffToClassifier(current_rtz_position, target_rtz_position, prediction_distance):
    return [1 if value else 0 for value in [
        # Current position r is more than the movement threshold less than the target r
        0.25 * prediction_distance < target_rtz_position[0] - current_rtz_position[0],
        # Current position r is more than the movement threshold greater than the target r
        0.25 * prediction_distance < current_rtz_position[0] - target_rtz_position[0],
        # Current position z is more than the movement threshold less than the target z
        0.25 * prediction_distance < target_rtz_position[2] - current_rtz_position[2],
        # Current position z is more than the movement threshold greater than the target z
        0.25 * prediction_distance < current_rtz_position[2] - target_rtz_position[2],
        # Current position theta is more than the given threshold less than the target theta
        math.pi/30. < target_rtz_position[1] - current_rtz_position[1],
        math.pi/100. < target_rtz_position[1] - current_rtz_position[1],
        math.pi/300. < target_rtz_position[1] - current_rtz_position[1],
        # Current position theta is more than the given threshold greater than the target theta
        math.pi/30. < current_rtz_position[1] - target_rtz_position[1],
        math.pi/100. < current_rtz_position[1] - target_rtz_position[1],
        math.pi/300. < current_rtz_position[1] - target_rtz_position[1],
    ]]

def dnnToActionFunction(dnn_labels, label_locations):
    """Return a function that will convert the DNN output to an xyz coordinate.

    Argument:
        dnn_labels (list[str]): The names of DNN outputs in the order that they appear.
        label_locations (list[int or slice]): Locations of DNN outputs
    Returns:
        (function or None, relative_movement:bool)

    """
    # A flag that indicates if the conversion function will also need the current coordinates
    relative_movement = False
    if "target_arm_position" in dnn_labels:
        nn_joint_slice = label_locations['target_arm_position']
        dnn_output_to_xyz = lambda out: computeGripperPosition(out[nn_joint_slice].tolist())
    elif "target_xyz_position" in dnn_labels:
        dnn_output_to_xyz = lambda out: out[label_locations['target_xyz_position']].tolist()
    elif "target_rtz_position" in dnn_labels:
        rtz_slice = label_locations['target_rtz_position']
        dnn_output_to_xyz = lambda out: RThetaZtoXYZ(*out[rtz_slice].tolist())
    elif "current_xyz_position" in dnn_labels:
        dnn_output_to_xyz = lambda out: out[label_locations['current_xyz_position']].tolist()
    elif ("goal_distance_x" in dnn_labels and
            "goal_distance_y" in dnn_labels and
            "goal_distance_z" in dnn_labels):
        x_label = label_locations['goal_distance_x']
        y_label = label_locations['goal_distance_y']
        z_label = label_locations['goal_distance_z']
        def distancesToTarget(x, y, z, dnn_out):
            return [x + dnn_out[x_label].item(),
                    y + dnn_out[y_label].item(),
                    z + dnn_out[z_label].item()]
        dnn_output_to_xyz = distancesToTarget
        # Indicate that movement is relative to the current position
        relative_movement = True
    elif "rtz_classifier" in dnn_labels:
        # Interpreting this output is more complicated than the others since there are multiple
        # steps. We will just call a utility function from arm_utility
        rtz_classify_slice = label_locations['rtz_classifier']
        def processRTZClassifier(x, y, z, dnn_out):
            # TODO This was a training parameter, it should be pulled from the dataset
            # Setting the threshold to 1cm here
            threshold = 0.01
            cur_rtz = XYZToRThetaZ(x, y, z)
            next_rtz = interpretRTZPrediction(*cur_rtz, threshold, dnn_out[rtz_classify_slice].tolist())
            return RThetaZtoXYZ(*next_rtz)

        dnn_output_to_xyz = processRTZClassifier
        # Indicate that movement is relative to the current position
        relative_movement = True
    else:
        dnn_output_to_xyz = None
    return (dnn_output_to_xyz, relative_movement)

def getWaistCoords(theta0, segment_G):
    waist_x = 0.
    waist_y = 0.
    waist_z = segment_G
    return [waist_x, waist_y, waist_z]

def getShoulderCoords(theta0, theta1, segment_G, segment_C):
    shoulder_x = (math.sin(theta1)*segment_C*math.cos(theta0))
    shoulder_y = (math.sin(theta1)*segment_C*math.sin(theta0))
    shoulder_z = segment_G + math.cos(theta1)*segment_C
    return [shoulder_x, shoulder_y, shoulder_z]

def getElbowCoords(theta0, theta1, theta2, segment_G, segment_C, segment_D):
    elbow_x = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D)*math.cos(theta0)
    elbow_y = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D)*math.sin(theta0)
    elbow_z = segment_G + math.cos(theta1)*segment_C - math.sin(theta1 + theta2)*segment_D
    return [elbow_x, elbow_y, elbow_z]

def computeAllJointPositions(joint_states, segment_lengths=[0.104, 0.158, 0.147, 0.175]):
    """Return the x,y,z positions of the robot waist, shoulder, elbow, and the end of the gripper

    Arguments:
        positions       (list[float]): Positions of the joints
        segment_lengths (list[float]): Lengths of each arm segment
    Returns:
        [(x,y,z)]: list of tuples of the gripper location, in meters
    """
    # Assign names to everything so that the equations make sense
    theta0 = joint_states[0]
    theta1 = joint_states[1]
    theta2 = joint_states[2]
    theta3 = joint_states[3]
    # The lengths of segments (or effective segments) that are moved by the previous joints,
    # in mm
    segment_G = segment_lengths[0]    # Height of the pedestal upon which theta1 rotates
    segment_C = segment_lengths[1]    # Effective length from theta1 to theta2
    segment_D = segment_lengths[2]    # Length from theta2 to theta3
    segment_H = segment_lengths[3]    # Length of the grasper from theta3
    return [
        getWaistCoords(theta0, segment_G),
        getShoulderCoords(*joint_states[0:2], *segment_lengths[0:2]),
        getElbowCoords(*joint_states[0:3], *segment_lengths[0:3]),
        computeGripperPosition(joint_states, segment_lengths)
    ]


# TODO These names are confusing (computeGripperPosition and getGripperPosition)
def computeGripperPosition(positions, segment_lengths=[0.104, 0.158, 0.147, 0.175]):
    """Get the x,y,z position of the gripper relative to the point under the waist in meters.

    Works for the px150 robot arm.

    Arguments:
        positions      (list[float]): Positions of the joints
    Returns:
        x,y,z tuple of the gripper location, in meters
    """
    # TODO These are specific values for the px150 Interbotix robot arm. They should be
    # placed into a library.
    # TODO FIXME These should be calibrated to a known 0 position or the distances will be
    # slightly off (or extremely off, depending upon the calibration).
    # The four joint positions that influence end effector position
    theta0 = positions[0]
    theta1 = positions[1]
    theta2 = positions[2]
    theta3 = positions[3]
    # The lengths of segments (or effective segments) that are moved by the previous joints,
    # in mm
    segment_G = segment_lengths[0]    # Height of the pedestal upon which theta1 rotates
    segment_C = segment_lengths[1]    # Effective length from theta1 to theta2
    segment_D = segment_lengths[2]    # Length from theta2 to theta3
    segment_H = segment_lengths[3]    # Length of the grasper from theta3
    arm_x = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta3 + theta2 + theta1)*segment_H)*math.cos(theta0)
    arm_y = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta3 + theta2 + theta1)*segment_H)*math.sin(theta0)
    arm_z = segment_G + math.cos(theta1)*segment_C - math.sin(theta1 + theta2)*segment_D - math.sin(theta1 + theta2 + theta3)*segment_H
    # Return the x,y,z end effector coordinates in meters
    return [arm_x, arm_y, arm_z]

    #model_generator = getattr(mrd, 'px150')
    #robot_model = model_generator()
    ## Below is "correct" but gets incorrect results. Instead we will use geometry and hand-measured
    ## values for the arm segments. There must be something wrong in the M or Slist matrices
    ## (probably the Slist) but the problem isn't immediately apparent.
    ## 'M' is the home configuration of the robot, Slist has the joint screw axes at the home
    ## position. This should return the end effector position.
    #T = modern_robotics.FKinSpace(robot_model.M, robot_model.Slist, positions)
    ## Return the x,y,z components of the translation matrix.
    #return (T[0][-1], T[1][-1], T[2][-2])


def getGripperPosition(robot_model, arm_record, segment_lengths=[0.104, 0.158, 0.147, 0.175]):
    """Get the x,y,z position of the gripper relative to the point under the waist in meters.

    This is basically the same as the get_ee_pose function for an interbotix arm, but takes in an
    arbitrary set of joint states. Like that function, this just calls something from the
    modern_robotics package.

    The robot model should be fetched from
         getattr(interbotix_xs_modules.xs_robot.mr_descriptions, 'px150')

    Arguments:
        robot_model      (class): String that identifies this robot arm in the interbotix modules.
        arm_record ({str: data}): ROS message data for the robot arm.
    Returns:
        x,y,z tuple of the gripper location, in meters
    """
    # TODO These are specific values for the px150 Interbotix robot arm. They should be
    # placed into a library.
    # TODO FIXME These should be calibrated to a known 0 position or the distances will be
    # slightly off (or extremely off, depending upon the calibration).
    # The four joint positions that influence end effector position
    assert "px150" == robot_model
    joint_positions = [
        arm_record['position'][arm_record['name'].index('waist')],
        arm_record['position'][arm_record['name'].index('shoulder')],
        arm_record['position'][arm_record['name'].index('elbow')],
        arm_record['position'][arm_record['name'].index('wrist_angle')],
    ]
    return computeGripperPosition(joint_positions, segment_lengths)


def grepGripperLocationFromTensors(positions):
    """Get the x,y,z locations from a 4-value tensors. A batch dimension is assumed

    Works for the px150 robot arm.

    Arguments:
        positions      (List[float]): Positions of the joints
    Returns:
        x,y,z tuple of the gripper location, in meters
    """
    # TODO These are specific values for the px150 Interbotix robot arm. They should be
    # placed into a library.
    theta0 = positions[:,0]
    theta1 = positions[:,1]
    theta2 = positions[:,2]
    theta3 = positions[:,3]
    # The lengths of segments (or effective segments) that are moved by the previous joints,
    # in mm
    segment_G = 104    # Height of the pedestal upon which theta1 rotates
    segment_C = 158    # Effective length from theta1 to theta2
    segment_D = 147    # Length from theta2 to theta3
    segment_H = 175    # Length of the grasper from theta4
    arm_x = (torch.sin(theta1)*segment_C + torch.cos(theta2 + theta1)*segment_D + torch.cos(theta3 + theta2 + theta1)*segment_H)*torch.cos(theta0)
    arm_y = (torch.sin(theta1)*segment_C + torch.cos(theta2 + theta1)*segment_D + torch.cos(theta3 + theta2 + theta1)*segment_H)*torch.sin(theta0)
    arm_z = segment_G + torch.cos(theta1)*segment_C - torch.sin(theta1 + theta2)*segment_D - torch.sin(theta1 + theta2 + theta3)*segment_H
    # Return the x,y,z end effector coordinates in meters. Restore the batch dimension in the return
    # value.
    return torch.cat(((arm_x/1000.).unsqueeze(1), (arm_y/1000.).unsqueeze(1),
            (arm_z/1000.).unsqueeze(1)), dim=1)


def getDistance(record_a, record_b):
    """Return the Euclidean distance of the manipulator in record a and record b"""
    return math.sqrt(sum([(a-b)**2 for a, b in zip(record_a, record_b)]))


def getStateAtNextPosition(reference_record, arm_records, movement_distance, robot_model,
        use_path_distance=False):
    """Get the arm state after the end affector moves the given distance

    Arguments:
        reference_record  ({str: data}): Reference position
        arm_records (list[{str: data}]): ROS data for the robot arm. Search begins at index 0.
        movement_distance (float): Distance in meters desired from the reference position.
        robot_model      (class): String that identifies this robot arm in the interbotix modules.
        use_path_distance (bool): If True, count distance along the path rather than from the
                                  reference record.
    Returns:
        tuple({str: data}, int): The arm record at the desired distance, or None if the records end
                                 before reaching the desired distance. Also returns the index of
                                 this record.
    """
    # Loop until we run out of records or hit the desired movement distance.
    reference_position = getGripperPosition(robot_model, reference_record)
    distance = 0
    idx = 0
    next_record = None
    first_path_distance = reference_record['total_distance']
    path_distance = 0

    # Search for the first record with the desired distance
    while (idx < len(arm_records) and ((not use_path_distance and distance < movement_distance) or
           (use_path_distance and path_distance < movement_distance))):
        next_record = arm_records[idx]
        next_position = getGripperPosition(robot_model, next_record)
        # Find the Euclidean distance from the reference position to the current position
        distance = getDistance(reference_position, next_position)
        path_distance = next_record['total_distance'] - first_path_distance
        idx += 1

    # If we ended up past the end of the records then they don't have anything at the desired
    # distance. If the record doesn't clear the desired distance also return None.
    distance_good = (
        (not use_path_distance and distance >= movement_distance) or
        (use_path_distance and path_distance >= movement_distance))
    if idx >= len(arm_records) or not distance_good:
        return None, None

    return next_record, idx-1

def getCalibrationDiff(manip_yaml, puppet_yaml) -> dict:
    """Find the differences between the manipulator and puppet servos.

    Arguments:
        manip_yaml  (str): Path to manipulator calibration
        puppet_yaml (str): Path to puppet calibration
    Returns:
        correction (dict[str, float]): Correction to apply to manipulator positions.
    """
    manip_values = {}
    puppet_values = {}

    with open(manip_yaml, 'r') as data:
        manip_values = yaml.safe_load(data)

    with open(puppet_yaml, 'r') as data:
        puppet_values = yaml.safe_load(data)

    # Do some sanity checks
    if 0 == len(manip_values) or 0 == len(puppet_values):
        raise RuntimeError(f"Failed to load calibration values.")

    if list(manip_values.keys()) != list(puppet_values.keys()):
        raise RuntimeError(f"Calibration parameters do not match.")

    # Create the corrections and return time.
    corrections = {}
    for joint in manip_values.keys():
        corrections[joint] = puppet_values[joint] - manip_values[joint]

    return corrections

class ArmReplay(InterbotixManipulatorXS):
    # This class has a core type through which we will access robot data.
    # Ros includes
    import rclpy
    from sensor_msgs.msg import JointState

    waist_step = 0.06
    rotate_step = 0.04
    translate_step = 0.01
    gripper_pressure_step = 0.125
    # Speed of the control loop in Hz
    # This must be faster than the rate that position commands are filled
    current_loop_rate = 60
    joint_names = None
    com_position = None
    com_velocity = None
    com_effort = None
    grip_moving = 0

    # The joints required to move the arm
    ordered_joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']

    def __init__(self, puppet_model, puppet_name, corrections, cmd_queue, args=None):
        InterbotixManipulatorXS.__init__(
            self,
            robot_model=puppet_model,
            robot_name=puppet_name,
            moving_time=0.2,
            accel_time=0.1,
            start_on_init=True,
            args=args
        )
        self.position_commands = cmd_queue
        self.corrections = [0,0,0,0,0]
        for joint_name in corrections.keys():
            #print("correction keys are {}, join keys are {}".format(list(corrections.keys()),
            #    self.arm.group_info.joint_names))
            # Don't try to correct the gripper or finger joints, only the arm joints
            if joint_name in self.arm.group_info.joint_names:
                joint_idx = self.arm.group_info.joint_names.index(joint_name)
                self.corrections[joint_idx] = corrections[joint_name]
        print("Joint position corrections from calibration are {}".format(self.corrections))

        self.calibrated_joint_publisher = self.core.create_publisher(
            msg_type=JointState,
            topic="/{}/calibrated_joint_states".format(puppet_name),
            qos_profile=1)

        self.rate = self.core.create_rate(self.current_loop_rate)

        # The actual minimum grip number comes from self.gripper.gripper_info.joint_lower_limits[0],
        # but in the gripper logic only this internal variable is used. This is the most
        # straightforward way to change it.
        self.gripper.left_finger_lower_limit -= 0.0012

        # Go into a relaxed position that does not stress the joints. Move there in 2 seconds so
        # that it isn't too jerky.
        # TODO FIXME From there, go to a relaxed position
        self.core.get_logger().info('Moving arm to a relaxed position.')
        relaxed = [0., 0., 0., 0., 0.]
        relaxed[self.ordered_joint_names.index('shoulder')] = -math.pi/3
        relaxed[self.ordered_joint_names.index('elbow')] = math.pi/3
        relaxed[self.ordered_joint_names.index('wrist_angle')] = math.pi/4
        self.goto(position=relaxed, delay=2.0)
        self.core.get_logger().info('Ready to receive commands.')

    def start_robot(self) -> None:
        try:
            self.start()
            while rclpy.ok():
                self.rate.sleep()
                # publish the current calibrated joint state
                calibrated_state = JointState()
                calibrated_state.velocity = self.core.joint_states.velocity
                calibrated_state.effort = self.core.joint_states.effort
                calibrated_state.position = [self.core.joint_states.position[i] - self.corrections[i] for i in range(len(self.arm.group_info.joint_names))]
                self.calibrated_joint_publisher.publish(calibrated_state)
                while not self.position_commands.empty():
                    position, delay = self.position_commands.get()
                    # Finish if a 'None' is provided for the position
                    if position is None:
                        print("No more positions. Robot shutting down.")
                        self.shutdown()
                        return
                    self.goto(position, delay)
            print("rclpy status is not okay. Robot shutting down.")
        except KeyboardInterrupt:
            print("Interrupt received. Robot shutting down.")
            self.shutdown()

    def cur_grip(self) -> float:
        return self.core.joint_states.position[self.gripper.left_finger_index]

    def goto(self, position, delay) -> None:
        """Update the puppet robot position.

        This is a blocking call.

        Arguments:
            position (list[float]): Joint positions
            delay          (float): Delay in seconds
        Returns:
            (bool): Success of the operation
        """
        # The first five positions correspond to the arm
        arm_joints = 5
        ordered_position = [position[self.ordered_joint_names.index(name)] for name in self.arm.group_info.joint_names]
        # Apply the calibration correction
        for idx in range(len(self.corrections)):
            ordered_position[idx] += self.corrections[idx]
        self.core.get_logger().info('Moving to {} in {}s.'.format(ordered_position[:arm_joints], delay))
        succ = self.arm.set_joint_positions(joint_positions=ordered_position[:arm_joints],
            moving_time=delay, blocking=False)
        print("Movement success: {}".format(succ))

    def _check_joint_limits(self, positions: List[float]) -> bool:
        """
        Ensure the desired arm group's joint positions are within their limits.

        :param positions: the positions [rad] to check
        :return: `True` if all positions are within limits; `False` otherwise
        """
        self.core.get_logger().debug(f'Checking joint limits for {positions=}')
        theta_list = [int(elem * 1000) / 1000.0 for elem in positions]
        speed_list = [
            abs(goal - current) / float(self.moving_time)
            for goal, current in zip(theta_list, self.joint_commands)
        ]
        # check position and velocity limits
        for x in range(self.group_info.num_joints):
            if not (
                self.group_info.joint_lower_limits[x]
                <= theta_list[x]
                <= self.group_info.joint_upper_limits[x]
            ):
                return False
            if speed_list[x] > self.group_info.joint_velocity_limits[x]:
                return False
        return True

    def _check_single_joint_limit(self, joint_name: str, position: float) -> bool:
        """
        Ensure a desired position for a given joint is within its limits.

        :param joint_name: desired joint name
        :param position: desired joint position [rad]
        :return: `True` if within limits; `False` otherwise
        """
        self.core.get_logger().debug(
            f'Checking joint {joint_name} limits for {position=}'
        )
        theta = int(position * 1000) / 1000.0
        speed = abs(
            theta - self.joint_commands[self.info_index_map[joint_name]]
        ) / float(self.moving_time)
        ll = self.group_info.joint_lower_limits[self.info_index_map[joint_name]]
        ul = self.group_info.joint_upper_limits[self.info_index_map[joint_name]]
        vl = self.group_info.joint_velocity_limits[self.info_index_map[joint_name]]
        if not (ll <= theta <= ul):
            return False
        if speed > vl:
            return False
        return True
