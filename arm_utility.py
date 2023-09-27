# Utilities for arm handling, such as calibration data or controlling an arm, and to analyze robot
# joint states, such as from a rosbag.


import math
import modern_robotics
import rclpy
import yaml

from interbotix_common_modules.angle_manipulation import angle_manipulation as ang
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


def getGripperPosition(robot_model, arm_record):
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
    theta0 = arm_record['position'][arm_record['name'].index('waist')]
    theta1 = arm_record['position'][arm_record['name'].index('shoulder')]
    theta2 = arm_record['position'][arm_record['name'].index('elbow')]
    theta3 = arm_record['position'][arm_record['name'].index('wrist_angle')]
    # The lengths of segments (or effective segments) that are moved by the previous joints,
    # in mm
    segment_G = 104    # Height of the pedestal upon which theta1 rotates
    segment_C = 158    # Effective length from theta1 to theta2
    segment_D = 150    # Length from theta2 to theta3
    segment_H = 170    # Length of the grasper from theta4
    arm_x = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta3 + theta2 + theta1)*segment_H)*math.cos(theta0)
    arm_y = (math.sin(theta1)*segment_C + math.cos(theta2 + theta1)*segment_D + math.cos(theta3 + theta2 + theta1)*segment_H)*math.sin(theta0)
    arm_z = segment_G + math.cos(-theta1)*segment_C + math.sin(-theta1 - theta2)*segment_D + math.sin(-theta1 - theta2 - theta3)*segment_H
    # Return the x,y,z end effector coordinates in meters
    return (arm_x/1000., arm_y/1000., arm_z/1000.)

    # Below is "correct" but gets incorrect results. Instead we will use geometry and hand-measured
    # values for the arm segments. There must be something wrong in the M or Slist matrices
    # (probably the Slist) but the problem isn't immediately apparent.
    # # TODO Since a mini-goal of this project is to handle actuation without calibration we won't be
    # # using this for labels (because it would require calibration for the x,y,z location of the
    # # gripper to be meaningful), but will be using this to determine the moved distance of the
    # # gripper, and from that we will determine the next pose to predict.
    # # The arm joints are separate from the gripper, which is represented by three "joints" even
    # # though it is a single motor.
    # gripper_names = ["wrist_rotate", "gripper", "left_finger", "right_finger"]
    # names = [name for name in arm_record['name'] if name not in gripper_names]
    # joint_positions = [
    #     arm_record['position'][names.index(name)] for name in names
    # ]
    # # 'M' is the home configuration of the robot, Slist has the joint screw axes at the home
    # # position. This should return the end effector position.
    # T = modern_robotics.FKinSpace(robot_model.M, robot_model.Slist, joint_positions)
    # # Return the x,y,z components of the translation matrix.
    # return (T[0][-1], T[1][-1], T[2][-2])


def getDistance(record_a, record_b):
    """Return the Euclidean distance of the manipulator in record a and record b"""
    return math.sqrt(sum([(a-b)**2 for a, b in zip(record_a, record_b)]))


def getStateAtNextPosition(reference_record, arm_records, movement_distance, robot_model):
    """Get the arm state after the end affector moves the given distance

    Arguments:
        reference_record  ({str: data}): Reference position
        arm_records (list[{str: data}]): ROS data for the robot arm. Search begins at index 0.
        movement_distance (float): Distance in meters desired from the reference position.
        robot_model      (class): String that identifies this robot arm in the interbotix modules.
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
    # Search for the first record with the desired distance
    while idx < len(arm_records) and distance < movement_distance:
        next_record = arm_records[idx]
        next_position = getGripperPosition(robot_model, next_record)
        # Find the Euclidean distance from the reference position to the current position
        distance = getDistance(reference_position, next_position)
        idx += 1

    # If we ended up past the end of the records then they don't have anything at the desired
    # distance
    if idx >= len(arm_records):
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

    print("two files are {} and {}".format(manip_yaml, puppet_yaml))
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

        self.rate = self.core.create_rate(self.current_loop_rate)

        # The actual minimum grip number comes from self.gripper.gripper_info.joint_lower_limits[0],
        # but in the gripper logic only this internal variable is used. This is the most
        # straightforward way to change it.
        self.gripper.left_finger_lower_limit -= 0.0012

        # Home position is a non-thunking position, meaning it shouldn't be bumping into anything to
        # start. Move there in 2 seconds so that it isn't too jerky.
        self.arm.go_to_home_pose(moving_time = 2.0, blocking = True)
        self.core.get_logger().info('Arm is in home position {}'.format(self.core.joint_states.position[:5]))

        self.core.get_logger().info('Ready to receive commands.')

    def start_robot(self) -> None:
        try:
            self.start()
            while rclpy.ok():
                self.rate.sleep()
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
        # Apply the calibration correction
        for idx in range(len(self.corrections)):
            position[idx] += self.corrections[idx]
        self.core.get_logger().info('Moving to {} in {}s.'.format(position[:arm_joints], delay))
        succ = self.arm.set_joint_positions(joint_positions=position[:arm_joints],
            moving_time=delay, blocking=False)
        print("Movement success: {}".format(succ))
