# Utilities for arm handling, such as calibration data or controlling an arm.


import yaml
import rclpy

from interbotix_common_modules.angle_manipulation import angle_manipulation as ang
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


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
    print("two values are {} and {}".format(manip_values, puppet_values))

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
        print("Corrections are {}".format(corrections))
        for joint_name in corrections.keys():
            print("correction keys are {}, join keys are {}".format(list(corrections.keys()),
                self.arm.group_info.joint_names))
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
