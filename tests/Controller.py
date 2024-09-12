class RobotArm:
    def __init__(self):
        # Initialize each joint angle to 0
        self.base_joint_angle = 0
        self.middle_joint_angle = 0
        self.tip_joint_angle = 0

    def move_base_joint(self, angle):
        """Moves the base joint of the robot arm if the angle has changed significantly."""
        if abs(angle - self.base_joint_angle) > 2:
            self.base_joint_angle = angle
            print("Changing base joint angle to: ", self.base_joint_angle)
        else:
            print("Base joint angle change too small; not moving.")

    def move_middle_joint(self, angle):
        """Moves the middle joint of the robot arm if the angle has changed significantly."""
        if abs(angle - self.middle_joint_angle) > 2:
            self.middle_joint_angle = angle
            print("Changing middle joint angle to: ", self.middle_joint_angle)
        else:
            print("Middle joint angle change too small; not moving.")

    def move_tip_joint(self, angle):
        """Moves the tip joint of the robot arm if the angle has changed significantly."""
        if abs(angle - self.tip_joint_angle) > 2:
            self.tip_joint_angle = angle
            print("Changing tip joint angle to: ", self.tip_joint_angle)
        else:
            print("Tip joint angle change too small; not moving.")

    def get_joint_angles(self):
        """Returns the current angles of all joints."""
        return {
            "Base Joint": self.base_joint_angle,
            "Middle Joint": self.middle_joint_angle,
            "Tip Joint": self.tip_joint_angle,
        }
