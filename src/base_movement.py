#!/usr/bin/env python3
"""
Minimal helper class to command the LoCoBot base *and* arm in Gazebo.

• Base  – /locobot/cmd_vel            (geometry_msgs/Twist)
• Arm   – */command topics per joint* (std_msgs/Float64)

Joint names in the stock Interbotix launch:
    waist, shoulder, elbow, wrist_angle, wrist_rotate
    left_finger, right_finger
Head pan/tilt are included for completeness.
"""
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock


class LocobotCommander:
    def __init__(self):
        # --- initialise ROS --------------------------------------------------
        rospy.init_node("locobot_commander", anonymous=False)
        rospy.loginfo("Waiting for /clock …")
        rospy.wait_for_message("/clock", Clock)
        rospy.loginfo("/clock active – starting")

        # --- publishers ------------------------------------------------------
        self.base_pub = rospy.Publisher("/locobot/cmd_vel",
                                        Twist, queue_size=10)

        joint_topics = [
            "waist", "shoulder", "elbow",
            "wrist_angle", "wrist_rotate",
            "left_finger", "right_finger",
            "pan", "tilt"
        ]
        self.joint_pubs = {
            j: rospy.Publisher(f"/locobot/{j}_controller/command",
                               Float64, queue_size=10)
            for j in joint_topics
        }

    # --------------------------------------------------------------------- #
    # Base motion helpers
    # --------------------------------------------------------------------- #
    def move_base(self, lin=0.0, ang=0.0, duration=0.0, rate_hz=10):
        """Send a Twist for *duration* seconds, then stop."""
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang

        rate = rospy.Rate(rate_hz)
        end_t = rospy.Time.now() + rospy.Duration.from_sec(duration)
        while not rospy.is_shutdown() and rospy.Time.now() < end_t:
            self.base_pub.publish(twist)
            rate.sleep()

        # brake
        self.base_pub.publish(Twist())
        rospy.sleep(0.1)

    # --------------------------------------------------------------------- #
    # Arm helpers
    # --------------------------------------------------------------------- #
    def set_joint(self, joint, value):
        """Publish a single position command once."""
        if joint not in self.joint_pubs:
            rospy.logwarn(f"Unknown joint '{joint}'")
            return
        self.joint_pubs[joint].publish(Float64(data=value))

    def set_arm(self, waist=0.0, shoulder=0.0, elbow=0.0,
                wrist_angle=0.0, wrist_rotate=0.0,
                grip_open=True):
        """Convenience – send one shot to all arm joints + gripper."""
        self.set_joint("waist", waist)
        self.set_joint("shoulder", shoulder)
        self.set_joint("elbow", elbow)
        self.set_joint("wrist_angle", wrist_angle)
        self.set_joint("wrist_rotate", wrist_rotate)

        # very simple open/close (positive opens fingers in default URDF)
        finger_pos = 0.02 if grip_open else 0.0
        self.set_joint("left_finger", finger_pos)
        self.set_joint("right_finger", finger_pos)

    # --------------------------------------------------------------------- #
    # Demo sequence
    # --------------------------------------------------------------------- #
    def demo(self):
        rospy.loginfo("⬆  Drive forward 0.3 m/s for 3 s")
        self.move_base(lin=0.3, duration=3.0)

        rospy.loginfo("🔄 Rotate 0.6 rad/s for 2 s")
        self.move_base(ang=0.6, duration=2.0)

        rospy.loginfo("🤖 Move arm to pick-ready pose")
        self.set_arm(waist=0.5,
                     shoulder=-0.4,
                     elbow=0.4,
                     wrist_angle=1.1,
                     wrist_rotate=0.0,
                     grip_open=True)
        rospy.sleep(2.0)

        rospy.loginfo("🖐️  Close gripper")
        self.set_arm(grip_open=False)
        rospy.sleep(1.0)

        rospy.loginfo("✅ Demo finished")


if __name__ == "__main__":
    try:
        bot = LocobotCommander()
        bot.demo()
    except rospy.ROSInterruptException:
        pass


# #!/usr/bin/env python3
# """
# Helper class to command the LoCoBot base *and* arm in Gazebo using
# a FollowJointTrajectory action for timed arm motions.
# """
# import rospy
# import actionlib
# from geometry_msgs.msg import Twist
# from std_msgs.msg import Float64
# from rosgraph_msgs.msg import Clock
# from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# class LocobotCommander:
#     def __init__(self):
#         # --- initialise ROS --------------------------------------------------
#         rospy.init_node("locobot_commander", anonymous=False)
#         rospy.loginfo("Waiting for /clock …")
#         rospy.wait_for_message("/clock", Clock)
#         rospy.loginfo("/clock active – starting")

#         # --- base publisher -------------------------------------------------
#         self.base_pub = rospy.Publisher("/locobot/cmd_vel",
#                                         Twist, queue_size=10)

#         # --- arm action client ----------------------------------------------
#         # This controller must be running in your launch:
#         # joint_trajectory_controller:
#         #   type: FollowJointTrajectory
#         #   ...
#         self.joint_names = [
#             "waist", "shoulder", "elbow",
#             "wrist_angle", "wrist_rotate",
#             "left_finger", "right_finger",
#             "pan", "tilt"
#         ]
#         self.traj_client = actionlib.SimpleActionClient(
#             '/locobot/joint_trajectory_controller/follow_joint_trajectory',
#             FollowJointTrajectoryAction)
#         rospy.loginfo("Waiting for joint_trajectory action server…")
#         self.traj_client.wait_for_server()
#         rospy.loginfo("Arm action server ready")

#     # --------------------------------------------------------------------- #
#     # Base motion helpers
#     # --------------------------------------------------------------------- #
#     def move_base(self, lin=0.0, ang=0.0, duration=0.0, rate_hz=10):
#         """Send a Twist for *duration* seconds, then stop."""
#         twist = Twist()
#         twist.linear.x = lin
#         twist.angular.z = ang

#         rate = rospy.Rate(rate_hz)
#         end_t = rospy.Time.now() + rospy.Duration.from_sec(duration)
#         while not rospy.is_shutdown() and rospy.Time.now() < end_t:
#             self.base_pub.publish(twist)
#             rate.sleep()

#         # brake
#         self.base_pub.publish(Twist())
#         rospy.sleep(0.1)

#     # --------------------------------------------------------------------- #
#     # Arm motion via action
#     # --------------------------------------------------------------------- #
#     def move_arm(self, positions, duration):
#         """
#         positions: dict of joint_name → target_angle (radians)
#         duration: total seconds for the motion
#         """
#         # build trajectory goal
#         goal = FollowJointTrajectoryGoal()
#         traj = JointTrajectory()
#         traj.joint_names = self.joint_names

#         # one waypoint at time_from_start = duration
#         point = JointTrajectoryPoint()
#         # fill positions in same order as self.joint_names
#         point.positions = [positions.get(j, 0.0) for j in self.joint_names]
#         point.time_from_start = rospy.Duration(duration)

#         traj.points = [point]
#         goal.trajectory = traj

#         # send and wait
#         self.traj_client.send_goal(goal)
#         self.traj_client.wait_for_result(rospy.Duration(duration + 1.0))

#     # --------------------------------------------------------------------- #
#     # Demo sequence
#     # --------------------------------------------------------------------- #
#     def demo(self):
#         # Drive forward
#         rospy.loginfo("⬆  Drive forward 0.3 m/s for 3 s")
#         self.move_base(lin=0.3, duration=3.0)

#         # Rotate in place
#         rospy.loginfo(" Rotate 0.6 rad/s for 2 s")
#         self.move_base(ang=0.6, duration=2.0)

#         # Move arm into pick-ready pose over 2 seconds
#         rospy.loginfo("烙 Move arm to pick-ready pose (2 s)")
#         pick_pose = {
#             'waist': 0.5,
#             'shoulder': -0.4,
#             'elbow': 0.4,
#             'wrist_angle': 1.1,
#             'wrist_rotate': 0.0,
#             'left_finger': 0.02,   # open
#             'right_finger': 0.02,  # open
#             'pan': 0.0,
#             'tilt': 0.0
#         }
#         self.move_arm(pick_pose, duration=2.0)

#         # Close gripper over 1 second
#         rospy.loginfo("️  Close gripper (1 s)")
#         grip_closed = {**pick_pose, 'left_finger': 0.0, 'right_finger': 0.0}
#         self.move_arm(grip_closed, duration=1.0)

#         rospy.loginfo("✅ Demo finished")


# if __name__ == "__main__":
#     try:
#         bot = LocobotCommander()
#         bot.demo()
#     except rospy.ROSInterruptException:
#         pass
