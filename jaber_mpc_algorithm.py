#!/usr/bin python3
import rospy, actionlib
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import LinkStates
from threading import Lock, Thread
import pandas as pd

import math
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
#from visual_kinematics.RobotSerial import *
# from visual_kinematics.examples.inverse import *
import subprocess
import time
import numpy as np
import tf



start_time = time.time()

try:
    import cubic_spline_planner
except ImportError:
    raise


lqr_Q = 5*np.eye(6)
lqr_R = 1*np.eye(9)
dt = 0.1
show_animation = True



# MAX_JOINT_VEL = np.array([1, 3, 3, 3, 3, 3, 3, 3])
MAX_JOINT_VEL = np.array([0.1, 1.25, 1.45, 1.57, 1.52, 1.57, 2.26, 2.26])
JOINT_ACTION_SERVER = 'arm_with_torso_controller/follow_joint_trajectory'
#JOINT_NAMES = names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
# JOINT_NAMES = names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
JOINT_NAMES = names = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
_joint_states = dict()
link_name_left = "fetch::l_gripper_finger_link"
link_name_right = "fetch::r_gripper_finger_link"
base_link_pose = None
link_pose_left = None
link_pose_right = None
lock = Lock()


def callback(msg):    
    # Subscribes to joint_states, returning a list of lists of joint positions (floats)
    lock.acquire()
    for i, name in enumerate(msg.name):
        if i >= len(msg.position):
            continue
        _joint_states[name] = msg.position[i]
    lock.release()

def get_latest_joint_state():
    """
    Returns: A list of the joint values. Values may be None if we do not
        have a value for that joint yet.
    """
    lock.acquire()

    ret = None
    if all(name in _joint_states for name in names):
        ret = [_joint_states[name] for name in names]
    lock.release()
    return ret if ret else None

def calc_dt(q1, q0):
    return (5.0*np.absolute(np.subtract(q1, q0))) / (8.0*MAX_JOINT_VEL)

def calculate_optimal_dts(waypoints):
    dts = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    for i, w in enumerate(waypoints):
        if i < len(waypoints)-1:
            dts.append(calc_dt(waypoints[i+1], waypoints[i]))
    optimal_dts = [max(dt) for dt in dts]
    return [sum(optimal_dts[:(i+1)]) for i, dt in enumerate(optimal_dts)]


def appendwp(lst, wp):
    lastwp = lst[-1]
    #print(wp[-1])
    if any([abs(lastwp[j] - wp[j]) > np.pi for j in range(8)]):
        #print(lastwp[j])
        appendwp(lst, [(lastwp[j] + wp[j])/2 for j in range(8)])
        appendwp(lst, wp)
    else:
        lst.append(wp)


def callback_link(msg):
    # Subscribes to gazebo/link_states, returning a list of lists of link positions (floats)
    global link_pose_right, link_pose_left, base_link_pose
    try:
      ind_left = msg.name.index(link_name_left)
      link_pose_left = msg.pose[ind_left]

      ind_right = msg.name.index(link_name_right)
      link_pose_right = msg.pose[ind_right]

      base_ind = msg.name.index('fetch::base_link')
      base_link_pose = msg.pose[base_ind]
    except ValueError:
      pass


def get_tf_matrices(listener):
        transformations = {}
        try:
          (trans,rot) = listener.lookupTransform('/base_link', '/shoulder_pan_link', rospy.Time(0))
          transformations["baseLinkToShoulderPan"] = tf.TransformerROS.fromTranslationRotation(listener,translation=trans, rotation=rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/shoulder_lift_link', rospy.Time(0))
          transformations["baseLinkToShoulderLift"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/upperarm_roll_link', rospy.Time(0))
          transformations["baseLinkToUpperarmRoll"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/elbow_flex_link', rospy.Time(0))
          transformations["baseLinkToElbowFlex"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/forearm_roll_link', rospy.Time(0))
          transformations["baseLinkToForearmRoll"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/wrist_flex_link', rospy.Time(0))
          transformations["baseLinkToWristFlex"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/wrist_roll_link', rospy.Time(0))
          transformations["baseLinkToWristRoll"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/l_gripper_finger_link', rospy.Time(0))
          transformations["baseLinkToLeftFinger"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/r_gripper_finger_link', rospy.Time(0))
          transformations["baseLinkToRightFinger"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        return transformations
        
# To get B Matrix, we need transformation matrix of each joint with respect to base in robot frame. 
# But if we can define DH parameters (Robot model) perfectly, we do not need these transformation matrices from Gazebo.


# End-Effector state, current joint angle combination from Gazebo

def lqr_speed_steering_control(state, world_ref_traj, i, world_ref_traj_without_noise, D, joint_angle_combination, sigma, base_heading_angle):
    
    lqr_Q = 500*np.eye(6)
    lqr_R = 1*np.eye(9)
    A = np.eye(6)
    B =  get_B(state, joint_angle_combination, base_heading_angle)
    
    ustar = dlqr(A, B, lqr_Q, lqr_R, world_ref_traj, i, state, world_ref_traj_without_noise, D, sigma)
    return ustar

def get_B (state, joint_angle_combination, base_heading_angle):
    

    # Need Jacobian Here for each time step:

    tf_matrices = get_tf_matrices(listener)
    link_01 = tf_matrices["baseLinkToShoulderPan"]
    z31_01 = link_01[[0, 1, 2], 2].reshape([3, ])
    t31_01 = link_01[[0, 1, 2], 3].reshape([3, ])

    link_02 = tf_matrices["baseLinkToShoulderLift"]
    z31_02 = link_02[[0, 1, 2], 2].reshape([3, ])
    t31_02 = link_02[[0, 1, 2], 3].reshape([3, ])

    link_03 = tf_matrices["baseLinkToUpperarmRoll"]
    z31_03 = link_03[[0, 1, 2], 2].reshape([3, ])
    t31_03 = link_03[[0, 1, 2], 3].reshape([3, ])

    link_04 = tf_matrices["baseLinkToElbowFlex"]
    z31_04 = link_04[[0, 1, 2], 2].reshape([3, ])
    t31_04 = link_04[[0, 1, 2], 3].reshape([3, ])

    link_05 = tf_matrices["baseLinkToForearmRoll"]
    z31_05 = link_05[[0, 1, 2], 2].reshape([3, ])
    t31_05 = link_05[[0, 1, 2], 3].reshape([3, ])

    link_06 = tf_matrices["baseLinkToWristFlex"]
    z31_06 = link_06[[0, 1, 2], 2].reshape([3, ])
    t31_06 = link_06[[0, 1, 2], 3].reshape([3, ])

    link_07 = tf_matrices["baseLinkToWristRoll"]
    z31_07 = link_07[[0, 1, 2], 2].reshape([3, ])
    t31_07 = link_07[[0, 1, 2], 3].reshape([3, ])
    
    link_08 = tf_matrices["baseLinkToRightFinger"]
    z31_08 = link_08[[0, 1, 2], 2].reshape([3, ])
    t31_08 = link_08[[0, 1, 2], 3].reshape([3, ])

    jacobian = np.zeros([6, 8])
    jacobian[0:3, 0] = np.cross(np.array([0., 0., 1.]), t31_08)
    jacobian[3:6, 0] = np.array([0., 0., 1.])

    jacobian[0:3, 1] = np.cross(z31_01, (t31_08 - t31_01))
    jacobian[3:6, 1] = z31_01

    jacobian[0:3, 2] = np.cross(z31_02, (t31_08 - t31_02))
    jacobian[3:6, 2] = z31_02

    jacobian[0:3, 3] = np.cross(z31_03, (t31_08 - t31_03))
    jacobian[3:6, 3] = z31_03

    jacobian[0:3, 4] = np.cross(z31_04, (t31_08 - t31_04))
    jacobian[3:6, 4] = z31_04
    
    jacobian[0:3, 5] = np.cross(z31_05, (t31_08 - t31_05))
    jacobian[3:6, 5] = z31_05

    jacobian[0:3, 6] = np.cross(z31_06, (t31_08 - t31_06))
    jacobian[3:6, 6] = z31_06

    jacobian[0:3, 7] = np.cross(z31_07, (t31_08 - t31_07))
    jacobian[3:6, 7] = z31_07

    
    B = np.zeros((6,9))
    B[:, 1:9] = jacobian
    # B[:, 1:9] = jacobian * dt
    B[0,0] = dt * math.cos(base_heading_angle)  # We need heading angle of the base
    B[1,0] = dt * math.sin(base_heading_angle)  # We need heading angle of the base
    
    return B
        
def dlqr(A, B, Q, R, world_ref_traj, i, state, world_ref_traj_without_noise, D, sigma):
    
    P, p, rt_c, rt_p = solve_dare(B, Q, R, i, world_ref_traj_without_noise, D, sigma)
    M = la.inv(R + (B.T @ P @ B)) @ B.T
    state_error_world = state - (world_ref_traj[:,i+1].reshape(-1,1) )
    # state_error_world = np.array([state.x_world, state.y_world, state.z_world, state.yaw_world, state.pitch_world, state.roll_world]).reshape(-1,1) - (world_ref_traj[:,i+1].reshape(-1,1) )
    ustar = - M @ (P @ (state_error_world + (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1))) + p )
        
    return ustar
   

def solve_dare(B, Q, R, i, world_ref_traj_without_noise, D, sigma):

    P = Q
    P_next = Q

    p = np.array([[0], 
        [0],
        [0],
        [0],
        [0],
        [0]])
    p_next = np.array([[0], 
        [ 0],
        [0],
        [0],
        [0],
        [0]])
    
    c = 0
    c_next = 0
    
    horizon = 5
        
    for j in range(horizon-1,-1,-1): 
        
        M = la.inv(R + (B.T @ P_next @ B)) @ B.T
        P_plus = P_next.copy()
        P_next = P_next - P_next @ B @ M @ P_next + Q      

        world_xyz_in_horizon_1 =([world_ref_traj_without_noise[:,i+j+1][0]], 
                                 [world_ref_traj_without_noise[:,i+j+1][1]], 
                                 [world_ref_traj_without_noise[:,i+j+1][2]] )  

        world_xyz_in_horizon_2 = ([world_ref_traj_without_noise[:,i+j+2][0]],
                                  [world_ref_traj_without_noise[:,i+j+2][1]],
                                  [world_ref_traj_without_noise[:,i+j+2][2]] ) 
        
        world_xyz_in_horizon_1_orientation = np.array([
            [world_ref_traj_without_noise[:,i+j+1][3]],
            [world_ref_traj_without_noise[:,i+j+1][4]],
            [world_ref_traj_without_noise[:,i+j+1][5]]
        ])

        world_xyz_in_horizon_1 = np.vstack((world_xyz_in_horizon_1, world_xyz_in_horizon_1_orientation))
        
        
        world_xyz_in_horizon_2_orientation = np.array([
            [world_ref_traj_without_noise[:,i+j+2][3]],
            [world_ref_traj_without_noise[:,i+j+2][4]],
            [world_ref_traj_without_noise[:,i+j+2][5]]
        ])

        world_xyz_in_horizon_2 = np.vstack((world_xyz_in_horizon_2, world_xyz_in_horizon_2_orientation))
        
        p_plus = p_next.copy()
        p_next = p_next  + P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) 
        p_next = p_next - P_next @ B @ M @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) 
        # p_next = p_next - P_next @ B @ M @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) 
        p_next = p_next  - P_next @ B @ M @ p_next
            
        noise_expectation = sigma @ (D.T @ P_plus @ D)
        trace_noise_expectation = np.trace(noise_expectation)

        c_next = c_next + (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2)
        c_next  = c_next + trace_noise_expectation
        c_next = c_next - (P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next).T @ B @ M @ (P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next)
        c_next = c_next + 2 * (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ p_next
    
    return P_plus, p_plus, world_xyz_in_horizon_1, world_xyz_in_horizon_2


def generate_trajectory(ax, ay,az):

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
    ax, ay, ds=0.1)
    cx5, cz, cyaw5, ck5, s = cubic_spline_planner.calc_spline_course(
    ax, az, ds=0.1)

    cx1 = np.linspace(0,1,len(cx))
    cx2 = np.linspace(0,1,1000)
    f = interp1d(cx1,cx)
    cx = f(cx2)

    cy1 = np.linspace(0,1,len(cy))
    cy2 = np.linspace(0,1,1000)
    f = interp1d(cy1,cy)
    cy = f(cy2)

    cz1 = np.linspace(0,1,len(cz))
    cz2 = np.linspace(0,1,1000)
    f = interp1d(cz1,cz)
    cz = f(cz2)


    cyaw1 = np.linspace(0,1,len(cyaw))
    cyaw2 = np.linspace(0,1,1000)
    f = interp1d(cyaw1,cyaw)
    cyaw = f(cyaw2)
    # cyaw_interpolated = f(cyaw2)
    cyaw[:] = 0.71



    cpitch = np.zeros(len(cx))
    croll = np.zeros(len(cx))
    # croll[:] = 1.57
    # cyaw= np.zeros(len(cx))






    world_ref_traj_without_noise = np.array([cx, cy, cz, cyaw, cpitch, croll])



    cx = np.array(cx) 
    mean = 0  
    std_dev = 0.0
    np.random.seed(42)
    noise1 = np.random.normal(mean, std_dev, size=cx.shape)
    noise2 = np.random.normal(mean, std_dev, size=cx.shape)
    noise3 = np.random.normal(mean, std_dev, size=cx.shape)
    sigma = np.array([[0.015*std_dev, 0, 0], 
        [0, 0.025*std_dev, 0],
        [0, 0, 0.015*std_dev]])
    D = np.array([[1, 0, 0], 
        [0, 1, 0],
        [0, 0, 1],
        [0,0,0],
        [0,0,0],
        [0,0,0]])
    cx = cx + (noise1 * 0.015) 
    cy = cy + (noise2 * 0.025)
    cz= cz + (noise3 * 0.015)
    noise = np.array([(noise1 * 0.015), (noise2*0.025), (noise3*0.015)])
    # cpitch = np.zeros(len(cx))
    # croll = np.zeros(len(cx))
    world_ref_traj = np.array([cx, cy, cz, cyaw, cpitch, croll])

    csv_filename = 'gazebo_reference_trajectory_data.csv'

    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['cx', 'cy', 'cz','cyaw','cpitch', 'croll']) 
        
        for i in range(len(cx)):
            csv_writer.writerow([cx[i], cy[i], cz[i], cyaw[i], cpitch[i], croll[i]])


    return world_ref_traj, world_ref_traj_without_noise, cx, cy, cz, noise, std_dev, D, sigma




def execute_waypoints_trajectory(waypoints, t, velocities=None):


    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names.extend(JOINT_NAMES)
    for i, w in enumerate(waypoints):
        point = JointTrajectoryPoint()
        goal.trajectory.points.append(point)
        # goal.trajectory.points[i].time_from_start = rospy.Duration(0.25)
        goal.trajectory.points[i].time_from_start = rospy.Duration(t[i])
        # rospy.loginfo("waypoint " + str(i) + " = " + str(waypoints[i]))
        # rospy.loginfo("waypoint " + str(i) + " = " + str(waypoints[i]))
        for j, p in enumerate(waypoints[i]):
            goal.trajectory.points[i].positions.append(waypoints[i][j])
        
        if velocities.any():
            if i < len(velocities):
                goal.trajectory.points[i].velocities.extend(velocities[i])
            else:
                rospy.logwarn("Insufficient velocity data for waypoint " + str(i))
                goal.trajectory.points[i].velocities.extend([0.0] * len(JOINT_NAMES))
        else:
            goal.trajectory.points[i].velocities.extend([0.0] * len(JOINT_NAMES))
        waypoints = np.array(waypoints)
        
        # goal.trajectory.points[i].accelerations.append(waypoints/0.01)
        goal.trajectory.points[i].accelerations.append(0)

    
    _joint_client.send_goal(goal)
    # _joint_client.wait_for_result(rospy.Duration(0.0))





def publish_base_velocity(base_velocities, durations):
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    twist = Twist()

    rate = rospy.Rate(10)  # 20 Hz ; how fast the base is responding with the given commands
    total_duration = sum(durations)
    start_time = rospy.Time.now()
    current_duration = 0

    for velocity, duration in zip(base_velocities, durations):
        twist.linear.x = velocity[0]
        twist.angular.z = velocity[1]
        while current_duration < duration:
            velocity_publisher.publish(twist)
            rate.sleep()
            current_duration = (rospy.Time.now() - start_time).to_sec()
        current_duration = 0

    # Stop the base by publishing a zero velocity command
    twist.linear.x = 0.0
    twist.angular.z = 0.0
    velocity_publisher.publish(twist)
##################################



if __name__ == '__main__':


    # Test 1

    # ax = [0, 0.1, 0.2, 0.65, 1, 1.33, 1.66, 2, 2.33, 2.66, 3, 3.25, 3.5,3.8, 4]  #traj_4
    # ax= [x + 0.9337 for x in ax]
    # ay = [0, 0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.7, 0.9, 1.1 , 1.5 , 1.3, 1.1, 1]   #traj_4
    # az = [0.9, 0.94, 0.98, 1.02, 1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95]  #traj_4

   
    # Test 2

    ax = [0, 0.25, 0.5, 0.4, 0.25, 0.45, 0.65, 0.85, 0.75, 0.65, 0.75, 1.75, 2.5, 3.25, 3.5, 3.75, 4, 4.25]
    ax= [x + 0.9337 for x in ax]
    ay = [0, 0, 0.05, 0.35, 0.45, 0.35, 0 ,0 ,0, 0, -0.07, -0.15, -0.35, -0.45, -0.35, -0.15, 0, 0.1]
    az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.02,1.03]


    # # Test 3

    # ax = [0.9337, 0.96, 1.05, 1.1, 1.15, 1.2, 1.25, 1.30, 1.35, 1.40, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2, 2.5, 3, 3.4, 3.8, 4.2, 4.5, 5, 5.4, 5.9]
    # ay = [0, 0.05, 0.12, 0.18, 0.24, 0.3, 0.4, 0.45, 0.5, 0.55 ,0.6 ,0.7, 0.8, 0.9, 0.95, 1, 1,1,1,1,1,1,1,1,1,1] # 23
    # az = [0.9, 0.9, 0.92,0.94, 0.97, 0.99,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.0,1.03, 1.01, 1.01, 0.99, 0.97, 0.97]


    # Test 4

    # ax = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25] 
    # ax= [x + 0.9337 for x in ax]
    # ay = [0, 0.3, 0.6, 0.6, 0.6, 0.3, 0.3,  0.3 ,0.4 ,0.5, 0.3, 0.1, 0.15, 0.3, 0.45, 0.55, 0.65, 0.75]
    # az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.02,1.03]
    
    
    # Test 5  --- 200

    # ax = [0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.65, 0.75, 0.85, 1.05, 1.3, 1.75, 2.5, 3.25, 3.5, 3.75, 4, 4.25]
    # # ax = [0, 0.25, 0.3, 0.4, 0.25, 0.45, 0.65, 0.85, 0.75, 0.65, 0.75, 1.75, 2.5, 3.25, 3.5, 3.75, 4, 4.25]
    # ax= [x + 0.9337 for x in ax]
    # ay = [0, -0.1, -0.2, -0.3, -0.35, -0.4, -0.6 ,-0.8 ,-0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.3, 0.5, 0.55]
    # az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.02,1.03]

    # ax = [0, 4, 8]
    # # ax = [0, 0.25, 0.3, 0.4, 0.25, 0.45, 0.65, 0.85, 0.75, 0.65, 0.75, 1.75, 2.5, 3.25, 3.5, 3.75, 4, 4.25]
    # ax= [x + 0.9337 for x in ax]
    # ay = [0, -1, 1]
    # az = [0.8, 0.8, 0.8]








    world_ref_traj, world_ref_traj_without_noise, cx, cy, cz, noise, std_dev, D, sigma = generate_trajectory(ax, ay, az)



    rospy.init_node('listener', anonymous=True)   

    rospy.Subscriber("/joint_states", JointState, callback)
    rospy.Subscriber("/gazebo/link_states", LinkStates, callback_link)
    
    listener = tf.TransformListener()


    starting_joint_states = None
    while starting_joint_states is None:
        starting_joint_states = get_latest_joint_state()
    
    
    global _joint_client 
    _joint_client = actionlib.SimpleActionClient(
        JOINT_ACTION_SERVER, FollowJointTrajectoryAction)
    _joint_client.wait_for_server(timeout=rospy.Duration(300))

    waypoints = [starting_joint_states]

    n_sum = np.array([[0],[0],[0]], dtype=float)


    x = [link_pose_left.position.x]
    y = [link_pose_left.position.y]
    z = [link_pose_left.position.z]
    yaw = [link_pose_left.orientation.x]
    pitch = [link_pose_left.orientation.y]
    roll = [link_pose_left.orientation.z]

    # for i in range(215):
    for i in range(len(cx)+1):
            print(i)
    # for i in range(12):

            if i == len(cx) - 10:
                break
            if i >= 1:
                n_sum = n_sum + noise[:,i-1].reshape(-1,1)

            world_ref_traj[:,i][0] = np.array([[(world_ref_traj[:,i][0] + n_sum[0])]])
            world_ref_traj[:,i][1] = np.array([[(world_ref_traj[:,i][0] + n_sum[1])]])
            world_ref_traj[:,i][2] = np.array([[(world_ref_traj[:,i][0] + n_sum[2])]])
            
            state = np.array([[link_pose_left.position.x],
                            [link_pose_left.position.y],
                            [link_pose_left.position.z],
                            [link_pose_left.orientation.x],
                            [link_pose_left.orientation.y],
                            [link_pose_left.orientation.z]])
            joint_angle_combination = np.array([_joint_states["shoulder_pan_joint"],
                                                _joint_states["shoulder_lift_joint"],
                                                _joint_states["upperarm_roll_joint"],
                                                _joint_states["elbow_flex_joint"],
                                                _joint_states["forearm_roll_joint"],
                                                _joint_states["wrist_flex_joint"],
                                                _joint_states["wrist_roll_joint"]])
            
            base_heading_angle = base_link_pose.orientation.z

            ustar = lqr_speed_steering_control(state, world_ref_traj, i, world_ref_traj_without_noise, D, joint_angle_combination, sigma, base_heading_angle)
            # print(ustar)
            ustar = np.insert(ustar, 2, 0, axis=0)
            print('------------------')
            arm_velocity = ustar[2:, 0]. reshape(1, -1)
            base_velocity = ustar[:2, 0]. reshape(1, -1)

            with open(f'arm_velocity_iteration.txt', 'a') as arm_file:
                np.savetxt(arm_file, arm_velocity)
    
            # Save base_velocity
            with open(f'base_velocity_iteration.txt', 'a') as base_file:
                np.savetxt(base_file, base_velocity)

            print(arm_velocity)


            dt = 0.1
            optimal_dts = [0.1]


            starting_joint_states = get_latest_joint_state()
            
            waypoints = [starting_joint_states]
            print((arm_velocity * dt))

            # if i != 50 & 100 & 200 & 150:
            # # if i != 20 & 40 & 60 & 80 & 100 & 120 & 160 & 140 & 180 & 200:
                
            #     # waypoint = [[0, 0, 0.5, 0, -1.5, 0, 0.9, 1.57]]
            #     waypoint = waypoints + (arm_velocity * dt)
            #     execute_waypoints_trajectory(waypoint, optimal_dts, arm_velocity)
            # if i == 50:
            # # if i == 20 & 100 & 200:
            #     # waypoint_PO_1 = np.array([0, 0.1, 0.5, 0.2, -1.5, 0, 0.9, 1.45])
            #     waypoint_PO_1 = waypoints + np.array([0, 0.1, 0, 0.2, 0, 0, 0, -0.12])
            #     execute_waypoints_trajectory(waypoint_PO_1, optimal_dts, arm_velocity)
            #     waypoints = waypoint_PO_1 

            # if i == 100:
            # # if i == 40 & 120:
            #     # waypoint_PO_1 = np.array([0, 0.1, 0.5, 0.2, -1.5, 0, 0.9, 1.45])
            #     waypoint_PO_2 = waypoints + np.array([0, 0, 0.1, 0, -0.15, 0, 0, 0])
            #     execute_waypoints_trajectory(waypoint_PO_2, optimal_dts, arm_velocity)
            #     waypoints = waypoint_PO_2

            # if i == 150:
            # # if i == 60 & 400:
            #     # waypoint_PO_1 = np.array([0, 0.1, 0.5, 0.2, -1.5, 0, 0.9, 1.45])
            #     waypoint_PO_3 = waypoints + np.array([0, -0.15, 0, -0.25, 0, 0, 0, 0.08])
            #     execute_waypoints_trajectory(waypoint_PO_3, optimal_dts, arm_velocity)
            #     waypoints = waypoint_PO_3  
                
            # if i == 200:
            # # if i == 80 & 200:
            #     # waypoint_PO_1 = np.array([0, 0.1, 0.5, 0.2, -1.5, 0, 0.9, 1.45])
            #     waypoint_PO_4 = waypoints + np.array([0, -0.1, 0, -0.25, 0, -0.24, 0, 0.37])
            #     execute_waypoints_trajectory(waypoint_PO_4, optimal_dts, arm_velocity)
            #     waypoints = waypoint_PO_4  


       
            waypoint = waypoints + (arm_velocity * dt)
            execute_waypoints_trajectory(waypoint, optimal_dts, arm_velocity)

          


            duration = [0.1]
            publish_base_velocity(base_velocity, duration)
      

            x.append(link_pose_left.position.x)
            y.append(link_pose_left.position.y)
            z.append(link_pose_left.position.z)
        
            yaw.append(link_pose_left.orientation.x)
            pitch.append(link_pose_left.orientation.y)
            roll.append(link_pose_left.orientation.z)
        

            if i == (len(cx) - 12): 
                print(link_pose_left.position)
                print(link_pose_left.orientation)

    csv_filename = 'coupled_tracking_trajectory_data.csv'
    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x', 'y', 'z', 'yaw', 'pitch', 'roll']) 
        for i in range(len(x)):
            csv_writer.writerow([x[i], y[i], z[i], yaw[i], pitch[i], roll[i]])


    




            
