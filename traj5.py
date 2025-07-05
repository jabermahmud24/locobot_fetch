import math
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from visual_kinematics.RobotSerial import *
# from visual_kinematics.examples.inverse import *
import subprocess
import time
import numpy as np

start_time = time.time()

try:
    import cubic_spline_planner
except ImportError:
    raise

lqr_Q = 1000*np.eye(6)
# print(lqr_Q)
# lqr_Q[0,0] = 1.5
# lqr_R = np.eye(9)
lqr_R = 1*np.eye(9)
dt = 0.1
show_animation = True

class State:

    def __init__(self,world_ref_traj ):   
        # dh_params = np.array([[0.72,0,0,0],
        #                   [0.06,0.117, -0.5*pi, 0],
        #                   [0, 0, 0.5*pi, 0.5*pi],
        #                   [0.219+0.133, 0,  -0.5 * pi, 0],
        #                   [0, 0, +0.5 * pi, 0],
        #                   [0.197+0.1245, 0, -0.5 * pi, 0],
        #                   [0, 0, +0.5 * pi,0],
        #                   [0.1385+0.1665,0, 0, 0]])
        
        dh_params = np.array([[0.72,0,0,0],
                          [0.06,0.117, -0.5*pi, 0],
                          [0, 0, 0.5*pi, 1.57],
                          [0.219+0.133, 0,  0.5 * pi, 0],
                          [0, 0, -0.5 * pi, 0],
                          [0.197+0.1245, 0, -0.5 * pi, 0],
                          [0, 0, +0.5 * pi,0],
                          [0.1385+0.1665,0, 0, 0]])
        
        
    
        self.x_base = 0
        self.y_base = 0.5
        self.yaw_base = 0
        xyz = np.array([[world_ref_traj[0,0]-self.x_base], [world_ref_traj[1,0]-self.y_base], [world_ref_traj[2,0]]])
        abc= np.array([world_ref_traj[3,0], world_ref_traj[4,0], world_ref_traj[5,0]])
        robot = RobotSerial(dh_params)
        end = Frame.from_euler_3(abc, xyz)
        robot.inverse(end)
        self.theta = robot.axis_values.copy()
        self.yaw_base = self.theta[0]
       
     
        robot = RobotSerial(dh_params)
        f = robot.forward(self.theta)
        
        self.x_world = self.x_base + f.t_3_1.reshape([3, ])[0]
        # self.x_world = self.x_base + f.t_3_1.reshape([3, ])[0]*math.cos(self.yaw_base) - f.t_3_1.reshape([3, ])[1]*math.sin(self.yaw_base)
        self.y_world = self.y_base +f.t_3_1.reshape([3, ])[1]
        # self.y_world = self.y_base + f.t_3_1.reshape([3, ])[0]*math.sin(self.yaw_base) + f.t_3_1.reshape([3, ])[1]*math.cos(self.yaw_base)
        self.z_world = f.t_3_1.reshape([3, ])[2]
        self.yaw_world = f.euler_3[2]
        self.pitch_world = f.euler_3[1]
        self.roll_world = f.euler_3[0]


        self.x_body = f.t_3_1.reshape([3, ])[0]
        self.y_body = f.t_3_1.reshape([3, ])[1]
        self.z_body = f.t_3_1.reshape([3, ])[2]
        self.yaw_body = f.euler_3[2]
        self.pitch_body = f.euler_3[1]
        self.roll_body = f.euler_3[0]
   
  

def update(state, ustar, f, dh_params, B, joint_angle_combination):  
    state.theta = state.theta + (dt * ustar[1:].reshape(1,-1))
    state.theta = state.theta.astype(float)
    # print(state.theta)
    # print("================================================")
    state.yaw_base += ustar[1] * dt
    state.yaw_base = float(state.yaw_base)
    
    state.yaw_base = state.yaw_base % (2 * math.pi)
    if state.yaw_base > math.pi:
        state.yaw_base -= 2 * math.pi
    elif state.yaw_base < -math.pi:
        state.yaw_base += 2 * math.pi
    
    
    state.x_base += (ustar[0] * dt * math.cos(state.yaw_base))
    state.x_base = float(state.x_base)
    state.y_base += (ustar[0] * dt * math.sin(state.yaw_base)) 
    state.y_base = float(state.y_base)


    robot = RobotSerial(dh_params)
    f = robot.forward(state.theta)

    state.x_body = f.t_3_1.reshape([3, ])[0]
    state.y_body = f.t_3_1.reshape([3, ])[1]
    state.z_body = f.t_3_1.reshape([3, ])[2]
    state.yaw_body = f.euler_3[2]
    state.pitch_body = f.euler_3[1]
    state.roll_body = f.euler_3[0]
    
    # state.x_world = state.x_base + f.t_3_1.reshape([3, ])[0]*math.cos(state.yaw_base) - f.t_3_1.reshape([3, ])[1]*math.sin(state.yaw_base)
    state.x_world = state.x_base + f.t_3_1.reshape([3, ])[0]
    # state.y_world = state.y_base + f.t_3_1.reshape([3, ])[0]*math.sin(state.yaw_base) + f.t_3_1.reshape([3, ])[1]*math.cos(state.yaw_base)
    state.y_world = state.y_base + f.t_3_1.reshape([3, ])[1]
    state.z_world = f.t_3_1.reshape([3, ])[2]
    state.yaw_world =  f.euler_3[2]
    state.pitch_world =  f.euler_3[1]
    state.roll_world =  f.euler_3[0]

    ee_pose = np.array([[state.x_world], [state.y_world], [state.z_world], [state.yaw_world], [state.pitch_world], [state.roll_world]]) 
    
    return state, ee_pose

def get_B (dh_params, state, joint_angle_combination):
    robot = RobotSerial(dh_params)
    theta = joint_angle_combination
    # theta = np.array(state.theta)

    f = robot.forward(theta)

    jacobian = []
    with open('jacobian_matrix1.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            jacobian.append([cell for cell in row])
    jacobian = np.array(jacobian, dtype= float) 
    

    # phi = math.atan2(state.y_body,state.x_body)
    B = np.zeros((6,9))
    B[:, 1:9] = jacobian * dt
    # B[:, 2:9] = B[:, 2:9] * dt
  
    
    B[0,0] = dt * math.cos(state.yaw_base)
    B[1,0] = dt * math.sin(state.yaw_base)
    
    
    # B[0,0] = dt 
    # B[1,0] = 0
    
    
    # print(B)
    # B[3,1] = dt * 1
    return B, f


def solve_dare(A, B, Q, R, world_ref_traj, n_sum,i, std_dev, state, world_ref_traj_without_noise, D, sigma):

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
    
    horizon = 8

    
    for j in range(horizon-1,-1,-1): 
        
        M = la.inv(R + (B.T @ P_next @ B)) @ B.T
        P_plus = P_next.copy()
        P_next = P_next - P_next @ B @ M @ P_next + Q
        x_base = state.x_base
        y_base = state.y_base
        yaw_base = state.yaw_base
        

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
        p_next = p_next - P_next @ B @ M @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) 
        p_next = p_next  - P_next @ B @ M @ p_next
        
        noise_expectation = sigma @ (D.T @ P_plus @ D)
        trace_noise_expectation = np.trace(noise_expectation)
         
        
        # print(trace_noise_expectation)
        c_next = c_next + (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2)
        
        # c_next  = c_next + trace_noise_expectation
        
        c_next = c_next - (P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next).T @ B @ M @ (P_next @ (world_xyz_in_horizon_1 - world_xyz_in_horizon_2) + p_next)
        c_next = c_next + 2 * (world_xyz_in_horizon_1 - world_xyz_in_horizon_2).T @ p_next
    
    return P_plus, p_plus, c_next, world_xyz_in_horizon_1, world_xyz_in_horizon_2, c_next, P_next, p_next



def dlqr(A, B, Q, R, world_ref_traj, n_sum, i, state,std_dev, world_ref_traj_without_noise, D, sigma):
    P, p, c, rt_c, rt_p, c_t, P_t, p_t = solve_dare(A, B, Q, R, world_ref_traj, n_sum, i, std_dev, state, world_ref_traj_without_noise, D, sigma)
    M = la.inv(R + (B.T @ P @ B)) @ B.T
    
    state_error_world = np.array([state.x_world, state.y_world, state.z_world, state.yaw_world, state.pitch_world, state.roll_world]).reshape(-1,1) - (world_ref_traj[:,i+1].reshape(-1,1) )

    
    ustar = - M @ (P @ (state_error_world + (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1))) + p )
    ustar[1] = ustar[1] % (2 * math.pi)
    if ustar[1] > math.pi:
        ustar[1] -= 2 * math.pi
    elif ustar[1] < -math.pi:
        ustar[1] += 2 * math.pi
        
    # print(ustar)
    
    
    
    
    return ustar, P, p , c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t

def lqr_speed_steering_control(state,lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise,dh_params, i, std_dev, world_ref_traj_without_noise, D, joint_angle_combination, sigma):
    
    A = np.eye(6)
    B, f =  get_B(dh_params, state, joint_angle_combination)
    ustar, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t = dlqr(A, B, lqr_Q, lqr_R, world_ref_traj, n_sum, i, state, std_dev, world_ref_traj_without_noise, D, sigma)
    ustar_cost = ustar.T @ lqr_R @ ustar
    error_cost = state_error_world.T @ lqr_Q @ state_error_world
    return ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost




def check_cost(state, lqr_Q, B, P, p, theta, joint_angle_combination, state_error_world, c, rt_c, rt_p, M, c_t, P_t, p_t):
    # state_error_body = np.vstack((state_error_body_xyz.reshape(-1,1), state_error_body_orientation.reshape(-1,1)))
   
    angle_change_cost = np.sum(np.abs(state.theta - joint_angle_combination))
    cost = 0
    # cost = (state_error_world.T @ lqr_Q @ state_error_world +
    #         state_error_world.T @ P @ state_error_world -
    #         state_error_world.T @ P @ B @ M @ P @ state_error_world -
    #         2 * state_error_world.T @ P @ B @ M @ p +
    #         2 * state_error_world.T @ P @ (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1)) -
    #         2 * state_error_world.T @ P @ B @ M @ P @ (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1)) +
    #         2 * state_error_world.T @ p +
    #         c) 
    
    # In the above equation, you should use c^{t+1}, which is c_plus, but we don't have it.
    # c = c^{t+1}
    # c_t = c^t
    
    cost =  state_error_world.T @ P_t @ state_error_world + 2 * state_error_world.T @ p_t + c_t
    # Please check here, you can use this equation with c^t, which is c_next. 
            
    total_cost = cost + angle_change_cost
    # print(total_cost)
    # print(angle_change_cost)
    return total_cost, angle_change_cost, cost, state_error_world


# def true_cost():
#     pass



def do_simulation(cx, cy, cz, cyaw, cpitch, croll,  world_ref_traj, noise, D, std_dev, world_ref_traj_without_noise,sigma):

    n_sum = np.array([[0],[0],[0]], dtype=float)

    time_ind = np.arange(0.0, len(cx), 1).astype(int).reshape(1, -1)

    state = State(world_ref_traj=world_ref_traj )
    
    x = [state.x_world]
    y = [state.y_world]
    z = [state.z_world]
    yaw = [state.yaw_world]
    pitch = [state.pitch_world]
    roll = [state.roll_world]
    ustar = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    values_for_offset_zero = []
    best_costs = []  
    angle_change_costs = []
    prev_costs = []
    costs_and_combinations = []
    state_error_world_dataset = []
    ustar_dataset = []
    

    for i in range(len(cx)+1):

        if i == len(cx) - 10:
            break
        if i >= 1:
            n_sum = n_sum + noise[:,i-1].reshape(-1,1)

        # print(n_sum)
        
        # dh_params = np.array([[0.72,0,0,0],
        #                   [0.06,0.117, -0.5*pi, 0],
        #                   [0, 0, 0.5*pi, 0.5*pi],
        #                   [0.219+0.133, 0,  -0.5 * pi, 0],
        #                   [0, 0, +0.5 * pi, 0],
        #                   [0.197+0.1245, 0, -0.5 * pi, 0],
        #                   [0, 0, +0.5 * pi,0],
        #                   [0.1385+0.1665,0, 0, 0]])
        
        dh_params = np.array([[0.72,0,0,0],
                          [0.06,0.117, -0.5*pi, 0],
                          [0, 0, 0.5*pi, 1.57],
                          [0.219+0.133, 0,  0.5 * pi, 0],
                          [0, 0, -0.5 * pi, 0],
                          [0.197+0.1245, 0, -0.5 * pi, 0],
                          [0, 0, +0.5 * pi,0],
                          [0.1385+0.1665,0, 0, 0]])
        
       
        world_ref_traj[:,i][0] = np.array([[(world_ref_traj[:,i][0] + n_sum[0])]])
        world_ref_traj[:,i][1] = np.array([[(world_ref_traj[:,i][0] + n_sum[1])]])
        world_ref_traj[:,i][2] = np.array([[(world_ref_traj[:,i][0] + n_sum[2])]])
 

        CT = 1
        if CT ==1:  
            theta = state.theta   
            joint_angle_combination = theta    
            # best_costs = []  
            # angle_change_costs = []
            # prev_costs = []
            ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost  = lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, dh_params, i, std_dev, world_ref_traj_without_noise, D, joint_angle_combination, sigma)
            total_cost, angle_change_cost, cost, state_error_world = check_cost(state, lqr_Q, B, P, p, theta, joint_angle_combination, state_error_world, c, rt_c, rt_p, M, c_t, P_t, p_t)
            best_costs.append(total_cost)
            angle_change_costs.append(angle_change_cost)
            prev_costs.append(cost)
            state_error_world_dataset.append(error_cost)
            ustar_dataset.append(ustar_cost)
       
        # pose optimization start
        
        PO = 0
        if PO == 1:
            costs_and_combinations = []
            theta = state.theta
            joint_angle_combination = theta 
            robot = RobotSerial(dh_params)
               
            # print(joint_angle_combination)
            ll = robot.forward(joint_angle_combination)
            # print(ll[0,3], ll[1,3], ll[2,3])
            
            # print(len(joint_angle_combination))
            ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost  = lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, dh_params, i, std_dev, world_ref_traj_without_noise, D, joint_angle_combination, sigma)
            total_cost, angle_change_cost, cost, state_error_world = check_cost(state,lqr_Q, B, P, p, theta, joint_angle_combination, state_error_world, c, rt_c, rt_p, M, c_t, P_t, p_t)
            costs_and_combinations.append((total_cost, joint_angle_combination, ustar, B, angle_change_cost, cost, state_error_world))
            values_for_offset_zero.append(total_cost)
            
            
            # abc_body = np.array([state.yaw_body, state.pitch_body, state.roll_body])
            
            # xyz_body= np.array([[state.x_body], [state.y_body], [state.z_body]])
            # print(xyz_body)
            # print(len(joint_angle_combination))
            # print(joint_angle_combination.shape)
            # print(joint_angle_combination[6])
            
            
            if (len(joint_angle_combination) == 8):
                # for offset in [0.01, 0.015]:
                for offset in [-0.02, -0.015, -0.01, -0.005, -0.025, -0.03, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
                # for offset in [-0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02]:
                    joint_angle_combination[6] = joint_angle_combination[6] + offset 
                    joint_angle_combination[7] = joint_angle_combination[7] + offset 
                    joint_angle_combination[5] = joint_angle_combination[5] + offset 
                    # ustar, B, f, P, p, c  = lqr_speed_steering_control(state, cx, cy, cz, cyaw, ck, lqr_Q, lqr_R, point, time_ind, noise, joint_angle_combination, dh_params, end, base_x_y_movement, i, x, y, z, yaw)
                    ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost  = lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, dh_params, i, std_dev, world_ref_traj_without_noise, D, joint_angle_combination, sigma)
                    
                    
                    total_cost, angle_change_cost, cost, state_error_world = check_cost(state,lqr_Q, B, P, p, theta, joint_angle_combination, state_error_world, c, rt_c, rt_p, M, c_t, P_t, p_t)
                    # if offset == 0:
                    costs_and_combinations.append((total_cost, joint_angle_combination, ustar, B, angle_change_cost, cost, state_error_world))
                    # print(costs_and_combinations)
            
            if (len(joint_angle_combination) == 1):
                # for offset in [0.01, 0.015]:
                for offset in [-0.02, -0.015, -0.01, -0.005, -0.025, -0.03, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
                # for offset in [-0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02]:
                    joint_angle_combination[0,6] = joint_angle_combination[0,6] + offset
                    joint_angle_combination[0,7] = joint_angle_combination[0,7] + offset 
                    joint_angle_combination[0,5] = joint_angle_combination[0,5] + offset 
                    # ustar, B, f, P, p, c  = lqr_speed_steering_control(state, cx, cy, cz, cyaw, ck, lqr_Q, lqr_R, point, time_ind, noise, joint_angle_combination, dh_params, end, base_x_y_movement, i, x, y, z, yaw)
                    ustar, B, f, P, p, c, state_error_world, c_t, rt_c, rt_p, M, P_t, p_t, ustar_cost, error_cost  = lqr_speed_steering_control(state, lqr_Q, lqr_R, world_ref_traj, n_sum, time_ind, noise, dh_params, i, std_dev, world_ref_traj_without_noise, D, joint_angle_combination, sigma)
                    
                    
                    total_cost, angle_change_cost, cost, state_error_world = check_cost(state,lqr_Q, B, P, p, theta, joint_angle_combination, state_error_world, c, rt_c, rt_p, M, c_t, P_t, p_t)
                    # if offset == 0:
                    costs_and_combinations.append((total_cost, joint_angle_combination, ustar, B, angle_change_cost, cost, state_error_world))
                    # print(costs_and_combinations)
            
        
            best_cost, joint_angle_combination, ustar, B, angle_change_cost, prev_cost, state_error_world = min(costs_and_combinations, key=lambda x: x[0])
            # if joint_angle_combination[0] > 3.14 or joint_angle_combination[1] > 3.14 or joint_angle_combination[2] > 3.14 or joint_angle_combination[3] > 3.14 or joint_angle_combination[4] > 3.14 or joint_angle_combination[5] > 3.14 or joint_angle_combination[6] > 3.14 or joint_angle_combination[7] > 3.14:
            if (len(joint_angle_combination) == 1) and (joint_angle_combination[0,0] > 3.14 or joint_angle_combination[0,1] > 3.14 or joint_angle_combination[0,2] > 3.14 or joint_angle_combination[0,3] > 3.14 or joint_angle_combination[0,4] > 3.14 or joint_angle_combination[0,5] > 3.14 or joint_angle_combination[0,6] > 3.14 or joint_angle_combination[0,7] > 3.14):
                best_cost, joint_angle_combination, ustar, B, angle_change_cost, prev_cost, state_error_world = costs_and_combinations[0]
                
            # if any(joint_angle_combination > 3.14):c
            if len(joint_angle_combination) == 8 and any(joint_angle_combination > 3.14):
                best_cost, joint_angle_combination, ustar, B, angle_change_cost, prev_cost, state_error_world = costs_and_combinations[0]
                
            # print(ustar)
            # print(joint_angle_combination)

            # print('================================================================')
            best_costs.append(best_cost)
            angle_change_costs.append(angle_change_cost)
            prev_costs.append(prev_cost)
            state_error_world_dataset.append(error_cost)
            ustar_dataset.append(ustar_cost)
            
            #pose optimization end
        
   
        state, ee_pose = update(state, ustar, f, dh_params, B,joint_angle_combination)    
        
        # print(state.x_world)    
            
        x.append(state.x_world)
        y.append(state.y_world)
        z.append(state.z_world)
        yaw.append(state.yaw_world)
        pitch.append(state.pitch_world)
        roll.append(state.roll_world)
            
        
        
        
        
    # if i % 1 == 0 and show_animation:
    #     plt.cla()
    #     plt.gcf().canvas.mpl_connect('key_release_event',
    #         lambda event: [exit(0) if event.key == 'escape' else None])

    #     plt.plot(cx, cy, cz, "or", label="Reference Trajectory with Noise")
    #     plt.plot(x, y, z, "ob", label="End-Effector Trajectory")

    #     # Adjust the limits of the axes for each dimension
    #     plt.xlim([min(min(cx), min(x)), max(max(cx), max(x))])
    #     plt.ylim([min(min(cy), min(y)), max(max(cy), max(y))])
    #     # plt.zlim([min(min(cz), min(z)), max(max(cz), max(z))])

    #     plt.grid(True)
    #     plt.legend()
    #     plt.pause(0.0001)  
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
        
    return x, y, z, yaw,pitch, roll, PO, CT, best_costs, angle_change_costs, prev_costs, values_for_offset_zero, state_error_world_dataset, ustar_dataset



def main():

    
    # # ax = [0, 0, 0, 0.65, 1.33, 1.33, 1.33, 2, 2.66, 2.66, 2.66, 3.25, 4,4, 4]  #traj_4
   
    # ax = [0, 0, 0, 1, 1.5, 1.5, 1.75, 2, 2.15, 2.5, 2.75, 3, 3.25, 3.5, 3.75,4]  #traj_5
    
    
    
    # #
    # # ax = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25]


    # # ay = [0, 0.35, 0.7, 0.7, 0.7, 0, -0.7, -0.7, -0.7, 0, 0.7 , 0.7 , 0.7, 0.35, 0]   #traj_4
    # # 
    # ay = [0, 0.3, 0.6, 0.6, 0.6, 0.3, 0.3,  0.3 ,0.4 ,0.5, 0.3, 0.1, 0.15, 0.3, 0.45, 0.3]  #traj_5
    

    # # az = [0.9, 0.94, 0.98, 1.02, 1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95]  #traj_4
    
    
    # az = [0.9, 0.94, 0.98, 1.02, 1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.94]  #traj_5
    
    


    

    # ax = [1.128, 1.5, 1.7, 1.9, 2, 2, 2.1, 2.2, 2.22, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25]
    # ax = [1.128, 1.15, 1.17, 1.19, 1.2, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    # ax = [0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25]
    # ax = [0.9337, 0.96, 1.05, 1.1, 1.2, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95]
    # ax = [0.9337, 0.96, 1.05, 1.1, 1.2, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25]

    # ax = np.linspace(0.4, 4, 80)
    # ay = np.linspace(-0.2, 2, 80)
    # az = np.linspace(0.4,  1.1, 80)

    # ay = [0, 0.05, 0.12, 0.18, 0.3, 0.4, 0.5 ,0.6 ,0.7, 0.8, 0.9, 0.95, 1]
    ay = [0, 0.05, 0.12, 0.18, 0.3, 0.4, 0.5 ,0.6 ,0.7, 0.8, 0.9, 0.95, 1, 1,1,1,1,1,1,1,1,1,1]

    # ay = [0.06, 0, 0, 0, 0.3, 0.7, 1 ,1 ,1, 1, 1, 1, 1]
    # az = [0.8,0.85,0.89,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.0,1.03]
    # az = [0.5,0.55,0.65,0.7,0.78,0.75,0.69, 0.65, 0.59, 0.66,0.69,0.75,0.81]
    # az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03]
    # az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.0,1.03]
    # az = [0.8,0.8,0.8,0.8,0.8,0.8,0.8, 0.8, 0.8,0.8,0.8,0.8,0.8]
    # az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03]

    # az = [0.90,0.90,0.90,0.90,0.90,0.90,0.90, 0.90, 0.90,0.90,0.90,0.90,0.90]
    az = [0.9,0.94,0.98,1.02,1.06,1.1,1.14, 1.14, 1.12, 1.09,1.1,1.07,1.03,0.99,0.95, 0.96, 1.0,1.03, 1.01, 1.01, 0.99, 0.97, 0.97]

    # az = [0.9,0.9,0.9,0.9,0.9,0.9,0.9, 0.9, 0.9,0.9,0.9,0.9,0.9, 0.9,0.9,0.9, 0.9,0.9,0.9, 0.9,0.9,0.9, 0.9]

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
    
    # cyaw1 = np.linspace(0,1,len(cyaw))
    # cyaw2 = np.linspace(0,1,1000)
    # f = interp1d(cyaw1,cyaw)
    # cyaw = f(cyaw2)
    
    # cyaw = np.zeros(300)

    cz1 = np.linspace(0,1,len(cz))
    cz2 = np.linspace(0,1,1000)
    f = interp1d(cz1,cz)
    cz = f(cz2)
    cpitch = np.zeros(len(cx))
    croll = np.zeros(len(cx))


    cpitch1 = np.linspace(0,1,len(cpitch))
    cpitch2 = np.linspace(0,1,1000)
    f = interp1d(cpitch1,cpitch)
    cpitch = f(cpitch2)
    # cyaw_interpolated = f(cyaw2)
    cpitch[:] = 1.57



    cyaw = np.zeros(len(cx))
    croll = np.zeros(len(cx))




    world_ref_traj_without_noise = np.array([cx, cy, cz, cyaw, cpitch, croll])
    
    csv_filename = 'traj5_without_noise_reference_trajectory_data.csv'

    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['cx', 'cy', 'cz','cyaw','cpitch', 'croll']) 
        
        for i in range(len(cx)):
            csv_writer.writerow([cx[i], cy[i], cz[i], cyaw[i], cpitch[i], croll[i]])


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

    
    x, y, z, yaw,pitch, roll, PO, CT, best_costs, angle_change_costs, prev_costs, values_for_offset_zero, state_error_world_dataset, ustar_dataset  = do_simulation(cx, cy, cz, cyaw, cpitch, croll, world_ref_traj, noise, D, std_dev, world_ref_traj_without_noise, sigma)
    
    if PO == 1:
        csv_filename = '1000nhutraj5_0.04_pose_optimized_tracking_trajectory_data.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'z', 'yaw', 'pitch', 'roll']) 
            for i in range(len(x)):
                csv_writer.writerow([x[i], y[i], z[i], yaw[i], pitch[i], roll[i]])
            
    if CT == 1:
        csv_filename = '1000traj5_0.04_coupled_tracking_trajectory_data.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'z', 'yaw', 'pitch', 'roll']) 
            for i in range(len(x)):
                csv_writer.writerow([x[i], y[i], z[i], yaw[i], pitch[i], roll[i]])
                
    data_to_write = []
    data_to_write.append(["Header for opt_total_best_costs"])  # You can change this to an appropriate header
    data_to_write.extend([[total_cost] for total_cost in best_costs])
    
    
    data_to_write.append([])
    data_to_write.append(["Header for opt_angle_change_costs"])  # You can change this to an appropriate header
    data_to_write.extend([[angle_change_cost] for angle_change_cost in angle_change_costs])
    
    data_to_write.append([])
    data_to_write.append(["Header for opt_PREV_best_costs"])  # You can change this to an appropriate header
    data_to_write.extend([[cost] for cost in prev_costs])
    
    data_to_write.append([])
    data_to_write.append(["Header for cost_offset_zero"])  # You can change this to an appropriate header
    data_to_write.extend([[total_cost_without] for total_cost_without in values_for_offset_zero])
    
    data_to_write.append([])
    data_to_write.append(["Header for ustar"])  # You can change this to an appropriate header
    data_to_write.extend([[ustarcost] for ustarcost in ustar_dataset])
    
    data_to_write.append([])
    data_to_write.append(["Header for error"])  # You can change this to an appropriate header
    data_to_write.extend([[state_error_world] for state_error_world in state_error_world_dataset])
    
    
    

    # Write data to CSV
    if CT == 1:
        with open('1000traj5_0.04_coupled_candidate_cost.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_to_write) 
    if PO == 1:
        with open('1000nhutraj5_0.04_PO_candidate_cost.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data_to_write) 
    
if __name__ == '__main__':
    main()
    