import sys 
sys.path.append("/viscam/u/jiamanli/github/egoego_private")
import os 
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import yaml
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import shutil 

import torch.utils.data as data

import scipy.io
import scipy.ndimage

import tqdm

import cv2
import math
import time
import torch

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from matplotlib.animation import FuncAnimation
from PIL import Image 

import colorsys

from utils.extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
# from utils.extrinsic2pyramid.util.camera_parameter_loader import CameraParameterLoader

def camera_pose_vis(head_trans, head_rot_mat, dest_img_path, color=None):
    # head_trans: K X T X 3 
    # head_rot_mat: K X T X 3 X 3 
    head_trans = head_trans - head_trans[:, 0:1, :] 

    visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [0, 3])
    
    num_scenes = head_trans.shape[0]
    num_steps = head_trans.shape[1]
    for s_idx in range(num_scenes):
        for idx in range(num_steps):
            extrinsic = np.zeros((4, 4)) # 4 X 4 
            extrinsic[:3, :3] = head_rot_mat[s_idx, idx]
            extrinsic[:3, 3] = head_trans[s_idx, idx]
            extrinsic[3, 3] = 1 
            if color is None:
                if s_idx == 0:
                    curr_color = 'g'
                else:
                    curr_color = 'r'
                # curr_color = plt.cm.rainbow(s_idx / num_scenes)
            else:
                curr_color = color
            visualizer.extrinsic2pyramid(extrinsic, curr_color, 1)

    # visualizer.colorbar(num_scenes)
    visualizer.save(dest_img_path)

def gen_head_pose_trajectory_for_vis(head_trans, head_orientation):
    # head_trans: T X 3, head_orientation: T X 3 X 3  
    if not torch.is_tensor(head_trans):
        head_trans = torch.from_numpy(head_trans).float()
    
    if not torch.is_tensor(head_orientation):
        head_orientation = torch.from_numpy(head_orientation).float()

    head_trans = head_trans.cpu().float()
    head_orientation = head_orientation.cpu().float() 

    init_x = np.asarray([1, 0, 0])
    init_y = np.asarray([0, 1, 0])
    init_z = np.asarray([0, 0, 1]) 
    init_center = np.asarray([0, 0, 0])

    timesteps = head_trans.shape[0]
    init_x = torch.from_numpy(init_x)[None, :, None].repeat(timesteps, 1, 1).float() # T X 3 X 1
    init_y = torch.from_numpy(init_y)[None, :, None].repeat(timesteps, 1, 1).float() 
    init_z = torch.from_numpy(init_z)[None, :, None].repeat(timesteps, 1, 1).float() 

    center_arr = head_trans # T X 3 
    x_arr = torch.matmul(head_orientation, init_x).squeeze(-1) + center_arr # T X 3 
    y_arr = torch.matmul(head_orientation, init_y).squeeze(-1) + center_arr
    z_arr = torch.matmul(head_orientation, init_z).squeeze(-1) + center_arr 

    channels = torch.cat((center_arr[:, None, :], x_arr[:, None, :], y_arr[:, None, :], z_arr[:, None, :]), dim=1) # T X 4 X 3 
    
    return channels 

def vis_single_head_pose_traj(head_trans, head_orientation, dest_vis_path):
    channels = gen_head_pose_trajectory_for_vis(head_trans, head_orientation)
    show_head_pose_animation(channels[None].data.cpu().numpy(), dest_vis_path)

def vis_multiple_head_pose_traj(head_trans_list, head_orientation_list, dest_vis_path):
    channels_list = []
    for s_idx in range(len(head_trans_list)):
        channels = gen_head_pose_trajectory_for_vis(head_trans_list[s_idx], head_orientation_list[s_idx])
        channels_list.append(channels[None]) 

    channels_all = torch.cat(channels_list, dim=0).data.cpu().numpy() # K X T X 4 X 3 
    show_head_pose_animation(channels_all, dest_vis_path)

def show_head_pose_animation(channels, dest_vis_path):
    # channels: K X T X 4 X 3
    # if k == 1, visualize for a single head pose trajectory, then use different colors for x, y, z axes. 
    # if k > 1, visualize for multiple head pose trajectory, 
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig) 

    color_list = ['#27AE60', '#E74C3C', '#0000FF'] # green, red, blue 
    # Align the first frame to (0, 0, 0)
    vals = channels - channels[:, 0:1, 0:1, :] # K X T X 4 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
   
    connections = [[0, 1], [0, 2], [0, 3]]

    XYZ = np.array([0, 1, 2])

    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):   
            if num_cmp > 1:
                cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color_list[cmp_idx])[0])
            else: # num_cmp == 1 
                if XYZ[ind] == 0:
                    cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color_list[0])[0])
                elif XYZ[ind] == 1:
                    cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color_list[1])[0])
                elif XYZ[ind] == 2:
                    cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color_list[2])[0])

        lines.append(cur_line)

    # ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
  
    def animate(i):
        changed = []
        for ai in range(len(vals)):
            for ind, (p_idx, j_idx) in enumerate(connections):
                lines[ai][ind].set_data([vals[ai][i, j_idx, 0], vals[ai][i, p_idx, 0]], \
                    [vals[ai][i, j_idx, 1], vals[ai][i, p_idx, 1]])
                lines[ai][ind].set_3d_properties(
                    [vals[ai][i, j_idx, 2], vals[ai][i, p_idx, 2]])
            changed += lines

        return changed

    RADIUS = 2.5  # space around the subject
    xroot, yroot, zroot = vals[0, 0, 0, 0], vals[0, 0, 0, 1], vals[0, 0, 0, 2]
    # xroot, yroot, zroot = 0, 0, 0 # For debug
    ax.view_init(30, 45) # Used in training AMASS dataset
          
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  

    ani.save(dest_vis_path,                       
            writer="imagemagick",                                                
            fps=30) 

    ax.clear() 
    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()


def vis_multiple_2d_traj(pred_pos, gt_pos, dest_img_path, label_list=['pred', 'gt']):
    # pred_pos: T X 2, gt_pos: T X 2 
    x1 = pred_pos[:, 0]
    y1 = pred_pos[:, 1]
    x2 = gt_pos[:, 0]
    y2 = gt_pos[:, 1]

    # plt.plot(x1, y1, x2, y2)
    # plt.show()
    # plt.savefig(dest_img_path)

    # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax.plot(x1, y1, label=label_list[0])  # Plot some data on the axes.
    ax.plot(x2, y2, label=label_list[1])  # Plot more data on the axes...
    ax.set_xlabel('x')  # Add an x-label to the axes.
    ax.set_ylabel('y')  # Add a y-label to the axes.
    ax.set_title("2D Trajectory")  # Add a title to the axes.
    ax.legend()  #

    plt.draw()
    plt.savefig(dest_img_path)
    # plt.pause(0.01)
    plt.cla()

    plt.close()

def vis_single_frame_point_only(pose_data, dest_img_path):
    fig = plt.figure(figsize=(12, 7))

    # ax = fig.add_subplot('111', projection='3d', aspect=1)
    ax = Axes3D(fig)
   
    show3Dtraj_point_only(pose_data, ax, radius=5)

    plt.draw()
    # plt.xticks(fontsize=18)
    plt.savefig(dest_img_path)
    # plt.pause(0.01)
    plt.cla()

    plt.close()

def vis_multiple_frames_point_only(pose_data_list, dest_img_path, tag_list):
    # fig = plt.figure(figsize=(12, 7))
    fig = plt.figure(figsize=(24, 14))

    # ax = fig.add_subplot('111', projection='3d', aspect=1)
    ax = Axes3D(fig)

    color_list = ['#3A8F1D', '#FFA500', '#5725B3', '#B30000']
   
    # show3Dpose_multiple(pose_data_list, ax, color_list, radius=1.2)
    # show3Dtraj_multiple(pose_data_list, ax, color_list, radius=0.5)
    show3Dtraj_multiple_point_only(pose_data_list, ax, color_list, radius=1.0, tag_list=tag_list) # For visualizing SLAM results comparisons 
    # plt.legend( [ 'gt'   
    #         , 'lerp', 'our', 'slerp'       
    #         ], prop={'size': 14})

    plt.draw()
    plt.savefig(dest_img_path)
    # plt.pause(0.01)
    plt.cla()

    plt.close()


def show3Dtraj_multiple_point_only(channels_list, ax, color_list, radius=60, tag_list=['gt', 'our']):
    vals = channels_list[0] # T X 3 

    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    # xroot, yroot, zroot = -0.2, -0.2, 0
   
    # tag_list = ['gt', 'our']
    # tag_list = ['head', 'root'] # For comparing using SLAM results
    for idx in range(len(channels_list)):
        lcolor = color_list[idx]

        vals = channels_list[idx] # T X 3 
        timesteps = vals.shape[0] 
        for k_idx in range(timesteps):
            ax.scatter(channels_list[idx][k_idx, 0], channels_list[idx][k_idx, 1], channels_list[idx][k_idx, 2], c=lcolor, marker='o', s=10)  

        # ax.scatter(channels_list[idx][0, 0], channels_list[idx][0, 1], channels_list[idx][0, 2], c="red", marker='*', s=200)            
    # ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')
    
    ax.legend(prop={'size': 32})

    # ax.view_init(0, 180) # Used in training AMASS dataset
    ax.view_init(30, 45) 

    # ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    # ax.set_zlim3d([-RADIUS/1.0 + zroot, RADIUS/1.0 + zroot])
    # ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlim3d([-RADIUS/2+xroot, RADIUS/2 + xroot])
    ax.set_zlim3d([-RADIUS/2 + zroot, zroot])
    ax.set_ylim3d([-RADIUS/2+yroot, RADIUS/2 + yroot])

    # ax.set_xticks(fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.zaxis.set_tick_params(labelsize=18)

    # ax.set_axis_off()
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel("y", fontsize=24)
    ax.set_zlabel("z", fontsize=24)

def show3Dtraj_point_only(channels, ax, color=None, radius=80):
    if color is None:
        lcolor = '#A9A9A9'
    else:
        lcolor = color 

    vals = channels # T X 3
        
    ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], marker='o')

    ax.view_init(30, 45) 
    
    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS/2+xroot, RADIUS/2 + xroot])
    ax.set_zlim3d([-RADIUS/2 + zroot, zroot])
    ax.set_ylim3d([-RADIUS/2+yroot, RADIUS/2 + yroot])

    # ax.set_axis_off()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")