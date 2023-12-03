import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from matplotlib.animation import FuncAnimation

def project_root_position(position_arr: np.array, file_name: str):
    """
    Take batch of root arrays and porject it on 2D plane

    N: samples
    L: trajectory length
    J: joints

    position_arr: [N,L,J,3]
    """

    root_joints = position_arr[:, :, 0]

    x_pos = root_joints[:, :, 0]
    y_pos = root_joints[:, :, 2]

    fig = plt.figure()

    for i in range(x_pos.shape[1]):

        if i == 0:
            plt.scatter(x_pos[:, i], y_pos[:, i], c="b")
        elif i == x_pos.shape[1] - 1:
            plt.scatter(x_pos[:, i], y_pos[:, i], c="r")
        else:
            plt.scatter(x_pos[:, i], y_pos[:, i], c="k", marker="*", s=1)

    plt.title(f"Root Position: {file_name}")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.xlim((-300, 300))
    plt.ylim((-300, 300))
    plt.grid()
    plt.savefig(f"{file_name}.png", dpi=200)


def plot_single_pose(
    pose,
    frame_idx,
    skeleton,
    save_dir,
    prefix,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    parent_idx = skeleton.parents()

    for i, p in enumerate(parent_idx):
        if i > 0:
            ax.plot(
                [pose[i, 0], pose[p, 0]],
                [pose[i, 2], pose[p, 2]],
                [pose[i, 1], pose[p, 1]],
                c="k",
            )

    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")

    ax.set_ylim(z_min, z_max)
    ax.set_ylabel("$Y$ Axis")

    ax.set_zlim(y_min, y_max)
    ax.set_zlabel("$Z$ Axis")

    plt.draw()

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()


def plot_pose(
    start_pose,
    inbetween_pose,
    target_pose,
    frame_idx,
    skeleton,
    save_dir,
    prefix,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    parent_idx = skeleton.parents()

    for i, p in enumerate(parent_idx):
        if i > 0:
            ax.plot(
                [start_pose[i, 0], start_pose[p, 0]],
                [start_pose[i, 2], start_pose[p, 2]],
                [start_pose[i, 1], start_pose[p, 1]],
                c="b",
            )
            ax.plot(
                [inbetween_pose[i, 0], inbetween_pose[p, 0]],
                [inbetween_pose[i, 2], inbetween_pose[p, 2]],
                [inbetween_pose[i, 1], inbetween_pose[p, 1]],
                c="k",
            )
            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose[:, 0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose[:, 0].max(), target_pose[:, 0].max()]
    )

    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose[:, 1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose[:, 2].max(), target_pose[:, 2].max()]
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")

    ax.set_ylim(z_min, z_max)
    ax.set_ylabel("$Y$ Axis")

    ax.set_zlim(y_min, y_max)
    ax.set_zlabel("$Z$ Axis")

    plt.draw()

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()


def plot_pose_with_stop(
    start_pose,
    inbetween_pose,
    target_pose,
    stopover,
    frame_idx,
    skeleton,
    save_dir,
    prefix,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    parent_idx = skeleton.parents()

    for i, p in enumerate(parent_idx):
        if i > 0:
            ax.plot(
                [start_pose[i, 0], start_pose[p, 0]],
                [start_pose[i, 2], start_pose[p, 2]],
                [start_pose[i, 1], start_pose[p, 1]],
                c="b",
            )
            ax.plot(
                [inbetween_pose[i, 0], inbetween_pose[p, 0]],
                [inbetween_pose[i, 2], inbetween_pose[p, 2]],
                [inbetween_pose[i, 1], inbetween_pose[p, 1]],
                c="k",
            )
            ax.plot(
                [target_pose[i, 0], target_pose[p, 0]],
                [target_pose[i, 2], target_pose[p, 2]],
                [target_pose[i, 1], target_pose[p, 1]],
                c="r",
            )

            ax.plot(
                [stopover[i, 0], stopover[p, 0]],
                [stopover[i, 2], stopover[p, 2]],
                [stopover[i, 1], stopover[p, 1]],
                c="indigo",
            )

    x_min = np.min(
        [start_pose[:, 0].min(), inbetween_pose[:, 0].min(), target_pose[:, 0].min()]
    )
    x_max = np.max(
        [start_pose[:, 0].max(), inbetween_pose[:, 0].max(), target_pose[:, 0].max()]
    )

    y_min = np.min(
        [start_pose[:, 1].min(), inbetween_pose[:, 1].min(), target_pose[:, 1].min()]
    )
    y_max = np.max(
        [start_pose[:, 1].max(), inbetween_pose[:, 1].max(), target_pose[:, 1].max()]
    )

    z_min = np.min(
        [start_pose[:, 2].min(), inbetween_pose[:, 2].min(), target_pose[:, 2].min()]
    )
    z_max = np.max(
        [start_pose[:, 2].max(), inbetween_pose[:, 2].max(), target_pose[:, 2].max()]
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")

    ax.set_ylim(z_min, z_max)
    ax.set_ylabel("$Y$ Axis")

    ax.set_zlim(y_min, y_max)
    ax.set_zlabel("$Z$ Axis")

    plt.draw()

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()


def show3Dpose_animation(channels, parents, dest_vis_path, vis_behave=False):
    # channels: K X T X n_joints X 3
    # parents: a list containing the parent joint index 
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig) 

    color_list = ['#E74C3C', '#27AE60'] # Red, green  

    vals = channels # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
   
    # Generate connnections list based on parents list 
    connections = []
    num_joints = len(parents)
    for j_idx in range(num_joints):
        if j_idx > 0:
            connections.append([parents[j_idx], j_idx])

    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):
            cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color_list[cmp_idx])[0])
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

    RADIUS = 2  # space around the subject
    xroot, yroot, zroot = vals[-1, 0, 0, 0], vals[-1, 0, 0, 1], vals[-1, 0, 0, 2]
    # xroot, yroot, zroot = 0, 0, 0 # For debug

    if vis_behave:
        ax.view_init(-90, -90)
        # ax.view_init(-90, 90)
    else:
        ax.view_init(-90, 90) # Used in LAFAN data
    
    # ax.view_init(0, 0)
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  

    dest_vis_folder = "/".join(dest_vis_path.split("/")[:-1])
    if not os.path.exists(dest_vis_folder):
        os.makedirs(dest_vis_folder)
    ani.save(dest_vis_path,                       
            writer="imagemagick",                                                
            fps=30) 

    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()

def show3Dpose_animation_smpl22(channels, dest_vis_path, use_joints24=False):
    # channels: K X T X n_joints X 3
    # parents: a list containing the parent joint index 
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig) 

    color_list = ['#E74C3C', '#27AE60'] # Red, green  

    vals = channels # K X T X 24 X 3, K represents how many skeleton showing in same figure(K=2: show gt and generation)
    num_cmp = vals.shape[0]
   
    # SMPL connections 22 joints 
    if use_joints24:
        connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
    else:
        connections = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]]

    lines = []
    for cmp_idx in range(num_cmp):
        cur_line = []
        for ind, (i,j) in enumerate(connections):
            cur_line.append(ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color_list[cmp_idx])[0])
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

    RADIUS = 2  # space around the subject
    xroot, yroot, zroot = vals[-1, 0, 0, 0], vals[-1, 0, 0, 1], vals[-1, 0, 0, 2]
    # xroot, yroot, zroot = 0, 0, 0 # For debug
   
    ax.view_init(0, 120) 
    
    # ax.view_init(90, 0)
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    # ax.set_axis_off()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    ani = FuncAnimation(fig,                                            
                        animate,                                        
                        np.arange(len(vals[0])),                  
                        interval=33.33)  

    dest_vis_folder = "/".join(dest_vis_path.split("/")[:-1])
    if not os.path.exists(dest_vis_folder):
        os.makedirs(dest_vis_folder)
    ani.save(dest_vis_path,                       
            writer="imagemagick",                                                
            fps=30) 

    # plt.draw()
    # plt.savefig(dest_img_path)
    plt.cla()
    plt.close()
