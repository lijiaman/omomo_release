import os 
import subprocess 
import trimesh 
import imageio 
import numpy as np 

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    # command = [
    #     'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', output_vid_file,
    # ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def images_to_video_w_imageio(img_folder, output_vid_file):
    img_files = os.listdir(img_folder)
    img_files.sort()
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=30, quality=8) 

def run_blender_rendering_and_save2video_test():
    obj_folder_path = "/viscam/u/jiamanli/datasets/BEHAVE/for_debug_tmp_use/Date07_Sub04_tablesquare_lift"
    scene_blend_path = "/viscam/u/jiamanli/github/hm_interaction/utils/blender_utils/floor_3_scene.blend"
    out_folder_path = "/viscam/u/jiamanli/datasets/BEHAVE/for_debug_tmp_use/Date07_Sub04_tablesquare_lift_rendered_imgs"
    
    subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
    -P /viscam/u/jiamanli/github/hm_interaction/Conditional-Motion-In-Betweening/cmib/vis/blender_vis_utils.py \
    -b -- --folder "+obj_folder_path+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True)    

    out_vid_path = "/viscam/u/jiamanli/datasets/BEHAVE/for_debug_tmp_use/Date07_Sub04_tablesquare_lift_vis.mp4"

    use_ffmpeg = False
    if use_ffmpeg:
        images_to_video(out_folder_path, out_vid_path)
    else:
        images_to_video_w_imageio(out_folder_path, out_vid_path)

def run_blender_rendering_and_save2video(obj_folder_path, out_folder_path, out_vid_path, \
    scene_blend_path="/viscam/u/jiamanli/github/hoi_syn/utils/blender_utils/floor_colorful_mat.blend", \
    vis_object=False, vis_human=True, vis_hand_and_object=False, vis_gt=False, \
    vis_handpose_and_object=False, hand_pose_path=None, mat_color="blue"):
    
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    if vis_object:
        if vis_human: # vis both human and object
            subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
            -P /viscam/u/jiamanli/github/hoi_syn/manip/vis/blender_vis_utils.py \
            -b -- --folder "+obj_folder_path+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True) 
        else: # vis object only 
            subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
            -P /viscam/u/jiamanli/github/hoi_syn/manip/vis/blender_vis_object_utils.py \
            -b -- --folder "+obj_folder_path+" --scene "+scene_blend_path+" --out-folder "+out_folder_path, shell=True)  
    elif vis_hand_and_object:
        if vis_gt:
            subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
            -P /viscam/u/jiamanli/github/hoi_syn/manip/vis/blender_vis_hand_and_object_utils.py \
            -b -- --folder "+obj_folder_path+" --scene "+\
            scene_blend_path+" --out-folder "+out_folder_path+" --visgt", shell=True)   
        else:
            subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
            -P /viscam/u/jiamanli/github/hoi_syn/manip/vis/blender_vis_hand_and_object_utils.py \
            -b -- --folder "+obj_folder_path+" --scene "+\
            scene_blend_path+" --out-folder "+out_folder_path, shell=True)    
    elif vis_handpose_and_object:
        subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
            -P /viscam/u/jiamanli/github/hoi_syn/manip/vis/blender_vis_handpose_and_object_utils.py \
            -b -- --folder "+obj_folder_path+" --scene "+\
            scene_blend_path+" --head-path "+hand_pose_path+" --out-folder "+out_folder_path, shell=True) 
    else: # Vis human only 
        subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
        -P /viscam/u/jiamanli/github/hoi_syn/manip/vis/blender_vis_human_utils.py \
        -b -- --folder "+obj_folder_path+" --scene "+\
        scene_blend_path+" --out-folder "+out_folder_path+" --material-color "+mat_color, shell=True)    

    use_ffmpeg = False
    if use_ffmpeg:
        images_to_video(out_folder_path, out_vid_path)
    else:
        images_to_video_w_imageio(out_folder_path, out_vid_path)

def run_blender_rendering_and_save2video_cmp(obj_folder_path, gt_obj_folder_path, out_folder_path, out_vid_path, \
    scene_blend_path="/viscam/u/jiamanli/github/egoego_private/utils/blender_utils/floor_colorful_mat.blend", \
    mat_color="blue"):
    
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
    -P /viscam/u/jiamanli/github/egoego_private/egoego/vis/blender_vis_cmp_human_utils.py \
    -b -- --folder "+obj_folder_path+" --gt-folder "+gt_obj_folder_path+" --scene "+\
    scene_blend_path+" --out-folder "+out_folder_path+" --material-color "+mat_color, shell=True)    

    images_to_video_w_imageio(out_folder_path, out_vid_path)

def run_blender_rendering_and_save2video_head_pose(npy_path, out_folder_path, out_vid_path, \
    scene_blend_path="/viscam/u/jiamanli/github/egoego_private/utils/blender_utils/floor_colorful_mat_w_head_pose.blend"):
    
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    subprocess.call("/viscam/u/jiamanli/blender-3.2.0-linux-x64/blender \
    -P /viscam/u/jiamanli/github/egoego_private/egoego/vis/blender_vis_headpose_utils.py \
    -b -- --folder "+out_folder_path+" --scene "+\
    scene_blend_path+" --out-folder "+out_folder_path+" --head-path "+npy_path, shell=True)    

    use_ffmpeg = False
    if use_ffmpeg:
        images_to_video(out_folder_path, out_vid_path)
    else:
        images_to_video_w_imageio(out_folder_path, out_vid_path)

def save_verts_faces_to_mesh_file(mesh_verts, mesh_faces, save_mesh_folder, save_gt=False):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        if save_gt:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_gt.obj")
        else:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".obj")
        mesh.export(curr_mesh_path)

def save_verts_faces_to_mesh_file_w_object(mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".ply")
        mesh.export(curr_mesh_path)

        obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx],
                        faces=obj_faces)
        curr_obj_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_object.ply")
        obj_mesh.export(curr_obj_mesh_path)

if __name__ == "__main__":
    run_blender_rendering_and_save2video_test()
   