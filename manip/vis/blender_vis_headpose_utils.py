import numpy as np
import json
import os
import math
import argparse

import bpy

if __name__ == "__main__":
    import sys
    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--")+1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment.')
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH',
                        help='path to output folder which include rendered img files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .blend path for 3D scene',
                        default='')
    parser.add_argument('--head-path', type=str, 
                        help='head pose numpy path',
                        default='')
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    # Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    scene_name = args.scene.split("/")[-1].replace("_scene.blend", "")
    print("scene name:{0}".format(scene_name))
   
    obj_folder = args.folder
    output_dir = args.out_folder
    print("obj_folder:{0}".format(obj_folder))
    print("output dir:{0}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load head pose numpy 
    head_pose_data = np.load(args.head_path) # T X 7 
    head_trans = head_pose_data[:, :3] # T X 3 
    head_rot_quat = head_pose_data[:, 3:] # T X 4 

    num_frames = head_pose_data.shape[0] 

    for frame_idx in range(num_frames):

        head_object = bpy.data.objects.get("coord.001")
        head_object.hide_render = False 

        # head_dup = head_object.copy()
        # bpy.data.scenes[0].objects.link(head_dup)
        
        head_object.rotation_quaternion = (head_rot_quat[frame_idx, 0], head_rot_quat[frame_idx, 1], \
        head_rot_quat[frame_idx, 2], head_rot_quat[frame_idx, 3]) # The default seems 90, 0, 0 while importing .obj into blender 
       
        head_object.location = (head_trans[frame_idx, 0], head_trans[frame_idx, 1], head_trans[frame_idx, 2])

        # bpy.context.object.hide_render = False

        # bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, file_name.replace(".ply", ".png"))
        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
        bpy.ops.render.render(write_still=True)

        # Delet materials
        # for block in bpy.data.materials:
        #     if block.users == 0:
        #         bpy.data.materials.remove(block)

        # bpy.data.objects.remove(head_dup, do_unlink=True)     

    bpy.ops.wm.quit_blender()
