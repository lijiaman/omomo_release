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
    parser.add_argument("--visgt", action="store_true")
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
    
    # Prepare ply paths 
    ori_obj_files = os.listdir(obj_folder)
    ori_obj_files.sort()
    obj_files = []
    for tmp_name in ori_obj_files:
        # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
        if ".ply" in tmp_name and "object" in tmp_name:
            obj_files.append(tmp_name)

    for frame_idx in range(len(obj_files)):
        file_name = obj_files[frame_idx]

        # Iterate folder to process all model
        path_to_file = os.path.join(obj_folder, file_name)
        object_path_to_file = path_to_file

        # Load object mesh and set material 
        if ".obj" in object_path_to_file:
            new_obj = bpy.ops.import_scene.obj(filepath=object_path_to_file, split_mode ="OFF")
        elif ".ply" in object_path_to_file:
            new_obj = bpy.ops.import_mesh.ply(filepath=object_path_to_file)
        # obj_object = bpy.context.selected_objects[0]
        obj_object = bpy.data.objects[file_name.replace(".ply", "")]
        # obj_object.scale = (0.3, 0.3, 0.3)
        mesh = obj_object.data
        for f in mesh.polygons:
            f.use_smooth = True
        
        obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
        # obj_object.location.y = 0

        mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        obj_object.data.materials.append(mat)
        mat.use_nodes = True
        principled_bsdf = mat.node_tree.nodes['Principled BSDF']
        if principled_bsdf is not None:
            # principled_bsdf.inputs[0].default_value = (220/255.0, 220/255.0, 220/255.0, 1) # Gray, close to white after rendering 
            # principled_bsdf.inputs[0].default_value = (10/255.0, 30/255.0, 225/255.0, 1) # Light Blue, used for floor scene 
            principled_bsdf.inputs[0].default_value = (153/255.0, 51/255.0, 255/255.0, 1) # Light Purple

        obj_object.active_material = mat

        has_foot = False 
        # Set up ball mesh 
        for b_idx in range(4):
            if args.visgt:
                path_to_file = os.path.join(obj_folder, file_name.replace("object.ply", "gt_"+str(b_idx)+".ply"))
            else:
                path_to_file = os.path.join(obj_folder, file_name.replace("object.ply", "pred_"+str(b_idx)+".ply"))

            if not os.path.exists(path_to_file):
                break 

            object_path_to_file = path_to_file

            # Load object mesh and set material 
            if ".obj" in object_path_to_file:
                new_obj = bpy.ops.import_scene.obj(filepath=object_path_to_file, split_mode ="OFF")
            elif ".ply" in object_path_to_file:
                new_obj = bpy.ops.import_mesh.ply(filepath=object_path_to_file)
            # obj_object = bpy.context.selected_objects[0]
            if b_idx == 0:
                if args.visgt:
                    ball_obj_lhand_object = bpy.data.objects[file_name.replace("object.ply", "gt_"+str(b_idx))]
                else:
                    ball_obj_lhand_object = bpy.data.objects[file_name.replace("object.ply", "pred_"+str(b_idx))]

                mesh = ball_obj_lhand_object.data
                for f in mesh.polygons:
                    f.use_smooth = True
                
                ball_obj_lhand_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
            
                ball_obj_lhand_object.active_material = bpy.data.materials.get("orange")

            elif b_idx == 1:
                if args.visgt:
                    ball_obj_rhand_object = bpy.data.objects[file_name.replace("object.ply", "gt_"+str(b_idx))]
                else:
                    ball_obj_rhand_object = bpy.data.objects[file_name.replace("object.ply", "pred_"+str(b_idx))]

                mesh = ball_obj_rhand_object.data
                for f in mesh.polygons:
                    f.use_smooth = True
                
                ball_obj_rhand_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
            
                ball_obj_rhand_object.active_material = bpy.data.materials.get("blue")
            elif b_idx == 2:
                if args.visgt:
                    ball_obj_lfoot_object = bpy.data.objects[file_name.replace("object.ply", "gt_"+str(b_idx))]
                else:
                    ball_obj_lfoot_object = bpy.data.objects[file_name.replace("object.ply", "pred_"+str(b_idx))]

                mesh = ball_obj_lfoot_object.data
                for f in mesh.polygons:
                    f.use_smooth = True
                
                ball_obj_lfoot_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
            
                ball_obj_lfoot_object.active_material = bpy.data.materials.get("purple")

                has_foot = True 
            elif b_idx == 3:
                if args.visgt:
                    ball_obj_rfoot_object = bpy.data.objects[file_name.replace("object.ply", "gt_"+str(b_idx))]
                else:
                    ball_obj_rfoot_object = bpy.data.objects[file_name.replace("object.ply", "pred_"+str(b_idx))]

                mesh = ball_obj_rfoot_object.data
                for f in mesh.polygons:
                    f.use_smooth = True
                
                ball_obj_rfoot_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
            
                ball_obj_rfoot_object.active_material = bpy.data.materials.get("green")
           
                has_foot = True 

        # bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, file_name.replace(".ply", ".png"))
        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
        bpy.ops.render.render(write_still=True)

        # Delet materials
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        bpy.data.objects.remove(obj_object, do_unlink=True)          
       
        bpy.data.objects.remove(ball_obj_lhand_object, do_unlink=True)  
        bpy.data.objects.remove(ball_obj_rhand_object, do_unlink=True) 

        if has_foot:
            bpy.data.objects.remove(ball_obj_lfoot_object, do_unlink=True)  
            bpy.data.objects.remove(ball_obj_rfoot_object, do_unlink=True)   


    bpy.ops.wm.quit_blender()
