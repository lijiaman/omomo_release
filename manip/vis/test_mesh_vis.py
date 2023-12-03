
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import os
import torch
import numpy as np
import math 
import time 

import scenepic as sp

import trimesh

from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb
from psbody.mesh.lines import Lines
 
def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

    points = points.reshape(-1,3)
    n_points = points.shape[0]

    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

        spheres.append(sphs)

    spheres = Mesh.concatenate_meshes(spheres)
    return spheres

def get_ground(grnd_size = 5):

    g_points = np.array([[-.2, 0.0, -.2],
                         [.2, 0.0, .2],
                         [.2, 0.0, -0.2],
                         [-.2, 0.0, .2]])
    g_points = g_points * 5
    g_points[:, 1] = g_points[:, 1] + 0.242
    # g_points[:, 1] = g_points[:, 1] + 0.5
  
    g_faces = np.array([[0, 1, 2], [0, 3, 1]])
    grnd_mesh = Mesh(v=grnd_size * g_points, f=g_faces, vc=name_to_rgb['DarkGrey']) # default gray 

    return grnd_mesh

class sp_animation():
    def __init__(self,
                 width = 1600,
                 height = 1600,
                 ):
        super(sp_animation, self).__init__()

        self.scene = sp.Scene()
        self.main = self.scene.create_canvas_3d(width=width, height=height)
        self.colors = sp.Colors

    def meshes_to_sp(self,meshes_list, layer_names):

        sp_meshes = []

        for i, m in enumerate(meshes_list):
            params = {'vertices' : m.v.astype(np.float32),
                      'normals' : m.estimate_vertex_normals().astype(np.float32),
                      'triangles' : m.f,
                      'colors' : m.vc.astype(np.float32)}
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            sp_m = self.scene.create_mesh(layer_id = layer_names[i])
            sp_m.add_mesh_with_normals(**params)
            if layer_names[i] == 'ground_mesh':
                sp_m.double_sided=True
            sp_meshes.append(sp_m)

        return sp_meshes

    def add_frame(self,meshes_list_ps, layer_names):

        meshes_list = self.meshes_to_sp(meshes_list_ps, layer_names)
        if not hasattr(self,'focus_point'):
            self.focus_point = meshes_list_ps[1].v.mean(0)
            center = self.focus_point
            center[1] += 0.4
            center[2] = 15
            rotation = sp.Transforms.rotation_about_z(math.radians(180))
            self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)
            # self.camera = sp.Camera(center=center, look_at=self.focus_point)

        main_frame = self.main.create_frame(focus_point=self.focus_point, camera=self.camera)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])

def test_vis():
    mesh_folder = "/move/u/jiamanli/datasets/BEHAVE/mesh_for_blender"
    seq_folders = os.listdir(mesh_folder)
    base_movie_path = "/viscam/u/jiamanli/test_goal_vis_in_behave"
    if not os.path.exists(base_movie_path):
        os.makedirs(base_movie_path)

    # Prepare ground plane mesh 
    grnd_mesh = get_ground()

    save_meshes = True
    sp_anim_motion = sp_animation()

    for seq_name in seq_folders:
        if "move" in seq_name:
            motion_meshes_path = os.path.join(base_movie_path, seq_name + '_motion_meshes')
            motion_path =  os.path.join(base_movie_path, seq_name + '_motion.html')
    
            seq_folder_path = os.path.join(mesh_folder, seq_name)
            ply_files = os.listdir(seq_folder_path)
            ply_files.sort() 

            i = 0
            start_time = time.time()
            for ply_name in ply_files:
                if "object" not in ply_name:
                    ply_path = os.path.join(seq_folder_path, ply_name)
                    obj_ply_path = ply_path.replace(".ply", "_object.ply")

                    # human_mesh = o3d.io.read_triangle_mesh(ply_path)
                    # obj_mesh =  o3d.io.read_triangle_mesh(obj_ply_path)
                    human_mesh = trimesh.load_mesh(ply_path)
                    obj_mesh = trimesh.load_mesh(obj_ply_path)
                    
                    sbj_i = Mesh(v=human_mesh.vertices, f=human_mesh.faces, vc=name_to_rgb['seashell1']) # default pink
                    obj_i = Mesh(v=obj_mesh.vertices, f=obj_mesh.faces, vc=name_to_rgb['plum']) # default yellow

                    # if save_meshes:
                    #     sbj_i.write_ply(motion_meshes_path + f'/{i:05d}_sbj.ply')
                    #     obj_i.write_ply(motion_meshes_path + f'/{i:05d}_obj.ply')

                    sp_anim_motion.add_frame([sbj_i, obj_i, grnd_mesh], ['sbj_mesh', 'obj_mesh', 'ground_mesh'])

                    i += 1 

                    if i >= 120:
                        break 
                
            if save_meshes:
                sp_anim_motion.save_animation(motion_path)

            print("For 120 frames, it takes:{0} seconds".format(time.time()-start_time))
            break 

if __name__ == "__main__":
    test_vis() 

'''
For 60 frames, it takes:35.824827671051025 seconds
'''