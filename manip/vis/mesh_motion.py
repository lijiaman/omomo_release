import sys 
sys.path.append('.')
sys.path.append('..')
sys.path.append("/viscam/u/jiamanli/github/hoi_syn")

import torch 

import os 
import math 
import time 
import numpy as np

import scenepic as sp

import trimesh

# from psbody.mesh import Mesh
# from psbody.mesh.sphere import Sphere
# from psbody.mesh.colors import name_to_rgb
# from psbody.mesh.lines import Lines

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS, KEYPT_VERTS

NUM_BETAS = 10

def get_body_model_sequence(smplh_path, gender, num_frames,
                  pose_body, pose_hand, betas, root_orient, trans):
    gender = str(gender)
    # bm_path = os.path.join(smplh_path, gender + '/model.npz')
    bm_path = os.path.join(smplh_path, "SMPLH_"+gender+".pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_path=bm_path, num_betas=NUM_BETAS, batch_size=num_frames).to(device)

    # pose_body = pose_body.to(device)
    # pose_hand = pose_hand.to(device)
    betas = torch.Tensor(np.repeat(betas[:NUM_BETAS][np.newaxis], num_frames, axis=0)).to(device)
    # root_orient = root_orient.to(device)
    # trans = trans.to(device)

    body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient, trans=trans)
    
    return body

def get_posed_obj_mesh(rest_mesh_ply_path, obj_rot_mat, obj_trans):
    # obj_rot_mat: T X 3 X 3
    # obj_trans: T X 3 
    mesh = trimesh.load_mesh(rest_mesh_ply_path)
    verts = np.asarray(mesh.vertices) # Nv X 3 
    faces = np.asarray(mesh.faces) # Nf X 3 
   
    num_verts = verts.shape[0]
    num_steps = obj_rot_mat.shape[0] 

    verts = torch.from_numpy(verts).float().to(obj_rot_mat.device)[None].repeat(num_steps, 1, 1) # T X Nv X 3 

    # Center the object mesh 
    center = torch.mean(verts, 1)[:, None, :]
    verts -= center
    
    # Apply rotation and translation to the object mesh 
    posed_verts = torch.matmul(verts, \
        obj_rot_mat.transpose(1, 2)) + obj_trans[:, None, :] # T X Nv X 3 

    return posed_verts, faces 

def get_mesh_verts_faces(root_trans, aa_rot_rep, betas, gender):
    # root_trans: T X 3
    # aa_rot_rep: T X 22 X 3 
    num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(num_steps, 30, 3).to(aa_rot_rep.device) # T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=1) # T X 52 X 3 

    smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh" 

    # Convert SMPLH parameters to 3D human mesh 
    # gender = "male"
    # betas = torch.from_numpy(betas).float().to(aa_rot_rep.device) # 10 
    root_orient = aa_rot_rep[:, 0, :] # T X 3 
    pose_body = aa_rot_rep[:, 1:22, :].reshape(num_steps, -1) # T X 21 X 3 -> T X (21*3)
    pose_hand = aa_rot_rep[:, 22:, :].reshape(num_steps, -1) # T X 30 X 3 -> T X (30*3)
    body = get_body_model_sequence(smplh_path, gender, num_steps,
                        pose_body, pose_hand, betas, root_orient, root_trans)
    
    cur_joint_seq = body.Jtr.data.cpu().numpy()
    cur_body_joint_seq = cur_joint_seq[:, :len(SMPL_JOINTS), :]

    cur_vtx_seq = body.v.data.cpu().numpy() # T X Nv X 3
    cur_faces = body.f.data.cpu().numpy()  # Nf X 3

    return cur_vtx_seq, cur_faces 

def get_mesh_verts_faces_w_object(root_trans, aa_rot_rep, betas, gender, obj_rot_mat, obj_trans, seq_name):
    # root_trans: T X 3
    # aa_rot_rep: T X 22 X 3 
    # obj_rot_mat: T X 3 X 3
    # obj_trans: T X 3 
    num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(num_steps, 30, 3).to(aa_rot_rep.device) # T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=1) # T X 52 X 3 

    smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh" 

    # Convert SMPLH parameters to 3D human mesh 
    # gender = "male"
    # betas = torch.from_numpy(betas).float().to(aa_rot_rep.device) # 10 
    root_orient = aa_rot_rep[:, 0, :] # T X 3 
    pose_body = aa_rot_rep[:, 1:22, :].reshape(num_steps, -1) # T X 21 X 3 -> T X (21*3)
    pose_hand = aa_rot_rep[:, 22:, :].reshape(num_steps, -1) # T X 30 X 3 -> T X (30*3)
    body = get_body_model_sequence(smplh_path, gender, num_steps,
                        pose_body, pose_hand, betas, root_orient, root_trans)
    
    cur_joint_seq = body.Jtr
    cur_body_joint_seq = cur_joint_seq[:, :len(SMPL_JOINTS), :]

    # cur_vtx_seq = body.v.data.cpu() # T X Nv X 3
    cur_vtx_seq = body.v 
    cur_faces = body.f.data.cpu()  # Nf X 3

    # Generate posed object vertices and faces 
    object_name = seq_name.split("_")[2]
    rest_mesh_folder = "/move/u/jiamanli/datasets/BEHAVE/objects"
    rest_mesh_ply_path = os.path.join(rest_mesh_folder, object_name, object_name+".obj")

    posed_obj_verts, posed_obj_faces = get_posed_obj_mesh(rest_mesh_ply_path, obj_rot_mat, obj_trans)

    return cur_vtx_seq, cur_body_joint_seq, cur_faces, posed_obj_verts, posed_obj_faces 

def get_mesh_verts_faces_for_object_only(obj_rot_mat, obj_trans, seq_name):
    # obj_rot_mat: T X 3 X 3
    # obj_trans: T X 3 

    # Generate posed object vertices and faces 
    object_name = seq_name.split("_")[2]
    rest_mesh_folder = "/move/u/jiamanli/datasets/BEHAVE/objects"
    rest_mesh_ply_path = os.path.join(rest_mesh_folder, object_name, object_name+".obj")

    posed_obj_verts, posed_obj_faces = get_posed_obj_mesh(rest_mesh_ply_path, obj_rot_mat, obj_trans)

    return posed_obj_verts, posed_obj_faces   

def zero_pad_tensors(pad_list, pad_size):
    '''
    Assumes tensors in pad_list are B x D
    '''
    new_pad_list = []
    for pad_idx, pad_tensor in enumerate(pad_list):
        padding = torch.zeros((pad_size, pad_tensor.size(1))).to(pad_tensor)
        new_pad_list.append(torch.cat([pad_tensor, padding], dim=0))
    return new_pad_list

def get_mesh_verts_faces_for_human_only(root_trans, aa_rot_rep, betas, gender, bm_dict, smpl_batch_size):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 10 
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3 
    smpl_root_orient = aa_rot_rep[:, :, 0, :].reshape(-1, 3) # (BS*T) X 3 
    num_betas = betas.shape[1]
    smpl_betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(-1, num_betas) # (BS*T) X 10 
    smpl_pose_body = aa_rot_rep[:, :, 1:22, :].reshape(-1, 21, 3).reshape(-1, 21*3) # (BS*T) X (21*3)

    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        # need to pad extra frames with zeros in case not as long as expected 
        pad_size = smpl_batch_size - nbidx
        if nbidx == 0:
            # skip if no frames for this gender
            continue
        pad_list = gender_smpl_vals

        if pad_size < 0:
            raise Exception('SMPL model batch size not large enough to accomodate!')
        elif pad_size > 0:
            pad_list = zero_pad_tensors(pad_list, pad_size)
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose = pad_list
        bm = bm_dict[gender_name]

        # import pdb
        # pdb.set_trace() 
        dtype = cur_pred_pose.dtype
        pred_body = bm(pose_body=cur_pred_pose.float(), betas=cur_betas.float(), root_orient=cur_pred_orient.float(), trans=cur_pred_trans.float())
        if pad_size > 0:
            pred_joints.append(pred_body.Jtr[:-pad_size].to(dtype=dtype))
            pred_verts.append(pred_body.v[:-pad_size].to(dtype=dtype))
        else:
            pred_joints.append(pred_body.Jtr.to(dtype=dtype))
            pred_verts.append(pred_body.v.to(dtype=dtype))

    # cat all genders and reorder to original batch ordering
    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:,:len(SMPL_JOINTS),:]
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3)

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    return x_pred_smpl_joints, x_pred_smpl_verts, pred_body.f.to(dtype=dtype)

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

    # seems like in this visualization, world frame is different, y is coresponding to z of SMPL? 

    g_points = g_points * 5
    # g_points[:, 1] = g_points[:, 1] + 0.242
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
            self.focus_point = meshes_list_ps[0].v.mean(0)
            center = self.focus_point
            center[1] += 0.4
            center[2] = 15
            rotation = sp.Transforms.rotation_about_z(math.radians(180))
            self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)
            # self.camera = sp.Camera(center=center, look_at=self.focus_point)

        # main_frame = self.main.create_frame(focus_point=self.focus_point, camera=self.camera)
        main_frame = self.main.create_frame(focus_point=self.focus_point)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])
        
def vis_mesh_motion(root_trans, aa_rot_rep, betas, seq_name, motion_path, vis_object=False):
    # root_trans: T X 3
    # aa_rot_rep: T X 22 X 3 
    
    num_steps, num_joints, _ = aa_rot_rep.shape

    cur_vtx_seq, cur_faces = get_mesh_verts_faces(root_trans, aa_rot_rep, betas)

    sp_anim_motion = sp_animation()

    grnd_mesh = get_ground()

    for t_idx in range(num_steps):
        sbj_i = Mesh(v=cur_vtx_seq[t_idx], f=cur_faces, vc=name_to_rgb['seashell1']) # default pink
        # obj_i = Mesh(v=obj_mesh.vertices, f=obj_mesh.faces, vc=name_to_rgb['plum']) # default yellow

        # if save_meshes:
        #     sbj_i.write_ply(motion_meshes_path + f'/{i:05d}_sbj.ply')
        #     obj_i.write_ply(motion_meshes_path + f'/{i:05d}_obj.ply')

        sp_anim_motion.add_frame([sbj_i, grnd_mesh], ['sbj_mesh','ground_mesh'])
        # sp_anim_motion.add_frame([sbj_i, obj_i, grnd_mesh], ['sbj_mesh', 'obj_mesh', 'ground_mesh'])
      
    sp_anim_motion.save_animation(motion_path)

if __name__ == "__main__":
    root_trans = torch.zeros(60, 3)
    aa_rot_rep = torch.zeros(60, 22, 3)
    betas = np.zeros((10))
    seq_name = "debug_vis"
    motion_path = "./debug_vis_code.html"
    vis_mesh_motion(root_trans, aa_rot_rep, betas, seq_name, motion_path, vis_object=False)
   