import sys
sys.path.append("../../")

import os
import numpy as np
import joblib 
import json 
import trimesh 
import time 

import torch
from torch.utils.data import Dataset

import pytorch3d.transforms as transforms 

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder

from human_body_prior.body_model.body_model import BodyModel

from manip.lafan1.utils import rotate_at_frame_w_obj 

SMPLH_PATH = "./data/smpl_all_models/smplh_amass"

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def rotate(points, R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def get_smpl_parents(use_joints24=False):
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents() 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents() 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos, use_joints24=False):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        if i < lrot.shape[-2]:
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

def merge_two_parts(verts_list, faces_list):
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3 
        part_verts = verts_list[p_idx] # T X Nv X 3 
        part_faces = torch.from_numpy(faces_list[p_idx]) # T X Nf X 3 

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces+verts_num)

        verts_num += part_verts.shape[1] 

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).data.cpu().numpy() 

    return merged_verts, merged_faces 

    
class HandFootManipDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_object_splits=False,
    ):
        self.train = train
        
        self.window = window

        self.use_joints24 = True 

        self.use_object_splits = use_object_splits 
        self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", \
                    "trashcan", "monitor", "floorlamp", "clothesstand", "vacuum"] # 10 objects 
        self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod", "mop"]

        self.parents = get_smpl_parents() # 22 

        self.data_root_folder = data_root_folder 

        self.obj_geo_root_folder = os.path.join(data_root_folder, "captured_objects")

        self.bps_path = "./bps.pt"

        train_subjects = []
        test_subjects = []
        num_subjects = 17 
        for s_idx in range(1, num_subjects+1):
            if s_idx >= 16:
                test_subjects.append("sub"+str(s_idx))
            else:
                train_subjects.append("sub"+str(s_idx))

        dest_obj_bps_npy_folder = os.path.join(data_root_folder, "object_bps_npy_files_joints24")
        dest_obj_bps_npy_folder_for_test = os.path.join(data_root_folder, "object_bps_npy_files_for_eval_joints24")
      
        if not os.path.exists(dest_obj_bps_npy_folder):
            os.makedirs(dest_obj_bps_npy_folder)

        if not os.path.exists(dest_obj_bps_npy_folder_for_test):
            os.makedirs(dest_obj_bps_npy_folder_for_test)

        if self.train:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder 
        else:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test 

        if self.train:   
            seq_data_path = os.path.join(data_root_folder, "train_diffusion_manip_seq_joints24.p")  
            processed_data_path = os.path.join(data_root_folder, "train_diffusion_manip_window_"+str(self.window)+"_cano_joints24.p")
        else: 
            seq_data_path = os.path.join(data_root_folder, "test_diffusion_manip_seq_joints24.p")
            processed_data_path = os.path.join(data_root_folder, "test_diffusion_manip_window_"+str(self.window)+"_processed_joints24.p")
           
        min_max_mean_std_data_path = os.path.join(data_root_folder, "min_max_mean_std_data_window_"+str(self.window)+"_cano_joints24.p")
        
        self.prep_bps_data()

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)

            # if not self.train:
                # Mannually enable this.
                # self.get_bps_from_window_data_dict()
        else:
            if os.path.exists(seq_data_path):
                self.data_dict = joblib.load(seq_data_path)
            
            self.cal_normalize_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)            

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)
           
        self.global_jpos_min = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_min']).float().reshape(24, 3)[None]
        self.global_jpos_max = torch.from_numpy(min_max_mean_std_jpos_data['global_jpos_max']).float().reshape(24, 3)[None]
       
        if self.use_object_splits:
            self.window_data_dict = self.filter_out_object_split()

        # Get train and validation statistics. 
        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict)))
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict)))

        # Prepare SMPLX model 
        soma_work_base_dir = os.path.join(self.data_root_folder, 'smpl_all_models')
        support_base_dir = soma_work_base_dir 
        surface_model_type = "smplx"
        # surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
        # surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
        surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
        surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
        dmpl_fname = None
        num_dmpls = None 
        num_expressions = None
        num_betas = 16 

        self.male_bm = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)
        self.female_bm = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()
        
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

    def filter_out_object_split(self):
        # Remove some sequences from window_data_dict such that we have some unseen objects during testing. 
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data['seq_name']
            object_name = seq_name.split("_")[1]
            if self.train and object_name in self.train_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

            if (not self.train) and object_name in self.test_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_window_data_dict[new_cnt]['ori_w_idx'] = k 
                new_cnt += 1

        return new_window_data_dict

    def apply_transformation_to_obj_geometry(self, obj_mesh_path, obj_scale, obj_rot, obj_trans):
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3 

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3 
    
        seq_scale = torch.from_numpy(obj_scale).float() # T 
        seq_rot_mat = torch.from_numpy(obj_rot).float() # T X 3 X 3 
        if obj_trans.shape[-1] != 1:
            seq_trans = torch.from_numpy(obj_trans).float()[:, :, None] # T X 3 X 1 
        else:
            seq_trans = torch.from_numpy(obj_trans).float() # T X 3 X 1 
        transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
        seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return transformed_obj_verts, obj_mesh_faces  

    def load_object_geometry(self, object_name, obj_scale, obj_trans, obj_rot, \
        obj_bottom_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):
        obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name+"_cleaned_simplified.obj")
        if object_name == "vacuum" or object_name == "mop":
            two_parts = True 
        else:
            two_parts = False 

        if two_parts:
            top_obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name+"_cleaned_simplified_top.obj")
            bottom_obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name+"_cleaned_simplified_bottom.obj")

            top_obj_mesh_verts, top_obj_mesh_faces = self.apply_transformation_to_obj_geometry(top_obj_mesh_path, \
            obj_scale, obj_rot, obj_trans)
            bottom_obj_mesh_verts, bottom_obj_mesh_faces = self.apply_transformation_to_obj_geometry(bottom_obj_mesh_path, \
            obj_bottom_scale, obj_bottom_rot, obj_bottom_trans)

            obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts], \
            [top_obj_mesh_faces, bottom_obj_mesh_faces])
        else:
            obj_mesh_verts, obj_mesh_faces =self.apply_transformation_to_obj_geometry(obj_mesh_path, \
            obj_scale, obj_rot, obj_trans) # T X Nv X 3 

        return obj_mesh_verts, obj_mesh_faces 

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(x=obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=self.obj_bps.repeat(obj_trans.shape[0], \
                    1, 1)+obj_trans[:, None, :])['deltas'] # T X N X 3 

        return bps_object_geo

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0 
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(1, -1, 3)
            
            bps = {
                'obj': bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)
        
        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj']

    def get_bps_from_window_data_dict(self):
        # Given window_data_dict which contains canonizalized information, compute its corresponding BPS representation. 
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]

            seq_name = window_data['seq_name']
            object_name = seq_name.split("_")[1]

            curr_obj_scale = window_data['obj_scale']
            new_obj_x = window_data['obj_trans']
            new_obj_rot_mat = window_data['obj_rot_mat']

            # Get object geometry 
            if object_name in ["mop", "vacuum"]:
                curr_obj_bottom_scale = window_data['obj_bottom_scale']
                new_obj_bottom_x = window_data['obj_bottom_trans']
                new_obj_bottom_rot_mat = window_data['obj_bottom_rot_mat'] 

                obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale, \
                        new_obj_x, new_obj_rot_mat, \
                        curr_obj_bottom_scale, new_obj_bottom_x, \
                        new_obj_bottom_rot_mat) # T X Nv X 3, tensor

            else:
                obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale, \
                            new_obj_x, new_obj_rot_mat) # T X Nv X 3, tensor

            center_verts = obj_verts.mean(dim=1) # T X 3 
            dest_obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(k)+".npy")

            if not os.path.exists(dest_obj_bps_npy_path):
                object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
                np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy()) 

        import pdb 
        pdb.set_trace() 

    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0 
        for index in self.data_dict:
            seq_name = self.data_dict[index]['seq_name']

            object_name = seq_name.split("_")[1]

            betas = self.data_dict[index]['betas'] # 1 X 16 
            gender = self.data_dict[index]['gender']

            seq_root_trans = self.data_dict[index]['trans'] # T X 3 
            seq_root_orient = self.data_dict[index]['root_orient'] # T X 3 
            seq_pose_body = self.data_dict[index]['pose_body'].reshape(-1, 21, 3) # T X 21 X 3

            rest_human_offsets = self.data_dict[index]['rest_offsets'] # 22 X 3/24 X 3
            trans2joint = self.data_dict[index]['trans2joint'] # 3 

            obj_trans = self.data_dict[index]['obj_trans'][:, :, 0] # T X 3
            obj_rot = self.data_dict[index]['obj_rot'] # T X 3 X 3 

            obj_scale = self.data_dict[index]['obj_scale'] # T  

            obj_com_pos = self.data_dict[index]['obj_com_pos'] # T X 3 

            object_name = seq_name.split("_")[1]
            if object_name in ["mop", "vacuum"]:
                obj_bottom_trans = self.data_dict[index]['obj_bottom_trans'][:, :, 0] # T X 3 
                obj_bottom_rot = self.data_dict[index]['obj_bottom_rot'] # T X 3 X 3  

                obj_bottom_scale = self.data_dict[index]['obj_bottom_scale'] # T 
           
            num_steps = seq_root_trans.shape[0]
            for start_t_idx in range(0, num_steps, self.window//2):
                end_t_idx = start_t_idx + self.window - 1
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps 

                # Skip the segment that has a length < 30 
                if end_t_idx - start_t_idx < 30:
                    continue 

                self.window_data_dict[s_idx] = {}
                
                # Canonicalize the first frame's orientation. 
                joint_aa_rep = torch.cat((torch.from_numpy(seq_root_orient[start_t_idx:end_t_idx+1]).float()[:, None, :], \
                    torch.from_numpy(seq_pose_body[start_t_idx:end_t_idx+1]).float()), dim=1) # T X J X 3 
                X = torch.from_numpy(rest_human_offsets).float()[None].repeat(joint_aa_rep.shape[0], 1, 1).detach().cpu().numpy() # T X J X 3 
                X[:, 0, :] = seq_root_trans[start_t_idx:end_t_idx+1] 
                local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep) # T X J X 3 X 3 
                Q = transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy() # T X J X 4 

                obj_x = obj_trans[start_t_idx:end_t_idx+1].copy() # T X 3 
                obj_rot_mat = torch.from_numpy(obj_rot[start_t_idx:end_t_idx+1]).float()# T X 3 X 3 
                obj_q = transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy() # T X 4 

                curr_obj_scale = torch.from_numpy(obj_scale[start_t_idx:end_t_idx+1]).float() # T

                if object_name in ["mop", "vacuum"]:
                    obj_bottom_x = obj_bottom_trans[start_t_idx:end_t_idx+1].copy() # T X 3 
                    obj_bottom_rot_mat = torch.from_numpy(obj_bottom_rot[start_t_idx:end_t_idx+1]).float() # T X 3 X 3 
                    obj_bottom_q = transforms.matrix_to_quaternion(obj_bottom_rot_mat).detach().cpu().numpy() # T X 4 

                    curr_obj_bottom_scale = torch.from_numpy(obj_bottom_scale[start_t_idx:end_t_idx+1]).float() # T

                    _, _, new_obj_bottom_x, new_obj_bottom_q = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                    obj_bottom_x[np.newaxis], obj_bottom_q[np.newaxis], \
                    trans2joint[np.newaxis], self.parents, n_past=1, floor_z=True)
                    # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4 

                _, _, new_obj_x, new_obj_q = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                obj_x[np.newaxis], obj_q[np.newaxis], \
                trans2joint[np.newaxis], self.parents, n_past=1, floor_z=True)
                # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4 

                window_obj_com_pos = obj_com_pos[start_t_idx:end_t_idx+1].copy() # T X 3 
            
                X, Q, new_obj_com_pos, _ = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
                window_obj_com_pos[np.newaxis], obj_q[np.newaxis], \
                trans2joint[np.newaxis], self.parents, n_past=1, floor_z=True)
                # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4 

                new_seq_root_trans = X[0, :, 0, :] # T X 3 
                new_local_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(Q[0]).float()) # T X J X 3 X 3 
                new_local_aa_rep = transforms.matrix_to_axis_angle(new_local_rot_mat) # T X J X 3 
                new_seq_root_orient = new_local_aa_rep[:, 0, :] # T X 3
                new_seq_pose_body = new_local_aa_rep[:, 1:, :] # T X 21 X 3 
                
                new_obj_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_q[0]).float()) # T X 3 X 3 \
                
                cano_obj_mat = torch.matmul(new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1)) # 3 X 3 
                
                if object_name in ["mop", "vacuum"]:
                    new_obj_bottom_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_bottom_q[0]).float()) # T X 3 X 3 

                    obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale.detach().cpu().numpy(), \
                            new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy(), \
                            curr_obj_bottom_scale.detach().cpu().numpy(), new_obj_bottom_x[0], \
                            new_obj_bottom_rot_mat.detach().cpu().numpy()) # T X Nv X 3, tensor

                    center_verts = obj_verts.mean(dim=1) # T X 3 

                    query = self.process_window_data(rest_human_offsets, trans2joint, \
                        new_seq_root_trans, new_seq_root_orient.detach().cpu().numpy(), \
                        new_seq_pose_body.detach().cpu().numpy(), \
                        new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy(), \
                        curr_obj_scale.detach().cpu().numpy(), new_obj_com_pos[0], center_verts, \
                        new_obj_bottom_x[0], new_obj_bottom_rot_mat.detach().cpu().numpy(), \
                        curr_obj_bottom_scale.detach().cpu().numpy())
                else:
                    obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale.detach().cpu().numpy(), \
                            new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy()) # T X Nv X 3, tensor

                    center_verts = obj_verts.mean(dim=1) # T X 3 

                    query = self.process_window_data(rest_human_offsets, trans2joint, \
                        new_seq_root_trans, new_seq_root_orient.detach().cpu().numpy(), \
                        new_seq_pose_body.detach().cpu().numpy(),  \
                        new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy(), \
                        curr_obj_scale.detach().cpu().numpy(), new_obj_com_pos[0], center_verts)

                # Compute BPS representation for this window
                # Save to numpy file 
                dest_obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(s_idx)+".npy")

                if not os.path.exists(dest_obj_bps_npy_path):
                    object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
                    np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy()) 

                self.window_data_dict[s_idx]['cano_obj_mat'] = cano_obj_mat.detach().cpu().numpy() 

                curr_global_jpos = query['global_jpos'].detach().cpu().numpy()
                curr_global_jvel = query['global_jvel'].detach().cpu().numpy()
                curr_global_rot_6d = query['global_rot_6d'].detach().cpu().numpy()
              
                self.window_data_dict[s_idx]['motion'] = np.concatenate((curr_global_jpos.reshape(-1, 24*3), \
                curr_global_jvel.reshape(-1, 24*3), curr_global_rot_6d.reshape(-1, 22*6)), axis=1) # T X (24*3+24*3+22*6)

                self.window_data_dict[s_idx]['seq_name'] = seq_name
                self.window_data_dict[s_idx]['start_t_idx'] = start_t_idx
                self.window_data_dict[s_idx]['end_t_idx'] = end_t_idx 

                self.window_data_dict[s_idx]['betas'] = betas 
                self.window_data_dict[s_idx]['gender'] = gender

                self.window_data_dict[s_idx]['trans2joint'] = trans2joint 

                self.window_data_dict[s_idx]['obj_trans'] = query['obj_trans'].detach().cpu().numpy()
                self.window_data_dict[s_idx]['obj_rot_mat'] = query['obj_rot_mat'].detach().cpu().numpy()
                self.window_data_dict[s_idx]['obj_scale'] = query['obj_scale'].detach().cpu().numpy()

                self.window_data_dict[s_idx]['obj_com_pos'] = query['obj_com_pos'].detach().cpu().numpy()  
                self.window_data_dict[s_idx]['window_obj_com_pos'] = query['window_obj_com_pos'].detach().cpu().numpy() 

                if object_name in ["mop", "vacuum"]:
                    self.window_data_dict[s_idx]['obj_bottom_trans'] = query['obj_bottom_trans'].detach().cpu().numpy()
                    self.window_data_dict[s_idx]['obj_bottom_rot_mat'] = query['obj_bottom_rot_mat'].detach().cpu().numpy()
                    self.window_data_dict[s_idx]['obj_bottom_scale'] = query['obj_bottom_scale'].detach().cpu().numpy()

                s_idx += 1 

            # break 
       
    def extract_min_max_mean_std_from_data(self):
        all_global_jpos_data = []
        all_global_jvel_data = []

        for s_idx in self.window_data_dict:
            curr_window_data = self.window_data_dict[s_idx]['motion'] # T X D 

            all_global_jpos_data.append(curr_window_data[:, :24*3])
            all_global_jvel_data.append(curr_window_data[:, 24*3:2*24*3])

            start_t_idx = self.window_data_dict[s_idx]['start_t_idx'] 
            end_t_idx = self.window_data_dict[s_idx]['end_t_idx']
            curr_seq_name = self.window_data_dict[s_idx]['seq_name']

        all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(-1, 72) # (N*T) X 72 
        all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(-1, 72)

        min_jpos = all_global_jpos_data.min(axis=0)
        max_jpos = all_global_jpos_data.max(axis=0)
        min_jvel = all_global_jvel_data.min(axis=0)
        max_jvel = all_global_jvel_data.max(axis=0)

        stats_dict = {}
        stats_dict['global_jpos_min'] = min_jpos 
        stats_dict['global_jpos_max'] = max_jpos 
        stats_dict['global_jvel_min'] = min_jvel 
        stats_dict['global_jvel_max'] = max_jvel  

        return stats_dict 

    def normalize_jpos_min_max(self, ori_jpos):
        # ori_jpos: T X 22/24 X 3 
        normalized_jpos = (ori_jpos - self.global_jpos_min.to(ori_jpos.device))/(self.global_jpos_max.to(ori_jpos.device)\
        -self.global_jpos_min.to(ori_jpos.device))
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        return normalized_jpos # T X 22/24 X 3 

    def de_normalize_jpos_min_max(self, normalized_jpos):
        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range
        de_jpos = normalized_jpos * (self.global_jpos_max.to(normalized_jpos.device)-\
        self.global_jpos_min.to(normalized_jpos.device)) + self.global_jpos_min.to(normalized_jpos.device)

        return de_jpos # T X 22/24 X 3

    def normalize_jpos_min_max_hand_foot(self, ori_jpos, hand_only=True):
        # ori_jpos: BS X T X 2 X 3 
        lhand_idx = 22 
        rhand_idx = 23

        lfoot_idx = 10
        rfoot_idx = 11 

        bs = ori_jpos.shape[0] 
        num_steps = ori_jpos.shape[1] 
        ori_jpos = ori_jpos.reshape(bs, num_steps, -1) # BS X T X (2*3)

        if hand_only:
            hand_foot_jpos_max = torch.cat((self.global_jpos_max[0, lhand_idx], \
                    self.global_jpos_max[0, rhand_idx]), dim=0) # (3*4)

            hand_foot_jpos_min = torch.cat((self.global_jpos_min[0, lhand_idx], \
                    self.global_jpos_min[0, rhand_idx]), dim=0)
        else:
            hand_foot_jpos_max = torch.cat((self.global_jpos_max[0, lhand_idx], \
                    self.global_jpos_max[0, rhand_idx], \
                    self.global_jpos_max[0, lfoot_idx], \
                    self.global_jpos_max[0, rfoot_idx]), dim=0) # (3*4)

            hand_foot_jpos_min = torch.cat((self.global_jpos_min[0, lhand_idx], \
                    self.global_jpos_min[0, rhand_idx], \
                    self.global_jpos_min[0, lfoot_idx], \
                    self.global_jpos_min[0, rfoot_idx]), dim=0)

        hand_foot_jpos_max = hand_foot_jpos_max[None, None]
        hand_foot_jpos_min = hand_foot_jpos_min[None, None]
        normalized_jpos = (ori_jpos - hand_foot_jpos_min.to(ori_jpos.device))/(hand_foot_jpos_max.to(ori_jpos.device)\
        -hand_foot_jpos_min.to(ori_jpos.device))
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        normalized_jpos = normalized_jpos.reshape(bs, num_steps, -1, 3)

        return normalized_jpos # BS X T X 2 X 3 

    def de_normalize_jpos_min_max_hand_foot(self, normalized_jpos, hand_only=True):
        # normalized_jpos: BS X T X (3*4)
        lhand_idx = 22
        rhand_idx = 23 
       
        lfoot_idx = 10
        rfoot_idx = 11 

        bs, num_steps, _ = normalized_jpos.shape 

        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range

        if hand_only:
            hand_foot_jpos_max = torch.cat((self.global_jpos_max[0, lhand_idx], \
                    self.global_jpos_max[0, rhand_idx]), dim=0) # (3*4)

            hand_foot_jpos_min = torch.cat((self.global_jpos_min[0, lhand_idx], \
                    self.global_jpos_min[0, rhand_idx]), dim=0)
        else:
            hand_foot_jpos_max = torch.cat((self.global_jpos_max[0, lhand_idx], \
                    self.global_jpos_max[0, rhand_idx], \
                    self.global_jpos_max[0, lfoot_idx], \
                    self.global_jpos_max[0, rfoot_idx]), dim=0) # (3*4)

            hand_foot_jpos_min = torch.cat((self.global_jpos_min[0, lhand_idx], \
                    self.global_jpos_min[0, rhand_idx], \
                    self.global_jpos_min[0, lfoot_idx], \
                    self.global_jpos_min[0, rfoot_idx]), dim=0)

        hand_foot_jpos_max = hand_foot_jpos_max[None, None]
        hand_foot_jpos_min = hand_foot_jpos_min[None, None]

        de_jpos = normalized_jpos * (hand_foot_jpos_max.to(normalized_jpos.device)-\
        hand_foot_jpos_min.to(normalized_jpos.device)) + hand_foot_jpos_min.to(normalized_jpos.device)

        return de_jpos.reshape(bs, num_steps, -1, 3) # BS X T X 4(2) X 3 

    def process_window_data(self, rest_human_offsets, trans2joint, seq_root_trans, seq_root_orient, seq_pose_body, \
        obj_trans, obj_rot, obj_scale, obj_com_pos, center_verts, \
        obj_bottom_trans=None, obj_bottom_rot=None, obj_bottom_scale=None):
        random_t_idx = 0 
        end_t_idx = seq_root_trans.shape[0] - 1

        window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx+1]).cuda()
        window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx+1]).float().cuda()
        window_pose_body  = torch.from_numpy(seq_pose_body[random_t_idx:end_t_idx+1]).float().cuda()

        window_obj_scale = torch.from_numpy(obj_scale[random_t_idx:end_t_idx+1]).float().cuda() # T

        window_obj_rot_mat = torch.from_numpy(obj_rot[random_t_idx:end_t_idx+1]).float().cuda() # T X 3 X 3 
        window_obj_trans = torch.from_numpy(obj_trans[random_t_idx:end_t_idx+1]).float().cuda() # T X 3
        if obj_bottom_trans is not None:
            window_obj_bottom_scale = torch.from_numpy(obj_bottom_scale[random_t_idx:end_t_idx+1]).float().cuda() # T

            window_obj_bottom_rot_mat = torch.from_numpy(obj_bottom_rot[random_t_idx:end_t_idx+1]).float().cuda() # T X 3 X 3 
            window_obj_bottom_trans = torch.from_numpy(obj_bottom_trans[random_t_idx:end_t_idx+1]).float().cuda() # T X 3

        window_obj_com_pos = torch.from_numpy(obj_com_pos[random_t_idx:end_t_idx+1]).float().cuda() # T X 3
        window_center_verts = center_verts[random_t_idx:end_t_idx+1].to(window_obj_com_pos.device)

        move_to_zero_trans = window_root_trans[0:1, :].clone() # 1 X 3 
        move_to_zero_trans[:, 2] = 0 

        # Move motion and object translation to make the initial pose trans 0. 
        window_root_trans = window_root_trans - move_to_zero_trans 
        window_obj_trans = window_obj_trans - move_to_zero_trans 
        window_obj_com_pos = window_obj_com_pos - move_to_zero_trans 
        window_center_verts = window_center_verts - move_to_zero_trans 
        if obj_bottom_trans is not None:
            window_obj_bottom_trans = window_obj_bottom_trans - move_to_zero_trans 

        window_root_rot_mat = transforms.axis_angle_to_matrix(window_root_orient) # T' X 3 X 3 
        window_root_quat = transforms.matrix_to_quaternion(window_root_rot_mat)

        window_pose_rot_mat = transforms.axis_angle_to_matrix(window_pose_body) # T' X 21 X 3 X 3 

        # Generate global joint rotation 
        local_joint_rot_mat = torch.cat((window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 
        global_joint_rot_quat = transforms.matrix_to_quaternion(global_joint_rot_mat) # T' X 22 X 4 

        curr_seq_pose_aa = torch.cat((window_root_orient[:, None, :], window_pose_body), dim=1) # T' X 22 X 3/T' X 24 X 3 
        rest_human_offsets = torch.from_numpy(rest_human_offsets).float()[None] 
        curr_seq_local_jpos = rest_human_offsets.repeat(curr_seq_pose_aa.shape[0], 1, 1).cuda() # T' X 22 X 3/T' X 24 X 3  
        curr_seq_local_jpos[:, 0, :] = window_root_trans - torch.from_numpy(trans2joint).cuda()[None] # T' X 22/24 X 3 

        local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
        _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos, use_joints24=True)

        global_jpos = human_jnts # T' X 22/24 X 3 
        global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 22/24 X 3 

        global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 

        local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
        global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

        query = {}

        query['local_rot_mat'] = local_joint_rot_mat # T' X 22 X 3 X 3 
        query['local_rot_6d'] = local_rot_6d # T' X 22 X 6

        query['global_jpos'] = global_jpos # T' X 22/24 X 3 
        query['global_jvel'] = torch.cat((global_jvel, \
            torch.zeros(1, global_jvel.shape[1], 3).to(global_jvel.device)), dim=0) # T' X 22/24 X 3 
        
        query['global_rot_mat'] = global_joint_rot_mat # T' X 22 X 3 X 3 
        query['global_rot_6d'] = global_rot_6d # T' X 22 X 6

        query['obj_trans'] = window_obj_trans # T' X 3 
        query['obj_rot_mat'] = window_obj_rot_mat # T' X 3 X 3 

        query['obj_scale'] = window_obj_scale # T'

        query['obj_com_pos'] = window_obj_com_pos # T' X 3 

        query['window_obj_com_pos'] = window_center_verts # T X 3 

        if obj_bottom_trans is not None:
            query['obj_bottom_trans'] = window_obj_bottom_trans
            query['obj_bottom_rot_mat'] = window_obj_bottom_rot_mat 

            query['obj_bottom_scale'] = window_obj_bottom_scale # T'

        return query 

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, index):
        # index = 0 # For debug 
        data_input = self.window_data_dict[index]['motion']
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]['seq_name'] 
        object_name = seq_name.split("_")[1]

        start_t_idx = self.window_data_dict[index]['start_t_idx'] 
        end_t_idx = self.window_data_dict[index]['end_t_idx']
        
        trans2joint = self.window_data_dict[index]['trans2joint'] 
       
        if self.use_object_splits:
            ori_w_idx = self.window_data_dict[index]['ori_w_idx']
            obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(ori_w_idx)+".npy") 
        else:
            obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name+"_"+str(index)+".npy") 
        obj_bps_data = np.load(obj_bps_npy_path) # T X N X 3 
        obj_bps_data = torch.from_numpy(obj_bps_data) 

        # Load object point clouds 
        obj_scale = self.window_data_dict[index]['obj_scale']
        obj_trans = self.window_data_dict[index]['obj_trans']
        obj_rot_mat = self.window_data_dict[index]['obj_rot_mat']

        obj_com_pos = torch.from_numpy(self.window_data_dict[index]['window_obj_com_pos']).float()

        num_joints = 24 
               
        normalized_jpos = self.normalize_jpos_min_max(data_input[:, :num_joints*3].reshape(-1, num_joints, 3)) # T X 22 X 3 
       
        global_joint_rot = data_input[:, 2*num_joints*3:] # T X (22*6)

        new_data_input = torch.cat((normalized_jpos.reshape(-1, num_joints*3), global_joint_rot), dim=1)
        ori_data_input = torch.cat((data_input[:, :num_joints*3], global_joint_rot), dim=1)

        # Add padding. 
        actual_steps = new_data_input.shape[0]
        if actual_steps < self.window:
            paded_new_data_input = torch.cat((new_data_input, torch.zeros(self.window-actual_steps, new_data_input.shape[-1])), dim=0)
            paded_ori_data_input = torch.cat((ori_data_input, torch.zeros(self.window-actual_steps, ori_data_input.shape[-1])), dim=0)  

            paded_obj_bps = torch.cat((obj_bps_data.reshape(actual_steps, -1), \
                torch.zeros(self.window-actual_steps, obj_bps_data.reshape(actual_steps, -1).shape[1])), dim=0)
            paded_obj_com_pos = torch.cat((torch.from_numpy(self.window_data_dict[index]['window_obj_com_pos']).float(), \
                torch.zeros(self.window-actual_steps, 3)), dim=0)
           
            paded_obj_rot_mat = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_rot_mat']).float(), \
                torch.zeros(self.window-actual_steps, 3, 3)), dim=0)
            paded_obj_scale = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_scale']).float(), \
                torch.zeros(self.window-actual_steps,)), dim=0)
            paded_obj_trans = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_trans']).float(), \
                torch.zeros(self.window-actual_steps, 3)), dim=0)

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_bottom_rot_mat']).float(), \
                    torch.zeros(self.window-actual_steps, 3, 3)), dim=0)
                paded_obj_bottom_scale = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_bottom_scale']).float(), \
                    torch.zeros(self.window-actual_steps,)), dim=0)
                paded_obj_bottom_trans = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_bottom_trans']).float().squeeze(-1), \
                    torch.zeros(self.window-actual_steps, 3)), dim=0)
        else:
            paded_new_data_input = new_data_input 
            paded_ori_data_input = ori_data_input 

            paded_obj_bps = obj_bps_data.reshape(new_data_input.shape[0], -1)  
            paded_obj_com_pos = torch.from_numpy(self.window_data_dict[index]['window_obj_com_pos']).float()
        
            paded_obj_rot_mat = torch.from_numpy(self.window_data_dict[index]['obj_rot_mat']).float()
            paded_obj_scale = torch.from_numpy(self.window_data_dict[index]['obj_scale']).float()
            paded_obj_trans = torch.from_numpy(self.window_data_dict[index]['obj_trans']).float()

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.from_numpy(self.window_data_dict[index]['obj_bottom_rot_mat']).float()
                paded_obj_bottom_scale = torch.from_numpy(self.window_data_dict[index]['obj_bottom_scale']).float()
                paded_obj_bottom_trans = torch.from_numpy(self.window_data_dict[index]['obj_bottom_trans']).float().squeeze(-1)

        data_input_dict = {}
        data_input_dict['motion'] = paded_new_data_input
        data_input_dict['ori_motion'] = paded_ori_data_input 
    
        data_input_dict['obj_bps'] = paded_obj_bps
        data_input_dict['obj_com_pos'] = paded_obj_com_pos

        data_input_dict['obj_rot_mat'] = paded_obj_rot_mat
        data_input_dict['obj_scale'] = paded_obj_scale 
        data_input_dict['obj_trans'] = paded_obj_trans 

        if object_name in ["mop", "vacuum"]:
            data_input_dict['obj_bottom_rot_mat'] = paded_obj_bottom_rot_mat
            data_input_dict['obj_bottom_scale'] = paded_obj_bottom_scale 
            data_input_dict['obj_bottom_trans'] = paded_obj_bottom_trans 
        else:
            data_input_dict['obj_bottom_rot_mat'] = paded_obj_rot_mat
            data_input_dict['obj_bottom_scale'] = paded_obj_scale 
            data_input_dict['obj_bottom_trans'] = paded_obj_trans 

        data_input_dict['betas'] = self.window_data_dict[index]['betas']
        data_input_dict['gender'] = str(self.window_data_dict[index]['gender'])
       
        data_input_dict['seq_name'] = seq_name
        data_input_dict['obj_name'] = seq_name.split("_")[1]

        data_input_dict['seq_len'] = actual_steps 

        data_input_dict['trans2joint'] = trans2joint 

        return data_input_dict 
        # data_input_dict['motion']: T X (22*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3 
