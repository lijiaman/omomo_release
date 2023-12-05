import argparse
import os
import numpy as np
import yaml
import random
import json 

import trimesh 

from tqdm import tqdm
from pathlib import Path

import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
from multiprocessing import cpu_count

from human_body_prior.body_model.body_model import BodyModel

from manip.data.hand_foot_dataset import HandFootManipDataset, quat_ik_torch, quat_fk_torch

from manip.model.transformer_hand_foot_manip_cond_diffusion_model import CondGaussianDiffusion 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

from manip.model.transformer_fullbody_cond_diffusion_model import CondGaussianDiffusion as FullBodyCondGaussianDiffusion
from trainer_full_body_manip_diffusion import Trainer as FullBodyTrainer 

from evaluation_metrics import compute_metrics, compute_s1_metrics, compute_collision

from matplotlib import pyplot as plt

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=40000,
        results_folder='./results',
        use_wandb=True,  
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        self.window = opt.window

        self.use_object_split = self.opt.use_object_split 

        self.data_root_folder = self.opt.data_root_folder 

        self.prep_dataloader(window_size=opt.window)

        self.bm_dict = self.ds.bm_dict 

        self.test_on_train = self.opt.test_sample_res_on_train 

        self.add_hand_processing = self.opt.add_hand_processing 

        self.for_quant_eval = self.opt.for_quant_eval 

        self.use_gt_hand_for_eval = self.opt.use_gt_hand_for_eval 

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = HandFootManipDataset(train=True, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split)
        val_dataset = HandFootManipDataset(train=False, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=4))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=4))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def prep_temporal_condition_mask(self, data, t_idx=0):
        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 

    def train(self):
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                data = data_dict['motion'].cuda() # BS X T X (22*3+22*6)

                bs, num_steps, _ = data.shape 

                data = self.extract_palm_jpos_only_data(data)
                # BS X T X (2*3) 

                obj_bps_data = data_dict['obj_bps'].cuda() 
                obj_com_pos = data_dict['obj_com_pos'].cuda() # BS X T X 3 

                ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1) # BS X T X (3+1024*3)

                cond_mask = None 

                # Generate padding mask 
                actual_seq_len = data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(data.device)

                with autocast(enabled = self.amp):    
                    loss_diffusion = self.model(data, ori_data_cond, cond_mask, padding_mask)
                    
                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                        }
                        wandb.log(log_dict)

                    if idx % 10 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_data = val_data_dict['motion'].cuda()

                    bs, num_steps, _ = val_data.shape 

                    val_data = self.extract_palm_jpos_only_data(val_data)
                    # BS X T X (2*3) 

                    obj_bps_data = val_data_dict['obj_bps'].cuda()
                    obj_com_pos = val_data_dict['obj_com_pos'].cuda() 

                    ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1) # BS X T X (3+1024*3)

                    cond_mask = None 

                    # Generate padding mask 
                    actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                    tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                    self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_data.device)

                    # Get validation loss 
                    val_loss_diffusion = self.model(val_data, ori_data_cond, cond_mask, padding_mask)
                    val_loss = val_loss_diffusion 
                    if self.use_wandb:
                        val_log_dict = {
                            "Validation/Loss/Total Loss": val_loss.item(),
                            "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                        }
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every
            
                    bs_for_vis = 1

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

                        all_res_list = self.ema.ema_model.sample(val_data, ori_data_cond, cond_mask, padding_mask)
                        all_res_list = all_res_list[:bs_for_vis]

                        self.gen_vis_res(all_res_list, val_data_dict, self.step, vis_tag="pred_jpos")

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()

        num_sample = 50
        
        with torch.no_grad():
            for s_idx in range(num_sample):
                if self.test_on_train:
                    val_data_dict = next(self.dl)
                else:
                    val_data_dict = next(self.val_dl)
                val_data = val_data_dict['motion'].cuda()

                val_data = self.extract_palm_jpos_only_data(val_data)
                # BS X T X (2*3) 
     
                obj_bps_data = val_data_dict['obj_bps'].cuda()
                obj_com_pos = val_data_dict['obj_com_pos'].cuda() 

                ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1) # BS X T X (3+1024*3)

                cond_mask = None 

                # Generate padding mask 
                actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                max_num = 1

                all_res_list = self.ema.ema_model.sample(val_data, ori_data_cond, \
                cond_mask=cond_mask, padding_mask=padding_mask)

                vis_tag = str(milestone)+"_stage1_sample_"+str(s_idx)

                if self.test_on_train:
                    vis_tag = vis_tag + "_on_train"
                
                self.gen_vis_res(all_res_list[:max_num], val_data_dict, milestone, vis_tag=vis_tag)

    def extract_palm_jpos_only_data(self, data_input):
        # data_input: BS X T X D (22*3+22*6)
        lpalm_idx = 22
        rpalm_idx = 23 
        data_input = torch.cat((data_input[:, :, lpalm_idx*3:lpalm_idx*3+3], \
                data_input[:, :, rpalm_idx*3:rpalm_idx*3+3]), dim=-1)
        # BS X T X (2*3)

        return data_input 

    def create_ball_mesh(self, center_pos, ball_mesh_path):
        # center_pos: 4(2) X 3  
        lhand_color = np.asarray([255, 87, 51])  # red 
        rhand_color = np.asarray([17, 99, 226]) # blue
        lfoot_color = np.asarray([134, 17, 226]) # purple 
        rfoot_color = np.asarray([22, 173, 100]) # green 

        color_list = [lhand_color, rhand_color, lfoot_color, rfoot_color]

        num_mesh = center_pos.shape[0]
        for idx in range(num_mesh):
            ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=center_pos[idx])
            
            dest_ball_mesh = trimesh.Trimesh(
                vertices=ball_mesh.vertices,
                faces=ball_mesh.faces,
                vertex_colors=color_list[idx],
                process=False)

            result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding='ascii')
            output_file = open(ball_mesh_path.replace(".ply", "_"+str(idx)+".ply"), "wb+")
            output_file.write(result)
            output_file.close()

    def export_to_mesh(self, mesh_verts, mesh_faces, mesh_path):
        dest_mesh = trimesh.Trimesh(
            vertices=mesh_verts,
            faces=mesh_faces,
            process=False)

        result = trimesh.exchange.ply.export_ply(dest_mesh, encoding='ascii')
        output_file = open(mesh_path, "wb+")
        output_file.write(result)
        output_file.close()

    def process_hand_foot_contact_jpos(self, hand_foot_jpos, object_mesh_verts, object_mesh_faces, obj_rot):
        # hand_foot_jpos: T X 2 X 3 
        # object_mesh_verts: T X Nv X 3 
        # object_mesh_faces: Nf X 3 
        # obj_rot: T X 3 X 3 
        all_contact_labels = []
        all_object_c_idx_list = []
        all_dist = []

        obj_rot = torch.from_numpy(obj_rot).to(hand_foot_jpos.device)
        object_mesh_verts = object_mesh_verts.to(hand_foot_jpos.device)

        num_joints = hand_foot_jpos.shape[1]
        num_steps = hand_foot_jpos.shape[0]

        threshold = 0.03 # Use palm position, should be smaller. 
       
        joint2object_dist = torch.cdist(hand_foot_jpos, object_mesh_verts.to(hand_foot_jpos.device)) # T X 2 X Nv 
     
        all_dist, all_object_c_idx_list = joint2object_dist.min(dim=2) # T X 2
        all_contact_labels = all_dist < threshold # T X 2

        new_hand_foot_jpos = hand_foot_jpos.clone() # T X 2 X 3 

        # For each joint, scan the sequence, if contact is true, then use the corresponding object idx for the 
        # rest of subsequence in contact. 
        for j_idx in range(num_joints):
            continue_prev_contact = False 
            for t_idx in range(num_steps):
                if continue_prev_contact:
                    relative_rot_mat = torch.matmul(obj_rot[t_idx], reference_obj_rot.inverse())
                    curr_contact_normal = torch.matmul(relative_rot_mat, contact_normal[:, None]).squeeze(-1)

                    new_hand_foot_jpos[t_idx, j_idx] = object_mesh_verts[t_idx, subseq_contact_v_id] + \
                        curr_contact_normal  # 3  
                
                elif all_contact_labels[t_idx, j_idx] and not continue_prev_contact: # The first contact frame 
                    subseq_contact_v_id = all_object_c_idx_list[t_idx, j_idx]
                    subseq_contact_pos = object_mesh_verts[t_idx, subseq_contact_v_id] # 3 

                    contact_normal = new_hand_foot_jpos[t_idx, j_idx] - subseq_contact_pos # Keep using this in the following frames. 

                    reference_obj_rot = obj_rot[t_idx] # 3 X 3 

                    continue_prev_contact = True 

        return new_hand_foot_jpos 

    def gen_vis_res(self, all_res_list, data_dict, step, vis_gt=False, vis_tag=None):
        # all_res_list: BS X T X 12  
        lhand_color = np.asarray([255, 87, 51])  # red 
        rhand_color = np.asarray([17, 99, 226]) # blue
        lfoot_color = np.asarray([134, 17, 226]) # purple 
        rfoot_color = np.asarray([22, 173, 100]) # green 

        contact_pcs_colors = []
        contact_pcs_colors.append(lhand_color)
        contact_pcs_colors.append(rhand_color)
        contact_pcs_colors.append(lfoot_color)
        contact_pcs_colors.append(rfoot_color)
        contact_pcs_colors = np.asarray(contact_pcs_colors) # 4 X 3 
        
        seq_names = data_dict['seq_name'] # BS 
        seq_len = data_dict['seq_len'].detach().cpu().numpy() # BS 

        # obj_rot = data_dict['obj_rot_mat'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 X 3
        obj_com_pos = data_dict['obj_com_pos'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 

        num_seq, num_steps, _ = all_res_list.shape
        
        normalized_gt_hand_foot_pos = self.extract_palm_jpos_only_data(data_dict['motion']) 
        # Denormalize hand only 
        pred_hand_foot_pos = self.ds.de_normalize_jpos_min_max_hand_foot(all_res_list, hand_only=True) # BS X T X 2 X 3 

        gt_hand_foot_pos = self.ds.de_normalize_jpos_min_max_hand_foot(normalized_gt_hand_foot_pos, hand_only=True) # BS X T X 2 X 3
        gt_hand_foot_pos = gt_hand_foot_pos.reshape(-1, num_steps, 2, 3) 
        
        all_processed_hand_jpos = pred_hand_foot_pos.clone() 

        for seq_idx in range(num_seq):
            object_name = seq_names[seq_idx].split("_")[1]
            obj_scale = data_dict['obj_scale'][seq_idx].detach().cpu().numpy()
            obj_trans = data_dict['obj_trans'][seq_idx].detach().cpu().numpy()
            obj_rot = data_dict['obj_rot_mat'][seq_idx].detach().cpu().numpy() 
            if object_name in ["mop", "vacuum"]:
                obj_bottom_scale = data_dict['obj_bottom_scale'][seq_idx].detach().cpu().numpy() 
                obj_bottom_trans = data_dict['obj_bottom_trans'][seq_idx].detach().cpu().numpy()
                obj_bottom_rot = data_dict['obj_bottom_rot_mat'][seq_idx].detach().cpu().numpy()
            else:
                obj_bottom_scale = None 
                obj_bottom_trans = None 
                obj_bottom_rot = None 

            obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(object_name, \
            obj_scale, obj_trans, obj_rot, \
            obj_bottom_scale, obj_bottom_trans, obj_bottom_rot)

            # Add postprocessing for hand positions. 
            if self.add_hand_processing:
                curr_seq_pred_hand_foot_jpos = self.process_hand_foot_contact_jpos(pred_hand_foot_pos[seq_idx], \
                                    obj_mesh_verts, obj_mesh_faces, obj_rot)

                all_processed_hand_jpos[seq_idx] = curr_seq_pred_hand_foot_jpos 
            else:
                curr_seq_pred_hand_foot_jpos = pred_hand_foot_pos[seq_idx]

        if self.use_gt_hand_for_eval:
            all_processed_hand_jpos = self.ds.normalize_jpos_min_max_hand_foot(gt_hand_foot_pos.cuda())
        else:
            all_processed_hand_jpos = self.ds.normalize_jpos_min_max_hand_foot(all_processed_hand_jpos) # BS X T X 4 X 3 

        gt_hand_foot_pos = self.ds.normalize_jpos_min_max_hand_foot(gt_hand_foot_pos.cuda())

        return all_processed_hand_jpos, gt_hand_foot_pos  

    def run_two_stage_pipeline(self):
        fullbody_wdir = os.path.join(self.opt.project, self.opt.fullbody_exp_name, "weights")
       
        repr_dim = 24 * 3 + 22 * 6 
    
        loss_type = "l1"
        
        # Create full body diffusion model. 
        fullbody_diffusion_model = FullBodyCondGaussianDiffusion(self.opt, d_feats=repr_dim, d_model=opt.d_model, \
                    n_dec_layers=self.opt.n_dec_layers, n_head=self.opt.n_head, d_k=self.opt.d_k, d_v=self.opt.d_v, \
                    max_timesteps=self.opt.window+1, out_dim=repr_dim, timesteps=1000, \
                    objective="pred_x0", loss_type=loss_type, \
                    batch_size=self.opt.batch_size)
        fullbody_diffusion_model.to(device)

        fullbody_trainer = FullBodyTrainer(
            self.opt,
            fullbody_diffusion_model,
            train_batch_size=32, # 32
            train_lr=1e-4, # 1e-4
            train_num_steps=8000000,         # total training steps
            gradient_accumulate_every=2,    # gradient accumulation steps
            ema_decay=0.995,                # exponential moving average decay
            amp=True,                        # turn on mixed precision
            results_folder=fullbody_wdir,
            use_wandb=False 
        )
        fullbody_trainer.load(milestone=0, pretrained_path=self.opt.fullbody_checkpoint)
        fullbody_trainer.ema.ema_model.eval()
      
        # Load pretrained mdoel for stage 1 
        self.load(milestone=0, pretrained_path=self.opt.checkpoint)
        self.ema.ema_model.eval()

        s1_global_hand_jpe_list = [] 
        s1_global_lhand_jpe_list = []
        s1_global_rhand_jpe_list = [] 

        global_hand_jpe_list = [] 
        global_lhand_jpe_list = []
        global_rhand_jpe_list = [] 

        mpvpe_list = []
        mpjpe_list = []

        rot_dist_list = []
        root_trans_err_list = []
        
        collision_percent_list = []
        collision_depth_list = []

        gt_collision_percent_list = []
        gt_collision_depth_list = []

        gt_foot_sliding_jnts_list = []
        foot_sliding_jnts_list = []

        contact_precision_list = []
        contact_recall_list = [] 

        contact_acc_list = []
        contact_f1_score_list = []

        gt_contact_dist_list = []
        contact_dist_list = []

        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=8, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False) 
        
        if self.for_quant_eval:
            num_samples_per_seq = 20
        else:
            num_samples_per_seq = 1

        with torch.no_grad():
            for s_idx, val_data_dict in enumerate(test_loader):

                if (not s_idx % 8 == 0) and (not self.for_quant_eval): # Visualize part of data
                    continue 

                val_data = val_data_dict['motion'].cuda()

                bs, num_steps, _ = val_data.shape 

                val_data = self.extract_palm_jpos_only_data(val_data)

                obj_bps_data = val_data_dict['obj_bps'].cuda()
                obj_com_pos = val_data_dict['obj_com_pos'].cuda() 

                ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1) # BS X T X (3+1024*3)

                cond_mask = None 

                # Generate padding mask 
                actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                # Each sequence, sample multiple times to compute metrics. 
                s1_lhand_jpe_per_seq = []
                s1_rhand_jpe_per_seq = []
                s1_hand_jpe_per_seq = [] 
               
                hand_jpe_per_seq = []
                lhand_jpe_per_seq = []
                rhand_jpe_per_seq = []

                mpvpe_per_seq = []
                mpjpe_per_seq = []
                
                rot_dist_per_seq = []
                trans_err_per_seq = []
                
                gt_foot_sliding_jnts_per_seq = []
                foot_sliding_jnts_per_seq = []
                
                contact_precision_per_seq = []
                contact_recall_per_seq = [] 

                contact_acc_per_seq = []
                contact_f1_score_per_seq = [] 

                gt_contact_dist_per_seq = []
                contact_dist_per_seq = []

                sampled_all_res_per_seq = []
                for sample_idx in range(num_samples_per_seq):
                    # Stage 1 
                    pred_hand_foot_jpos = self.ema.ema_model.sample(val_data, ori_data_cond, \
                    cond_mask=cond_mask, padding_mask=padding_mask)

                    vis_tag = "stage1_sample_"+str(s_idx)
                    if self.add_hand_processing:
                        vis_tag = vis_tag + "_add_hand_processing"

                    if self.test_on_train:
                        vis_tag = vis_tag + "_on_train"

                    if self.use_object_split:
                        vis_tag += "_unseen_objects"

                    pred_hand_foot_jpos, gt_hand_foot_pos = self.gen_vis_res(pred_hand_foot_jpos, \
                    val_data_dict, 0, vis_tag=vis_tag)
                
                    tmp_pred_hand_jpos = self.ds.de_normalize_jpos_min_max_hand_foot(pred_hand_foot_jpos.reshape(bs, num_steps, -1), hand_only=True) # BS X T X 2 X 3 
                    tmp_gt_hand_jpos = self.ds.de_normalize_jpos_min_max_hand_foot(gt_hand_foot_pos.reshape(bs, num_steps, -1), hand_only=True)

                    for s1_s_idx in range(bs): 
                        s1_lhand_jpe, s1_rhand_jpe, s1_hand_jpe = compute_s1_metrics(tmp_pred_hand_jpos[s1_s_idx, \
                            :actual_seq_len[s1_s_idx]], tmp_gt_hand_jpos[s1_s_idx, :actual_seq_len[s1_s_idx]])
                      
                        s1_lhand_jpe_per_seq.append(s1_lhand_jpe)
                        s1_rhand_jpe_per_seq.append(s1_rhand_jpe)
                        s1_hand_jpe_per_seq.append(s1_hand_jpe)

                    # Feed the predicted hand and foot position to full-body diffusion model. 
                    all_res_list = fullbody_trainer.gen_fullbody_from_predicted_hand_foot(pred_hand_foot_jpos, val_data_dict)

                    sampled_all_res_per_seq.append(all_res_list) 

                    vis_tag = "two_stage_pipeline_sample_"+str(s_idx)+"_try_"+str(sample_idx)

                    if self.add_hand_processing:
                        vis_tag = vis_tag + "_add_hand_processing"

                    if self.test_on_train:
                        vis_tag = vis_tag + "_on_train"

                    if self.use_object_split:
                        vis_tag += "_unseen_objects"

                    if self.use_gt_hand_for_eval:
                        vis_tag += "_use_gt_hand"

                    num_seq = all_res_list.shape[0]
                    for seq_idx in range(num_seq):

                        # A trick to fix artifacts when using add_hand_processing.
                        # The artifact is that when the hand positions are the same in a row, the root translation would be suddenly changed. 
                        if self.add_hand_processing:
                            tmp_pred_hand_jpos = pred_hand_foot_jpos[seq_idx] # T X 2 X 3 
                            tmp_num_steps = actual_seq_len[seq_idx]-1
                            
                            repeat_idx = None 
                            for tmp_idx in range(tmp_num_steps-5, tmp_num_steps):
                                hand_jpos_diff = tmp_pred_hand_jpos[tmp_idx] - tmp_pred_hand_jpos[tmp_idx-1] # 2 X 3 
                                threshold = 0.001
                             
                                if (torch.abs(hand_jpos_diff[0, 0]) < threshold and torch.abs(hand_jpos_diff[0, 1]) < threshold \
                                and torch.abs(hand_jpos_diff[0, 2]) < threshold) or (torch.abs(hand_jpos_diff[1, 0]) < threshold \
                                and torch.abs(hand_jpos_diff[1, 1]) < threshold and torch.abs(hand_jpos_diff[1, 2]) < threshold):
                                    repeat_idx = tmp_idx 
                                    break 
                            
                            if repeat_idx is not None:
                                padding_last = all_res_list[seq_idx:seq_idx+1, repeat_idx-1:repeat_idx] # 1 X 1 X 198 
                                padding_last = padding_last.repeat(1, pred_hand_foot_jpos.shape[1]-repeat_idx, 1) # 1 X t' X D 
                                
                                curr_seq_res_list = torch.cat((all_res_list[seq_idx:seq_idx+1, :repeat_idx], padding_last), dim=1)
                            else:
                                curr_seq_res_list = all_res_list[seq_idx:seq_idx+1]
                        else:
                            curr_seq_res_list = all_res_list[seq_idx:seq_idx+1]

                        curr_vis_tag = vis_tag + "_seq_idx_in_bs_"+str(seq_idx) 
                      
                        pred_human_trans_list, pred_human_rot_list, pred_human_jnts_list, pred_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            fullbody_trainer.gen_vis_res(curr_seq_res_list, val_data_dict, \
                            0, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx)
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            fullbody_trainer.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                            0, vis_gt=True, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx)
                    
                        lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, gt_contact_dist, contact_dist, \
                                gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
                                contact_acc, contact_f1_score = \
                                compute_metrics(gt_human_verts_list, pred_human_verts_list, gt_human_jnts_list, pred_human_jnts_list, human_faces_list, \
                                gt_human_trans_list, pred_human_trans_list, gt_human_rot_list, pred_human_rot_list, \
                                obj_verts_list, obj_faces_list, actual_len_list, use_joints24=True)

                        hand_jpe_per_seq.append(hand_jpe)
                        lhand_jpe_per_seq.append(lhand_jpe)
                        rhand_jpe_per_seq.append(rhand_jpe) 

                        mpvpe_per_seq.append(mpvpe)
                        mpjpe_per_seq.append(mpjpe)
                        
                        rot_dist_per_seq.append(rot_dist)
                        trans_err_per_seq.append(trans_err) 
                        
                        gt_foot_sliding_jnts_per_seq.append(gt_foot_sliding_jnts)
                        foot_sliding_jnts_per_seq.append(foot_sliding_jnts)
                        
                        contact_precision_per_seq.append(contact_precision)
                        contact_recall_per_seq.append(contact_recall) 

                        contact_acc_per_seq.append(contact_acc) 
                        contact_f1_score_per_seq.append(contact_f1_score) 

                        gt_contact_dist_per_seq.append(gt_contact_dist)
                        contact_dist_per_seq.append(contact_dist)

                        # print("*****************************************Single Sequence*****************************************")
                        # print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(lhand_jpe, rhand_jpe, hand_jpe))
                        # print("MPJPE: {0}, MPVPE: {1}, Root Trans: {2}, Global Rot Err: {3}".format(mpjpe, mpvpe, trans_err, rot_dist))
                        # print("Foot sliding verts: {0}, Foot sliding jnts: {1}".format(foot_sliding_verts, foot_sliding_jnts))
                        # print("Collision percent: {0}, Collision depth: {1}".format(collision_percent, mean_collide_depth))
    
                        # fullbody_trainer.gen_vis_res(curr_seq_res_list, val_data_dict, \
                        #     milestone, vis_tag=vis_tag, selected_seq_idx=seq_idx)
                        # fullbody_trainer.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                        #     milestone, vis_gt=True, vis_tag=vis_tag, selected_seq_idx=seq_idx)

                        # break 
        
                if self.for_quant_eval:
                    s1_lhand_jpe_per_seq = np.asarray(s1_lhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq)
                    s1_rhand_jpe_per_seq = np.asarray(s1_rhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq)
                    s1_hand_jpe_per_seq = np.asarray(s1_hand_jpe_per_seq).reshape(num_samples_per_seq, num_seq)

                    hand_jpe_per_seq = np.asarray(hand_jpe_per_seq).reshape(num_samples_per_seq, num_seq)
                    lhand_jpe_per_seq = np.asarray(lhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq)
                    rhand_jpe_per_seq = np.asarray(rhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 

                    mpvpe_per_seq = np.asarray(mpvpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    mpjpe_per_seq = np.asarray(mpjpe_per_seq).reshape(num_samples_per_seq, num_seq) # Sample_num X BS 
                    
                    rot_dist_per_seq = np.asarray(rot_dist_per_seq).reshape(num_samples_per_seq, num_seq)
                    trans_err_per_seq = np.asarray(trans_err_per_seq).reshape(num_samples_per_seq, num_seq) 
                    
                    gt_foot_sliding_jnts_per_seq = np.asarray(gt_foot_sliding_jnts_per_seq).reshape(num_samples_per_seq, num_seq)
                    foot_sliding_jnts_per_seq = np.asarray(foot_sliding_jnts_per_seq).reshape(num_samples_per_seq, num_seq) 
                    
                    contact_precision_per_seq = np.asarray(contact_precision_per_seq).reshape(num_samples_per_seq, num_seq) 
                    contact_recall_per_seq = np.asarray(contact_recall_per_seq).reshape(num_samples_per_seq, num_seq) 

                    contact_acc_per_seq = np.asarray(contact_acc_per_seq).reshape(num_samples_per_seq, num_seq) 
                    contact_f1_score_per_seq = np.asarray(contact_f1_score_per_seq).reshape(num_samples_per_seq, num_seq) 

                    gt_contact_dist_per_seq = np.asarray(gt_contact_dist_per_seq).reshape(num_samples_per_seq, num_seq)
                    contact_dist_per_seq = np.asarray(contact_dist_per_seq).reshape(num_samples_per_seq, num_seq) 

                    best_sample_idx = mpjpe_per_seq.argmin(axis=0) # sample_num 

                    s1_hand_jpe = s1_hand_jpe_per_seq[best_sample_idx, list(range(num_seq))]
                    s1_lhand_jpe = s1_lhand_jpe_per_seq[best_sample_idx, list(range(num_seq))]
                    s1_rhand_jpe = s1_rhand_jpe_per_seq[best_sample_idx, list(range(num_seq))]

                    hand_jpe = hand_jpe_per_seq[best_sample_idx, list(range(num_seq))] # BS 
                    lhand_jpe = lhand_jpe_per_seq[best_sample_idx, list(range(num_seq))]
                    rhand_jpe = rhand_jpe_per_seq[best_sample_idx, list(range(num_seq))]
                    
                    mpvpe = mpvpe_per_seq[best_sample_idx, list(range(num_seq))]
                    mpjpe = mpjpe_per_seq[best_sample_idx, list(range(num_seq))]
                    
                    rot_dist = rot_dist_per_seq[best_sample_idx, list(range(num_seq))]
                    trans_err = trans_err_per_seq[best_sample_idx, list(range(num_seq))]
                   
                    gt_foot_sliding_jnts = gt_foot_sliding_jnts_per_seq[best_sample_idx, list(range(num_seq))]
                    foot_sliding_jnts = foot_sliding_jnts_per_seq[best_sample_idx, list(range(num_seq))]

                    contact_precision_seq = contact_precision_per_seq[best_sample_idx, list(range(num_seq))]
                    contact_recall_seq = contact_recall_per_seq[best_sample_idx, list(range(num_seq))] 
                    contact_acc_seq = contact_acc_per_seq[best_sample_idx, list(range(num_seq))]
                    contact_f1_score_seq = contact_f1_score_per_seq[best_sample_idx, list(range(num_seq))]

                    gt_contact_dist_seq = gt_contact_dist_per_seq[best_sample_idx, list(range(num_seq))]
                    contact_dist_seq = contact_dist_per_seq[best_sample_idx, list(range(num_seq))] 

                    sampled_all_res_per_seq = torch.stack(sampled_all_res_per_seq) # K X BS X T X D 
                    best_sampled_all_res = sampled_all_res_per_seq[best_sample_idx, list(range(num_seq))] # BS X T X D 
                    num_seq = best_sampled_all_res.shape[0]
                    for seq_idx in range(num_seq):
                        pred_human_trans_list, pred_human_rot_list, pred_human_jnts_list, pred_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            fullbody_trainer.gen_vis_res(best_sampled_all_res[seq_idx:seq_idx+1], val_data_dict, \
                            0, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx)
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            fullbody_trainer.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                            0, vis_gt=True, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx)

                        obj_scale = val_data_dict['obj_scale'][seq_idx]
                        obj_trans = val_data_dict['obj_trans'][seq_idx]
                        obj_rot_mat = val_data_dict['obj_rot_mat'][seq_idx]
                        actual_len = val_data_dict['seq_len'][seq_idx]
                        object_name = val_data_dict['obj_name'][seq_idx]

                        pred_collision_percent, pred_collision_depth = compute_collision(pred_human_verts_list.cpu(), \
                            human_faces_list, obj_verts_list.cpu(), obj_faces_list, object_name, \
                            obj_scale, obj_rot_mat, obj_trans, actual_len)
                            
                        gt_collision_percent, gt_collision_depth = compute_collision(gt_human_verts_list.cpu(), \
                            human_faces_list, obj_verts_list.cpu(), obj_faces_list, object_name, \
                            obj_scale, obj_rot_mat, obj_trans, actual_len)

                        collision_percent_list.append(pred_collision_percent)
                        collision_depth_list.append(pred_collision_depth)
                        gt_collision_percent_list.append(gt_collision_percent)
                        gt_collision_depth_list.append(gt_collision_depth) 
                        
                    for tmp_seq_idx in range(num_seq):
                        s1_global_lhand_jpe_list.append(s1_lhand_jpe[tmp_seq_idx])
                        s1_global_rhand_jpe_list.append(s1_rhand_jpe[tmp_seq_idx])
                        s1_global_hand_jpe_list.append(s1_hand_jpe[tmp_seq_idx])

                        global_hand_jpe_list.append(hand_jpe[tmp_seq_idx])
                        global_lhand_jpe_list.append(lhand_jpe[tmp_seq_idx])
                        global_rhand_jpe_list.append(rhand_jpe[tmp_seq_idx])

                        mpvpe_list.append(mpvpe[tmp_seq_idx])
                        mpjpe_list.append(mpjpe[tmp_seq_idx])
                        
                        rot_dist_list.append(rot_dist[tmp_seq_idx])
                        root_trans_err_list.append(trans_err[tmp_seq_idx])
                        
                        gt_foot_sliding_jnts_list.append(gt_foot_sliding_jnts[tmp_seq_idx])
                        foot_sliding_jnts_list.append(foot_sliding_jnts[tmp_seq_idx])

                        contact_precision_list.append(contact_precision_seq[tmp_seq_idx])
                        contact_recall_list.append(contact_recall_seq[tmp_seq_idx])

                        contact_acc_list.append(contact_acc_seq[tmp_seq_idx])
                        contact_f1_score_list.append(contact_f1_score_seq[tmp_seq_idx])

                        gt_contact_dist_list.append(gt_contact_dist_seq[tmp_seq_idx])
                        contact_dist_list.append(contact_dist_seq[tmp_seq_idx])

                # if s_idx > 0:
                #     break 

        if self.for_quant_eval:
            s1_mean_hand_jpe = np.asarray(s1_global_hand_jpe_list).mean()
            s1_mean_lhand_jpe = np.asarray(s1_global_lhand_jpe_list).mean()
            s1_mean_rhand_jpe = np.asarray(s1_global_rhand_jpe_list).mean() 

            mean_hand_jpe = np.asarray(global_hand_jpe_list).mean() 
            mean_lhand_jpe = np.asarray(global_lhand_jpe_list).mean()
            mean_rhand_jpe = np.asarray(global_rhand_jpe_list).mean()
            
            mean_mpvpe = np.asarray(mpvpe_list).mean()
            mean_mpjpe = np.asarray(mpjpe_list).mean() 

            mean_rot_dist = np.asarray(rot_dist_list).mean() 
            mean_root_trans_err = np.asarray(root_trans_err_list).mean()
            
            mean_collision_percent = np.asarray(collision_percent_list).mean()
            mean_collision_depth = np.asarray(collision_depth_list).mean() 

            gt_mean_collision_percent = np.asarray(gt_collision_percent_list).mean()
            gt_mean_collision_depth = np.asarray(gt_collision_depth_list).mean() 
            
            mean_gt_fsliding_jnts = np.asarray(gt_foot_sliding_jnts_list).mean()
            mean_fsliding_jnts = np.asarray(foot_sliding_jnts_list).mean() 

            mean_contact_precision = np.asarray(contact_precision_list).mean()
            mean_contact_recall = np.asarray(contact_recall_list).mean() 

            mean_contact_acc = np.asarray(contact_acc_list).mean()
            mean_contact_f1_score = np.asarray(contact_f1_score_list).mean() 

            mean_gt_contact_dist = np.asarray(gt_contact_dist_list).mean()
            mean_contact_dist = np.asarray(contact_dist_list).mean()

            print("*****************************************Quantitative Evaluation*****************************************")
            print("The number of sequences: {0}".format(len(mpjpe_list)))
            print("Stage 1 Left Hand JPE: {0}, Stage 1 Right Hand JPE: {1}, Stage 1 Two Hands JPE: {2}".format(s1_mean_lhand_jpe, s1_mean_rhand_jpe, s1_mean_hand_jpe))
            print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
            print("MPJPE: {0}, MPVPE: {1}, Root Trans: {2}, Global Rot Err: {3}".format(mean_mpjpe, mean_mpvpe, mean_root_trans_err, mean_rot_dist))
            print("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
            print("Collision percent: {0}, Collision depth: {1}".format(mean_collision_percent, mean_collision_depth))
            print("GT Collision percent: {0}, GT Collision depth: {1}".format(gt_mean_collision_percent, gt_mean_collision_depth))
            print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
            print("Contact Acc: {0}, Contact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score)) 
            print("Contact dist: {0}, GT Contact dist: {1}".format(mean_contact_dist, mean_gt_contact_dist))

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 2 * 3 
   
    loss_type = "l1"
  
    diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)
   
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=400000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
    )

    trainer.train()

    torch.cuda.empty_cache()

def run_sample(opt, device, run_pipeline=False):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model 
    repr_dim = 2 * 3 
    
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=400000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False 
    )
    
    if run_pipeline:
        trainer.run_two_stage_pipeline() 
    else:
        trainer.cond_sample_res()

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='output folder for weights and visualizations')
    parser.add_argument('--wandb_pj_name', type=str, default='wandb_proj_name', help='wandb project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='stage1_exp_out', help='save to project/exp_name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--fullbody_exp_name', default='stage2_exp_out', help='project/fullbody_exp_name')
    parser.add_argument('--fullbody_checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    # For running the whole pipeline. 
    parser.add_argument("--run_whole_pipeline", action="store_true")

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

    parser.add_argument("--use_gt_hand_for_eval", action="store_true")

    parser.add_argument("--use_object_split", action="store_true")

    parser.add_argument('--data_root_folder', default='data', help='root folder for dataset')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    elif opt.run_whole_pipeline:
        run_sample(opt, device, run_pipeline=True)
    else:
        run_train(opt, device)
