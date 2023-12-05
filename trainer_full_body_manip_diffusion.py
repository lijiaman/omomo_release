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

from manip.model.transformer_fullbody_cond_diffusion_model import CondGaussianDiffusion, cosine_beta_schedule 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

from evaluation_metrics import compute_metrics 
from evaluation_metrics import compute_collision

from matplotlib import pyplot as plt

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=False):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=int)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=int)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
        lmiddle_index= 28 
        rmiddle_index = 43 
        x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

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
        use_wandb=True   
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

        self.data_root_folder = self.opt.data_root_folder 

        self.window = opt.window

        self.use_object_split = self.opt.use_object_split
        self.prep_dataloader(window_size=opt.window)

        self.bm_dict = self.ds.bm_dict 

        self.test_on_train = self.opt.test_sample_res_on_train 

        self.add_hand_processing = self.opt.add_hand_processing  

        self.for_quant_eval = self.opt.for_quant_eval 

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

    def prep_joint_condition_mask(self, data, joint_idx, pos_only):
        # data: BS X T X D 
        # head_idx = 15 
        # hand_idx = 20, 21 
        # Condition part is zeros, while missing part is ones. 
        mask = torch.ones_like(data).to(data.device)

        cond_pos_dim_idx = joint_idx * 3 
        cond_rot_dim_idx = 24 * 3 + joint_idx * 6
        
        mask[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(data.shape[0], data.shape[1], 3).to(data.device)
        
        if not pos_only:
            mask[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(data.shape[0], data.shape[1], 6).to(data.device)

        return mask 

    def train(self):
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                data = data_dict['motion'].cuda()

                cond_mask = None 

                left_joint_mask = self.prep_joint_condition_mask(data, joint_idx=22, pos_only=True)
                right_joint_mask = self.prep_joint_condition_mask(data, joint_idx=23, pos_only=True)

                if cond_mask is not None:
                    cond_mask = cond_mask * left_joint_mask * right_joint_mask 
                else:
                    cond_mask = left_joint_mask * right_joint_mask 

                # Generate padding mask 
                actual_seq_len = data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(data.device)

                with autocast(enabled = self.amp):    
                    loss_diffusion = self.model(data, cond_mask, padding_mask)
                    
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

                    if idx % 50 == 0 and i == 0:
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

                    cond_mask = None  

                    left_joint_mask = self.prep_joint_condition_mask(val_data, joint_idx=22, pos_only=True)
                    right_joint_mask = self.prep_joint_condition_mask(val_data, joint_idx=23, pos_only=True)

                    if cond_mask is not None:
                        cond_mask = cond_mask * left_joint_mask * right_joint_mask 
                    else:
                        cond_mask = left_joint_mask * right_joint_mask 

                    # Generate padding mask 
                    actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                    tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                    self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_data.device)

                    # Get validation loss 
                    val_loss_diffusion = self.model(val_data, cond_mask, padding_mask)
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

                        all_res_list = self.ema.ema_model.sample(val_data, cond_mask, padding_mask)
                        all_res_list = all_res_list[:bs_for_vis]

                        # Visualization
                        for_vis_gt_data = val_data[:bs_for_vis]
                        self.gen_vis_res(for_vis_gt_data, val_data_dict, self.step, vis_gt=True)
                        self.gen_vis_res(all_res_list, val_data_dict, self.step)

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
        
        foot_sliding_jnts_list = []
        gt_foot_sliding_jnts_list = []
        
        contact_precision_list = []
        contact_recall_list = [] 
        contact_acc_list = []
        contact_f1_score_list = [] 

        contact_dist_list = []
        gt_contact_dist_list = []
      
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
                val_data = val_data_dict['motion'].cuda()

                cond_mask = None 

                left_joint_mask = self.prep_joint_condition_mask(val_data, joint_idx=22, pos_only=True)
                right_joint_mask = self.prep_joint_condition_mask(val_data, joint_idx=23, pos_only=True)

                if cond_mask is not None:
                    cond_mask = cond_mask * left_joint_mask * right_joint_mask 
                else:
                    cond_mask = left_joint_mask * right_joint_mask 

                # Generate padding mask 
                actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
                self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                hand_jpe_per_seq = []
                lhand_jpe_per_seq = []
                rhand_jpe_per_seq = []

                mpvpe_per_seq = []
                mpjpe_per_seq = []
                
                rot_dist_per_seq = []
                trans_err_per_seq = []
                
                gt_foot_sliding_jnts_per_seq = []
                foot_sliding_jnts_per_seq = []
                
                gt_contact_dist_per_seq = []
                contact_dist_per_seq = [] 

                contact_precision_per_seq = []
                contact_recall_per_seq = [] 

                contact_acc_per_seq = [] 
                contact_f1_score_per_seq = [] 

                sampled_all_res_per_seq = [] 

                for sample_idx in range(num_samples_per_seq):
                    all_res_list = self.ema.ema_model.sample(val_data, \
                    cond_mask=cond_mask, padding_mask=padding_mask) # BS X T X D 

                    sampled_all_res_per_seq.append(all_res_list) 

                    vis_tag = str(milestone)+"_sidx_"+str(s_idx)+"_sample_cnt_"+str(sample_idx)
                 
                    if self.test_on_train:
                        vis_tag = vis_tag + "_on_train"

                    num_seq = all_res_list.shape[0]
                    for seq_idx in range(num_seq):
                        curr_vis_tag = vis_tag + "_seq_idx_in_bs_"+str(seq_idx) 
                        pred_human_trans_list, pred_human_rot_list, pred_human_jnts_list, pred_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            self.gen_vis_res(all_res_list[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx)
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            self.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_gt=True, vis_tag=curr_vis_tag, for_quant_eval=self.for_quant_eval, selected_seq_idx=seq_idx)
                    
                        lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
                                gt_contact_dist, contact_dist, \
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

                if self.for_quant_eval:
                    hand_jpe_per_seq = np.asarray(hand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    lhand_jpe_per_seq = np.asarray(lhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    rhand_jpe_per_seq = np.asarray(rhand_jpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                  
                    mpvpe_per_seq = np.asarray(mpvpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                    mpjpe_per_seq = np.asarray(mpjpe_per_seq).reshape(num_samples_per_seq, num_seq) 
                  
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
                            self.gen_vis_res(best_sampled_all_res[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx)
                        gt_human_trans_list, gt_human_rot_list, gt_human_jnts_list, gt_human_verts_list, human_faces_list, \
                            obj_verts_list, obj_faces_list, actual_len_list = \
                            self.gen_vis_res(val_data_dict['motion'].cuda()[seq_idx:seq_idx+1], val_data_dict, \
                            milestone, vis_gt=True, vis_tag=vis_tag, for_quant_eval=True, selected_seq_idx=seq_idx)

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

                    # Get the min error 
                    for tmp_seq_idx in range(num_seq):
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

        if self.for_quant_eval:
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
            print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
            print("MPJPE: {0}, MPVPE: {1}, Root Trans: {2}, Global Rot Err: {3}".format(mean_mpjpe, mean_mpvpe, mean_root_trans_err, mean_rot_dist))
            print("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
            print("Collision percent: {0}, Collision depth: {1}".format(mean_collision_percent, mean_collision_depth))
            print("GT Collision percent: {0}, GT Collision depth: {1}".format(gt_mean_collision_percent, gt_mean_collision_depth))
            print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
            print("Contact Acc: {0}, COntact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score))
            print("Contact dist: {0}, GT Contact dist: {1}".format(mean_contact_dist, mean_gt_contact_dist))

    def gen_vis_res(self, all_res_list, data_dict, step, vis_gt=False, vis_tag=None, \
        for_quant_eval=False, selected_seq_idx=None):
        # all_res_list: N X T X D 
        num_seq = all_res_list.shape[0]

        num_joints = 24
        
        normalized_global_jpos = all_res_list[:, :, :num_joints*3].reshape(num_seq, -1, num_joints, 3)
    
        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, num_joints, 3))
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 22 X 3 
        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_rot_6d = all_res_list[:, :, -22*6:].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        trans2joint = data_dict['trans2joint'].to(all_res_list.device) # N X 3

        seq_len = data_dict['seq_len'].detach().cpu().numpy() # BS 
      
        # Used for quantitative evaluation. 
        human_trans_list = [] 
        human_rot_list = [] 
        human_jnts_list = []
        human_verts_list = []
        human_faces_list = []

        obj_verts_list = []
        obj_faces_list = [] 

        actual_len_list = []

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
          
            if selected_seq_idx is None:
                curr_trans2joint = trans2joint[idx:idx+1].clone()
            else:
                curr_trans2joint = trans2joint[selected_seq_idx:selected_seq_idx+1].clone()

            root_trans = curr_global_root_jpos + curr_trans2joint # T X 3 
         
            # Generate global joint position 
            bs = 1
            if selected_seq_idx is None:
                betas = data_dict['betas'][idx]
                gender = data_dict['gender'][idx]
                curr_obj_rot_mat = data_dict['obj_rot_mat'][idx]
                curr_obj_trans = data_dict['obj_trans'][idx]
                curr_obj_scale = data_dict['obj_scale'][idx]
                curr_seq_name = data_dict['seq_name'][idx]
                object_name = curr_seq_name.split("_")[1]
            else:
                betas = data_dict['betas'][selected_seq_idx]
                gender = data_dict['gender'][selected_seq_idx]
                curr_obj_rot_mat = data_dict['obj_rot_mat'][selected_seq_idx]
                curr_obj_trans = data_dict['obj_trans'][selected_seq_idx]
                curr_obj_scale = data_dict['obj_scale'][selected_seq_idx]
                curr_seq_name = data_dict['seq_name'][selected_seq_idx]
                object_name = curr_seq_name.split("_")[1]

            # Get human verts 
            mesh_jnts, mesh_verts, mesh_faces = \
                run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
                betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)

            # Get object verts 
            if object_name in ["mop", "vacuum"]:
                if selected_seq_idx is None:
                    curr_obj_bottom_rot_mat = data_dict['obj_bottom_rot_mat'][idx]
                    curr_obj_bottom_trans = data_dict['obj_bottom_trans'][idx]
                    curr_obj_bottom_scale = data_dict['obj_bottom_scale'][idx]
                else:
                    curr_obj_bottom_rot_mat = data_dict['obj_bottom_rot_mat'][selected_seq_idx]
                    curr_obj_bottom_trans = data_dict['obj_bottom_trans'][selected_seq_idx]
                    curr_obj_bottom_scale = data_dict['obj_bottom_scale'][selected_seq_idx]

                obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(object_name, \
                    curr_obj_scale.detach().cpu().numpy(), curr_obj_trans.detach().cpu().numpy(), \
                    curr_obj_rot_mat.detach().cpu().numpy(), \
                    curr_obj_bottom_scale.detach().cpu().numpy(), \
                    curr_obj_bottom_trans.detach().cpu().numpy(), \
                    curr_obj_bottom_rot_mat.detach().cpu().numpy(), \
                    ) 
            else:
                obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(object_name, \
                    curr_obj_scale.detach().cpu().numpy(), curr_obj_trans.detach().cpu().numpy(), \
                    curr_obj_rot_mat.detach().cpu().numpy()) 

           
            human_trans_list.append(root_trans) 
            human_jnts_list.append(mesh_jnts)
            human_verts_list.append(mesh_verts)
            human_faces_list.append(mesh_faces) 

            human_rot_list.append(curr_global_rot_mat)

            obj_verts_list.append(obj_mesh_verts)
            obj_faces_list.append(obj_mesh_faces) 

            if selected_seq_idx is None:
                actual_len_list.append(seq_len[idx])
            else:
                actual_len_list.append(seq_len[selected_seq_idx])
            
            if vis_tag is None:
                dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
            else:
                dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))
            
            if not self.for_quant_eval:
                if not os.path.exists(dest_mesh_vis_folder):
                    os.makedirs(dest_mesh_vis_folder)

                if vis_gt:
                    mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                    "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                    out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                    "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                    out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                    "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
                else:
                    mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                    "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                    out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                    "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                    out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                    "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")

                if selected_seq_idx is None:
                    actual_len = seq_len[idx]
                else:
                    actual_len = seq_len[selected_seq_idx]

                if not vis_gt:
                    save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0][:actual_len], mesh_faces.detach().cpu().numpy(), \
                        obj_mesh_verts.detach().cpu().numpy()[:actual_len], obj_mesh_faces, mesh_save_folder)
                    run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_file_path, vis_object=True)
         
        human_trans_list = torch.stack(human_trans_list)[0] # T X 3
        human_rot_list = torch.stack(human_rot_list)[0] # T X 22 X 3 X 3 
        human_jnts_list = torch.stack(human_jnts_list)[0, 0] # T X 22 X 3 
        human_verts_list = torch.stack(human_verts_list)[0, 0] # T X Nv X 3 
        human_faces_list = torch.stack(human_faces_list)[0].detach().cpu().numpy() # Nf X 3 

        obj_verts_list = torch.stack(obj_verts_list)[0] # T X Nv' X 3 
        obj_faces_list = np.asarray(obj_faces_list)[0] # Nf X 3

        actual_len_list = np.asarray(actual_len_list)[0] # scalar value 

        return human_trans_list, human_rot_list, human_jnts_list, human_verts_list, human_faces_list,\
        obj_verts_list, obj_faces_list, actual_len_list
    
    def convert_hand_foot_jpos_to_data_input(self, hand_foot_jpos, val_data_dict):
        # hand_foot_jpos: BS X T X 2 X 3
        num_joints = hand_foot_jpos.shape[2]

        bs, num_steps, _, _ = hand_foot_jpos.shape 
       
        data_input = torch.zeros(bs, num_steps, 24*3+22*6).to(hand_foot_jpos.device) 

        lhand_idx = 22 
        rhand_idx = 23 
       
        lfoot_idx = 10
        rfoot_idx = 11 
    
        data_input[:, :, lhand_idx*3:lhand_idx*3+3] = hand_foot_jpos[:, :, 0, :] 
        data_input[:, :, rhand_idx*3:rhand_idx*3+3] = hand_foot_jpos[:, :, 1, :] 

        if num_joints > 2:
            data_input[:, :, lfoot_idx*3:lfoot_idx*3+3] = hand_foot_jpos[:, :, 2, :] 
            data_input[:, :, rfoot_idx*3:rfoot_idx*3+3] = hand_foot_jpos[:, :, 3, :]  

        return data_input 

    def gen_fullbody_from_predicted_hand_foot(self, hand_foot_jpos, val_data_dict):
        # hand_foot_jpos: BS X T X 2 X 3
        bs = hand_foot_jpos.shape[0]
        num_steps = hand_foot_jpos.shape[1] 
        hand_foot_jpos = hand_foot_jpos.reshape(bs, num_steps, -1, 3) 
        
        with torch.no_grad():
            val_data = self.convert_hand_foot_jpos_to_data_input(hand_foot_jpos, val_data_dict)
          
            cond_mask = None 

            left_joint_mask = self.prep_joint_condition_mask(val_data, joint_idx=22, pos_only=True)
            right_joint_mask = self.prep_joint_condition_mask(val_data, joint_idx=23, pos_only=True)

            if cond_mask is not None:
                cond_mask = cond_mask * left_joint_mask * right_joint_mask 
            else:
                cond_mask = left_joint_mask * right_joint_mask 

            # Generate padding mask 
            actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
            tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
            self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
            # BS X max_timesteps
           
            padding_mask = tmp_mask[:, None, :].to(val_data.device)

            all_res_list = self.ema.ema_model.sample(val_data, \
            cond_mask=cond_mask, padding_mask=padding_mask)

        return all_res_list

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 24 * 3 + 22 * 6 
    
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

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    repr_dim = 24 * 3 + 22 * 6 
   
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
        train_num_steps=400000,         # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False 
    )
   
    trainer.cond_sample_res()

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='', help='project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='', help='save to project/name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--fullbody_checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

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
    else:
        run_train(opt, device)
