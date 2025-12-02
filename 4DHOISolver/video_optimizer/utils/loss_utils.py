#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from sdf import *
import torch.nn as nn
import neural_renderer.neural_renderer as nr
import torchvision.transforms.functional as TF
from pytorch3d.ops import knn_points
from PIL import Image
import os
import cv2 
from .image_utils import process_frame2square, process_frame2square_mask
from scipy.ndimage.morphology import distance_transform_edt
import time
import open3d as o3d
import matplotlib.pyplot as plt
import json


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def chamfer_distance(pcl1, pcl2):
    dist1, idx1, _ = knn_points(pcl1, pcl2, K=1)  # KNN for A -> B
    dist2, idx2, _ = knn_points(pcl2, pcl1, K=1)  # KNN for B -> A
    return dist1, dist2

def normalize(x:torch.Tensor):
    return (x- x.min()) / (x.max() - x.min())

def contact_compute(cf_distance,opacity,distance_threshold=0.001,opacity_threshold=0.05):
    cf_distance=cf_distance**2
    # print(cf_distance.shape)
    h_distance=cf_distance[:,:10475]
    o_distance=cf_distance[:,10475:]
    distance_score_h=normalize(h_distance)
    distance_score_o=normalize(o_distance)
    distance_score=torch.cat((distance_score_h,distance_score_o),dim=1)
    min_distance=distance_score.min(dim=1)[0]
    distance_threshold=distance_threshold*2


    opacity_h=opacity[:10475]
    opacity_o=opacity[10475:]
    opacity_score_h=normalize(opacity_h)
    opacity_score_o=normalize(opacity_o)
    opacity_score=torch.cat((opacity_score_h,opacity_score_o),dim=0)

    #save opacity
    # opacity_save=opacity_score.cpu().detach().numpy()[:10475]
    # np.save('opacity_save.npy',opacity_save)

    contact_region=(distance_score<distance_threshold) & (opacity_score<opacity_threshold)
    contact_region=contact_region[0]
    # contact_region=(opacity_score>=opacity_threshold)
    # print(contact_region.shape)
    return contact_region


def calculate_chamfer_distance(points_A, points_B):
    # Convert points to PyTorch tensors
    points_A_tensor = torch.tensor(points_A).to(torch.float32).unsqueeze(0)  # Shape [1, N, 3]
    points_B_tensor = torch.tensor(points_B).to(torch.float32).unsqueeze(0)  # Shape [1, M, 3]

    # Use pytorch3d's chamfer_distance function
    dist1, idx1, _ = knn_points(points_A_tensor, points_B_tensor, K=1)  # KNN for A -> B
    dist2, idx2, _ = knn_points(points_B_tensor, points_A_tensor, K=1)  # KNN for B -> A

    return idx1, idx2, dist1, dist2


def compute_vertex_normals(meshes):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 1],
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 2],
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 0],
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
            dim=1,
        ),
    )

    return torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)

class HOCollisionLoss(nn.Module):
# adapted from multiperson (links, multiperson.sdf.sdf_loss.py)

    def __init__(self, smpl_faces, grid_size=32, robustifier=None,):
        super().__init__()
        self.sdf = SDF()
        self.register_buffer('faces', torch.tensor(smpl_faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier


    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        # vertices: (n, 3)
        boxes = torch.zeros(2, 3, device=vertices.device)
        boxes[0, :] = vertices.min(dim=0)[0]
        boxes[1, :] = vertices.max(dim=0)[0]
        return boxes


    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # check x
        if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
            return False
        #check y
        if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
            return False
        #check z
        if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
            return False
        return True


    def forward(self, hoi_dict):
        # assume one person and one object
        # person_vertices: (n, 3), object_vertices: (m, 3)
        person_vertices, object_vertices = hoi_dict['smplx_v_centered'], hoi_dict['object_v_centered']
        object_vertices.retain_grad()
        b = person_vertices.shape[0]
        scale_factor = 0.2
        loss = torch.zeros(1).float().to(object_vertices.device)

        for b_idx in range(b):
            person_bbox = self.get_bounding_boxes(person_vertices[b_idx])
            object_bbox = self.get_bounding_boxes(object_vertices[b_idx])
            # print(person_bbox, object_bbox)
            if not self.check_overlap(person_bbox, object_bbox):
                return loss

            person_bbox_center = person_bbox.mean(dim=0).unsqueeze(0)
            person_bbox_scale = (1 + scale_factor) * 0.5 * (person_bbox[1] - person_bbox[0]).max()

            with torch.no_grad():
                person_vertices_centered = person_vertices[b_idx] - person_bbox_center
                person_vertices_centered = person_vertices_centered / person_bbox_scale
                assert(person_vertices_centered.min() >= -1)
                assert(person_vertices_centered.max() <= 1)
                phi = self.sdf(self.faces, person_vertices_centered.unsqueeze(0))
                assert(phi.min() >= 0)

            object_vertices_centered = (object_vertices[b_idx] - person_bbox_center) / person_bbox_scale
            object_vertices_grid = object_vertices_centered.view(1, -1, 1, 1, 3)
            phi_val = nn.functional.grid_sample(phi.unsqueeze(1), object_vertices_grid).view(-1)
            phi_val.retain_grad()
            cur_loss = phi_val
            if self.robustifier:
                frac = (cur_loss / self.robustifier) ** 2
                cur_loss = frac / (frac + 1)

            loss += cur_loss.sum()
            out_loss = loss.sum() / b
            # out_loss.backward()

            #print(object_vertices.grad, phi_val.grad, out_loss.grad,79879426951,torch.sum(object_vertices.grad))

        return out_loss
    



def compute_contact_loss(corresponding_points):
    body_points = corresponding_points['body_points']
    object_points = corresponding_points['object_points']
    
    if (object_points.shape[0] == 0 | body_points.shape[0] == 0):
        return torch.tensor(0.0, device=body_points.device)
    distances = torch.norm(body_points - object_points, dim=1)
    weights = torch.pow(distances + 0.1, 2)  
    weights = weights / weights.sum()
    weighted_loss = torch.sum(weights * distances**2)
    
    return weighted_loss

def compute_collision_loss(hverts, overts, hfaces, ofaces, h_weight=10.0, threshold=None):  
    hfaces = np.ascontiguousarray(hfaces, dtype=np.int64)
    ofaces = np.ascontiguousarray(ofaces, dtype=np.int64)
    
    hoi_dict = {  
        'smplx_v_centered': overts - torch.mean(overts, dim=1, keepdim=True),  
        'object_v_centered': hverts - torch.mean(overts, dim=1, keepdim=True),  
    }  
    h_in_o_collision_loss = HOCollisionLoss(ofaces).to(hverts.device)  
    h_in_o_loss_val = h_in_o_collision_loss(hoi_dict)  
    
    hoi_dict = {  
        'smplx_v_centered': hverts - torch.mean(hverts, dim=1, keepdim=True),  
        'object_v_centered': overts - torch.mean(hverts, dim=1, keepdim=True),   
    }  
    o_in_h_collision_loss = HOCollisionLoss(hfaces).to(hverts.device)  
    o_in_h_loss_val = o_in_h_collision_loss(hoi_dict)  
    
    out_loss = h_weight * h_in_o_loss_val + o_in_h_loss_val  
    
    if threshold is not None:  
        if out_loss < threshold:  
            return out_loss * 0.0  
    
    return out_loss  



def compute_mask_loss(width, height, video_dir, hverts, overts, hfaces, ofaces, mask_weight=1.5, edge_weight=1e-3, frame_idx=None):
    downsample_height = height // 4
    downsample_width = width // 4
    downsample_image_size = (downsample_height, downsample_width)

    output = torch.load(video_dir + "/motion/result.pt")
    K = output["K_fullimg"][0].cuda()
    scale_x = downsample_width / width
    scale_y = downsample_height / height
    K_nf = K.clone()
    R = torch.eye(3, dtype=torch.float32).unsqueeze(0).cuda()
    T = torch.zeros(3, dtype=torch.float32).unsqueeze(0).cuda()
    i = frame_idx 

    render_size = max(downsample_height, downsample_width)  # 安全起见，用 max
    renderer = nr.renderer.Renderer(
        image_size=render_size,
        K=K_nf.unsqueeze(0),
        R=R,
        t=T,
        orig_size=(width, height),  # 用 max(width, height) 来匹配 image_size
        anti_aliasing=True,
    )
    hfaces = torch.tensor(hfaces, dtype=torch.int64).cuda()
    ofaces = torch.tensor(ofaces, dtype=torch.int64).cuda()
    human_mask = renderer(hverts, hfaces.unsqueeze(0),  mode="silhouettes").squeeze()
    object_mask = renderer(overts, ofaces.unsqueeze(0),  mode="silhouettes").squeeze()
    human_mask = TF.resize(human_mask.unsqueeze(0), downsample_image_size, 
                        interpolation=TF.InterpolationMode.NEAREST)
    object_mask = TF.resize(object_mask.unsqueeze(0), downsample_image_size, 
                        interpolation=TF.InterpolationMode.NEAREST)
    """
    compute mask loss
    """
    if (i==0):
        mask = human_mask.squeeze().detach().cpu().numpy()  # shape: [H, W]
        mask = (mask * 255).astype('uint8')  # 转换为8位像素
        img = Image.fromarray(mask)
        img.save('resize_human_mask.png')
    if (i==0):
        mask = object_mask.squeeze().detach().cpu().numpy()  # shape: [H, W]
        mask = (mask * 255).astype('uint8')
        img = Image.fromarray(mask)
        img.save('resize_object_mask.png')
    gt_paths = {  
        'obj': os.path.join(video_dir, 'mask_dir', f"{str(i).zfill(5)}_mask.png"),  
        'human': os.path.join(video_dir, 'human_mask_dir', f"{str(i).zfill(5)}_mask.png")  
    }
    render_masks = {  
        'obj': object_mask,
        'human': human_mask
    }  
    cached_gt_masks = {}
    for key, path in gt_paths.items():  
        mask = Image.open(path).convert("L")
        gt_mask = process_frame2square_mask(np.array(mask, copy=True)[:,:,np.newaxis])
        gt_mask = torch.from_numpy(gt_mask).float().div(255).to(render_masks[key].device)[:,:,0]
        downsampled_gt = F.interpolate(  
            gt_mask.unsqueeze(0).unsqueeze(0),  
            size=render_masks[key].squeeze().shape,  
            mode='area'  
        ).squeeze()  
        cached_gt_masks[key] = downsampled_gt
    
    time_compute = time.time()
    h_gt_mask = cached_gt_masks['human'].unsqueeze(0)
    o_gt_mask = cached_gt_masks['obj'].unsqueeze(0)
    h_render_mask = render_masks['human']*(1 - o_gt_mask)
    o_render_mask = render_masks['obj']*(1 - h_gt_mask)
    power=0.5
    batch_size=1

    ## mask loss:
    # print("mask loss shape:", h_render_mask.shape, o_render_mask.shape, h_gt_mask.shape, o_gt_mask.shape)
    h_mask_loss= F.mse_loss(h_render_mask, h_gt_mask)
    o_mask_loss= F.mse_loss(o_render_mask, o_gt_mask)

    kernel_size = 7
    pool = torch.nn.MaxPool2d(
        kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)
    )
    h_edge_render=pool(h_render_mask)-h_render_mask
    o_edge_render=pool(o_render_mask)-o_render_mask
    h_edge_gt=pool(h_gt_mask)-h_gt_mask
    o_edge_gt=pool(o_gt_mask)-o_gt_mask
    h_edt = distance_transform_edt(1 - (h_edge_gt.detach().cpu().numpy() > 0)) ** (power * 2)
    o_edt = distance_transform_edt(1 - (o_edge_gt.detach().cpu().numpy() > 0)) ** (power * 2)
    # print("h_edt_max", h_edt.max(), "o_edt_max:", o_edt.max())
    h_edt = torch.from_numpy(h_edt).repeat(batch_size, 1, 1).float().cuda()
    o_edt = torch.from_numpy(o_edt).repeat(batch_size, 1, 1).float().cuda()

    edge_img = o_edge_render.squeeze(0).detach().cpu().numpy()
    edge_img_gt = o_edge_gt.squeeze(0).detach().cpu().numpy()
    h_edge_loss=torch.sum(h_edge_render * h_edt, dim=(1, 2))
    o_edge_loss=torch.sum(o_edge_render * o_edt, dim=(1, 2))
    total_loss = 0.5*(edge_weight * h_edge_loss + mask_weight * h_mask_loss)+ 1.5*(edge_weight * o_edge_loss + mask_weight * o_mask_loss)
    return total_loss


def projected_2D_loss(frame_idx=None):
    if frame_idx is None:
        frame_idx = self.current_frame
    corresponding_points = self.get_corresponding_point(frame_idx)
    body_points = corresponding_points['body_points']
    object_points = corresponding_points['object_points']
    image_size = (self.image_size, self.image_size)
    output = torch.load(self.video_dir + "/motion/result.pt")
    K = output["K_fullimg"][0].cuda()
    K_nf = K.clone()
    K_nf[0,0] = K[0, 0] * (image_size[0] / image_size[1])
    K_nf[0,2] = K[0, 2] * (image_size[0] / image_size[1])
    fx, fy = K_nf[0, 0], K_nf[1, 1]
    cx, cy = K_nf[0, 2], K_nf[1, 2]
    weighted_loss = torch.tensor(0.0, device='cuda')

    if body_points.shape[0] > 0 and object_points.shape[0] > 0:
        body_2d = torch.zeros((body_points.shape[0], 2), device=body_points.device)
        body_2d[:, 0] = fx * body_points[:, 0] / body_points[:, 2] + cx
        body_2d[:, 1] = fy * body_points[:, 1] / body_points[:, 2] + cy

        object_2d = torch.zeros((object_points.shape[0], 2), device=object_points.device)
        object_2d[:, 0] = fx * object_points[:, 0] / object_points[:, 2] + cx
        object_2d[:, 1] = fy * object_points[:, 1] / object_points[:, 2] + cy
        distances_2d = torch.norm(body_2d - object_2d, dim=1)
        weights = torch.pow(distances_2d + 0.1, 2) 
        weights = weights / weights.sum()
        weighted_loss = torch.sum(weights * distances_2d**2)
    pairs_2d_loss = torch.tensor(0.0, device='cuda')
    if len(self.pairs_2d[frame_idx]) > 0:
        pairs = self.pairs_2d[frame_idx]
        object_vertices = self.get_object_points(frame_idx)
        obj_indices = []
        target_2d_points = []
        
        for pair in pairs:
            obj_idx = pair[0]
            point_2d = pair[1]
            obj_indices.append(obj_idx)
            target_2d_points.append(point_2d)
        
        if len(obj_indices) > 0:
            selected_obj_points = object_vertices[obj_indices]
            selected_obj_2d = torch.zeros((len(obj_indices), 2), device='cuda')
            selected_obj_2d[:, 0] = fx * selected_obj_points[:, 0] / selected_obj_points[:, 2] + cx
            selected_obj_2d[:, 1] = fy * selected_obj_points[:, 1] / selected_obj_points[:, 2] + cy
            target_2d_tensor = torch.tensor(target_2d_points, dtype=torch.float32, device='cuda')
            pair_distances_2d = torch.norm(selected_obj_2d - target_2d_tensor, dim=1)
            pair_weights = torch.pow(pair_distances_2d + 0.1, 2)
            if pair_weights.sum() > 0:
                pair_weights = pair_weights / pair_weights.sum()
            pairs_2d_loss = torch.sum(pair_weights * pair_distances_2d**2)
    total_loss = weighted_loss + 1.5 * pairs_2d_loss
    
    return total_loss
def compute_temporal_pose_smoothness( pose_weight=1.0, shape_weight=1.0, H=5, F=5, frame_idx=None):  
    if frame_idx is None:
        frame_idx = self.current_frame
    
    loss = torch.tensor(0.0, device=self.body_pose_params[0].device)
    if frame_idx == 0:
        return loss
    left = max(0, frame_idx - H)
    right = min(self.seq_length - 1, frame_idx + F)
    i = frame_idx
    pose_seq = torch.stack(self.body_pose_params[left:right], dim=0)
    shape_seq = torch.stack(self.shape_params[left:right], dim=0)
    pose_diff = pose_seq[1:] - pose_seq[:-1]
    shape_diff = shape_seq[1:] - shape_seq[:-1]
    loss = pose_weight*torch.sum(torch.norm(pose_diff, dim=1)**2) + shape_weight*torch.sum(torch.norm(shape_diff, dim=1)**2)  
    return loss
def compute_temporal_object_smoothness(lambda_rot=1.0, lambda_trans=1.0, H=5, F=5, frame_idx=None):  
    if frame_idx is None:
        frame_idx = self.current_frame 
    all_loss = torch.tensor(0.0, device=self.obj_x_params[0].device)
    if self.is_static_object:
        return all_loss
        
    i = frame_idx
    left = max(0, i - H)
    right = min(self.seq_length - 1, i + F)
    overts_list = []
    for idx in range(left, right):
        overts = torch.tensor(np.asarray(self.sampled_obj_meshes[idx].vertices), dtype=torch.float32).cuda()
        transforms = self.get_object_transform(idx).cuda().float()
        overts = torch.mm(overts, transforms).unsqueeze(0)
        overts = self.obj_transl_params[idx] + overts
        overts_list.append(overts.squeeze(0))
    overts_seq = torch.stack(overts_list, dim=0)
    verts_diff = overts_seq[1:] - overts_seq[:-1]
    all_loss += torch.sum(torch.norm(verts_diff, dim=1)**2)
    return all_loss 

def joint_mask_parameters(smpl_model, optimizer, frame_idx, body_kp_name, joint_sim):
    with torch.no_grad():
        body_pose_param, left_hand_param, right_hand_param = None, None, None
        for group in optimizer.param_groups:
            group_name = group.get('name', '')
            if group_name == f'pose_{frame_idx}':
                body_pose_param = group['params'][0]
            elif group_name == f'left_hand_{frame_idx}':
                left_hand_param = group['params'][0]
            elif group_name == f'right_hand_{frame_idx}':
                right_hand_param = group['params'][0]
        device = body_pose_param.device
            
        lbs_weights = smpl_model.lbs_weights.to(device)
        gradients_to_zero = set()
        for kp in body_kp_name:
            joint_name = search_joint_name(kp, joint_sim)
            joint_idx = get_smplx_index(joint_name)
            # print(f"[Debug Frame {frame_idx}] For vertex {vertex_index}, most influential joint_idx: {joint_idx}")
            param_to_modify = None
            start_idx, end_idx = None, None
            if 1 <= joint_idx <= 21:
                param_to_modify = body_pose_param
                pose_idx = joint_idx - 1
                start_idx, end_idx = pose_idx * 3, pose_idx * 3 + 3

            elif 25 <= joint_idx <= 39:
                param_to_modify = left_hand_param
                pose_idx = joint_idx - 25
                start_idx, end_idx = pose_idx * 3, pose_idx * 3 + 3
                
            elif 40 <= joint_idx <= 54:
                param_to_modify = right_hand_param
                pose_idx = joint_idx - 40
                start_idx, end_idx = pose_idx * 3, pose_idx * 3 + 3

            if param_to_modify is not None and param_to_modify.grad is not None:
                gradients_to_zero.add((param_to_modify, start_idx, end_idx))
        slices_to_keep_by_param = {}
        for param, start, end in gradients_to_zero:
            param_id = id(param) # Use the memory address as a unique ID for the tensor
            if param_id not in slices_to_keep_by_param:
                slices_to_keep_by_param[param_id] = {'param': param, 'slices': []}
            slices_to_keep_by_param[param_id]['slices'].append((start, end))
        relevant_params = [body_pose_param, left_hand_param, right_hand_param]
        relevant_params = [p for p in relevant_params if p is not None]

        for param in relevant_params:
            if param.grad is None:
                continue
            param_id = id(param)
            if param_id in slices_to_keep_by_param:
                # If there are slices to keep, clone, zero, and restore them
                data = slices_to_keep_by_param[param_id]
                slices = data['slices']
                grad_copy = param.grad.clone()
                param.grad.zero_()
                for start, end in slices:
                    param.grad[start:end] += grad_copy[start:end]
            else:
                # If no slices to keep, zero out the entire gradient
                param.grad.zero_()


def visualize_vertex_gradients(smpl_model, optimizer, frame_idx, output_dir, body_verts_getter):
    with torch.no_grad():
        body_pose_param, left_hand_param, right_hand_param = None, None, None
        for group in optimizer.param_groups:
            group_name = group.get('name', '')
            if group_name == f'pose_{frame_idx}':
                body_pose_param = group['params'][0]
            elif group_name == f'left_hand_{frame_idx}':
                left_hand_param = group['params'][0]
            elif group_name == f'right_hand_{frame_idx}':
                right_hand_param = group['params'][0]

        if body_pose_param is None:
            print(f"[Warning] Could not find pose for frame {frame_idx} in optimizer for gradient visualization.")
            return

        device = body_pose_param.device
        num_joints = smpl_model.lbs_weights.shape[1]
        joint_grad_mags = torch.zeros(num_joints, device=device)

        if body_pose_param.grad is not None:
            body_grads = body_pose_param.grad.view(-1, 3)
            # SMPL-X body pose corresponds to joints 1-21
            joint_grad_mags[1:1+body_grads.shape[0]] = torch.norm(body_grads, dim=1)

        if left_hand_param is not None and left_hand_param.grad is not None:
            left_hand_grads = left_hand_param.grad.view(-1, 3)
            # SMPL-X left hand pose corresponds to joints 25-39
            joint_grad_mags[25:25+left_hand_grads.shape[0]] = torch.norm(left_hand_grads, dim=1)

        if right_hand_param is not None and right_hand_param.grad is not None:
            right_hand_grads = right_hand_param.grad.view(-1, 3)
            # SMPL-X right hand pose corresponds to joints 40-54
            joint_grad_mags[40:40+right_hand_grads.shape[0]] = torch.norm(right_hand_grads, dim=1)
        lbs_weights = smpl_model.lbs_weights.to(device)
        vertex_grad_mags = torch.einsum('vj,j->v', lbs_weights, joint_grad_mags)
        if vertex_grad_mags.max() > 1e-9:
            normalized_mags = vertex_grad_mags / vertex_grad_mags.max()
        else:
            normalized_mags = torch.zeros_like(vertex_grad_mags)

        colormap = plt.get_cmap('jet')
        vertex_colors = colormap(normalized_mags.cpu().numpy())[:, :3]
        body_vertices = body_verts_getter(frame_idx).cpu().numpy()
        body_faces = smpl_model.faces
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(body_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(body_faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        mesh.compute_vertex_normals()

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"grad_viz_frame_{frame_idx}.ply")
        o3d.io.write_triangle_mesh(output_path, mesh)
def create_reverse_joint_mapping(joint_sim_path):
    with open(joint_sim_path, 'r') as f:
        joint_sim = json.load(f)
    
    reverse_mapping = {}
    for smpl_joint, body_kp_list in joint_sim.items():
        for body_kp in body_kp_list:
            reverse_mapping[body_kp] = smpl_joint
    
    return reverse_mapping

def search_joint_name(kp, joint_sim):
    for smpl_joint, body_kp_list in joint_sim.items():
        if kp in body_kp_list:
            return smpl_joint
    return None

def get_smplx_index(smpl_joint_name):
    JOINT_NAMES = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "left_eye_smplhf",
        "right_eye_smplhf",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
        "nose",
        "right_eye",
        "left_eye",
        "right_ear",
        "left_ear",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
        "left_thumb",
        "left_index",
        "left_middle",
        "left_ring",
        "left_pinky",
        "right_thumb",
        "right_index",
        "right_middle",
        "right_ring",
        "right_pinky",
        "right_eye_brow1",
        "right_eye_brow2",
        "right_eye_brow3",
        "right_eye_brow4",
        "right_eye_brow5",
        "left_eye_brow5",
        "left_eye_brow4",
        "left_eye_brow3",
        "left_eye_brow2",
        "left_eye_brow1",
        "nose1",
        "nose2",
        "nose3",
        "nose4",
        "right_nose_2",
        "right_nose_1",
        "nose_middle",
        "left_nose_1",
        "left_nose_2",
        "right_eye1",
        "right_eye2",
        "right_eye3",
        "right_eye4",
        "right_eye5",
        "right_eye6",
        "left_eye4",
        "left_eye3",
        "left_eye2",
        "left_eye1",
        "left_eye6",
        "left_eye5",
        "right_mouth_1",
        "right_mouth_2",
        "right_mouth_3",
        "mouth_top",
        "left_mouth_3",
        "left_mouth_2",
        "left_mouth_1",
        "left_mouth_5",  # 59 in OpenPose output
        "left_mouth_4",  # 58 in OpenPose output
        "mouth_bottom",
        "right_mouth_4",
        "right_mouth_5",
        "right_lip_1",
        "right_lip_2",
        "lip_top",
        "left_lip_2",
        "left_lip_1",
        "left_lip_3",
        "lip_bottom",
        "right_lip_3",
        # Face contour
        "right_contour_1",
        "right_contour_2",
        "right_contour_3",
        "right_contour_4",
        "right_contour_5",
        "right_contour_6",
        "right_contour_7",
        "right_contour_8",
        "contour_middle",
        "left_contour_8",
        "left_contour_7",
        "left_contour_6",
        "left_contour_5",
        "left_contour_4",
        "left_contour_3",
        "left_contour_2",
        "left_contour_1",
    ]
    joint_index_map = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    
    return joint_index_map.get(smpl_joint_name, -1)
