import numpy as np  
import matplotlib.pyplot as plt  
import math
import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch  
import torch.optim as optim  
import torch.nn.functional as F   
import smplx  
import json  
import cv2  
import open3d as o3d  
from torch import nn  
import tqdm
from .utils.loss_utils import (
    HOCollisionLoss,
    compute_contact_loss,
    compute_collision_loss,  
    compute_mask_loss,
    joint_mask_parameters,
    visualize_vertex_gradients,
)  
from PIL import Image    
from tqdm import tqdm  
import torchvision.transforms.functional as TF
import time
import shutil
from .utils.camera_utils import transform_to_global, inverse_transform_to_incam
from .utils.hoi_utils import get_all_body_joints
from copy import deepcopy
from .utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
def resource_path(relative_path):
    try:
                               
        base_path = sys._MEIPASS
    except Exception:
                  
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

J_regressor = torch.load(resource_path("video_optimizer/J_regressor.pt")).float().cuda()
joint_sim = json.load(open(resource_path("video_optimizer/data/joint_sim.json")))

def load_downsampling_mapping(filepath):
    data = np.load(filepath)
    from scipy.sparse import csr_matrix
    D = csr_matrix((data['D_data'], data['D_indices'], data['D_indptr']), 
                   shape=data['D_shape'])
    faces_ds = data['faces_ds']
    print(f"Downsampling mapping loaded from {filepath}")
    return D, faces_ds

downsampling_file_path = resource_path("video_optimizer/smplx_downsampling_1000.npz")
D, faces_ds = load_downsampling_mapping(downsampling_file_path)
D_torch = torch.tensor(D.toarray(), dtype=torch.float32, device="cuda")
def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    R = R.to(dtype=torch.float32)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    q = q / (torch.linalg.norm(q) + 1e-8)
    return q
def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / (torch.linalg.norm(q) + 1e-8)
    w, x, y, z = q[0], q[1], q[2], q[3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)]),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)]),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)])
    ])
    return R.to(dtype=torch.float32)


def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    q0 = q0 / (torch.linalg.norm(q0) + 1e-8)
    q1 = q1 / (torch.linalg.norm(q1) + 1e-8)

    dot = torch.dot(q0, q1)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.where(dot < 0.0, -dot, dot)

    DOT_THRESHOLD = 0.9995
    if float(dot) > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / (torch.linalg.norm(result) + 1e-8)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / (sin_theta_0 + 1e-8)
    s1 = sin_theta / (sin_theta_0 + 1e-8)
    return s0 * q0 + s1 * q1
class VideoBodyObjectOptimizer:  
    def __init__(self,   
                 body_params,
                 global_body_params,
                 hand_params,  
                 object_points_idx,   
                 body_points_idx, 
                 body_kp_name,
                 pairs_2d, 
                 object_meshes, 
                 sampled_obj_meshes,
                 centers_depth,
                 icp_transform_matrix,
                 smpl_model,
                 start_frame,
                 end_frame,
                 video_dir,  
                 lr=0.1,
                 is_static_object=False,
                 best_frame=None):  
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.seq_length = end_frame - start_frame
        self.smpl_model = smpl_model  
        self.pairs_2d = pairs_2d
        self.body_params_sequence = body_params  
        self.global_body_params = global_body_params
        self.hand_poses = hand_params
        self.object_meshes = object_meshes
        self.sampled_obj_meshes = sampled_obj_meshes
        self.centers_depth = centers_depth
        self.icp_transform_matrix = icp_transform_matrix
        self.video_dir = video_dir
        self.is_static_object = is_static_object
        cap = cv2.VideoCapture(os.path.join(video_dir, "video.mp4"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = width
        self.height = height
        self.image_size = max(width, height)
        self.lr = lr
        
                                
        self.body_pose_params = []  
        self.shape_params = []
        self.left_hand_params = []
        self.right_hand_params = []
        self.global_orient = []
        self.transl = []
        self.obj_x_params = []  
        self.obj_y_params = []  
        self.obj_z_params = []  
        self.obj_transl_params = []  
        self.obj_transl_limit = torch.tensor([0.1, 0.1, 0.1]).cuda()

        self.icp_obj_R = []
        self.icp_obj_transl = []

        self.R_incam_static = []
        self.T_incam_static = []
        self.R_total_frames = None
        self.T_total_frames = None
        self.frames_optimized = []
        self.pre_contact_best_frame = None
        self.R_incam_pre_contact = {}
        self.T_incam_pre_contact = {}
        self.incam_pelvis, self.global_pelvis, self.incam_orient_o, self.global_orient_o, self.incam_transl_o, self.global_transl_o = get_all_body_joints(
            self.body_params_sequence, 
            self.global_body_params, 
            self.smpl_model, 
            self.start_frame, 
            self.end_frame
        )

        for i in range(self.start_frame, self.end_frame):
            self.body_pose_params.append(nn.Parameter(self.body_params_sequence["body_pose"][i].cuda(), requires_grad=True)) 
            self.shape_params.append(nn.Parameter(self.body_params_sequence['betas'][i].cuda(), requires_grad=True))  
            handpose=self.hand_poses[str(i)]
                           
            left_hand_pose = torch.from_numpy(np.asarray(handpose['left_hand']).reshape(-1,3)).float().cuda()
            right_hand_pose = torch.from_numpy(np.asarray(handpose['right_hand']).reshape(-1,3)).float().cuda()
            self.left_hand_params.append(nn.Parameter(left_hand_pose, requires_grad=True))
            self.right_hand_params.append(nn.Parameter(right_hand_pose, requires_grad=True))
            self.global_orient.append(self.body_params_sequence['global_orient'][i].cuda())
            self.transl.append(self.body_params_sequence['transl'][i].cuda())

            trans_mat = self.icp_transform_matrix[i - self.start_frame]
            R_mat = trans_mat[:3, :3]
            transl_vec = trans_mat[:3, 3]
            
            self.icp_obj_R.append(torch.from_numpy(R_mat).float().cuda())
            self.icp_obj_transl.append(torch.from_numpy(transl_vec).float().cuda())

            self.obj_x_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_y_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_z_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_transl_params.append(nn.Parameter(torch.zeros(3, dtype=torch.float32).cuda(), requires_grad=True))

        self.body_points_idx = body_points_idx
        self.object_points_idx = object_points_idx  
        self.body_kp_name = body_kp_name
        if best_frame is not None:
            self.best_frame = best_frame
        self.mask = None  
        self.optimizer = None  
        self.current_frame = 0  
    
    def training_setup(self):
        params_list = []
        for i in range(self.seq_length):
            frame_params = [
                {'params': [self.body_pose_params[i]], 'lr': 0.001, 'name': f'pose_{i}'},
                {'params': [self.shape_params[i]], 'lr': 0.001, 'name': f'shape_{i}'},
                {'params': [self.left_hand_params[i]], 'lr': 0.003, 'name': f'left_hand_{i}'},
                {'params': [self.right_hand_params[i]], 'lr': 0.003, 'name': f'right_hand_{i}'},
                {'params': [self.obj_x_params[i]], 'lr': 0.002, 'name': f'x_angle_{i}'},
                {'params': [self.obj_y_params[i]], 'lr': 0.002, 'name': f'y_angle_{i}'},
                {'params': [self.obj_z_params[i]], 'lr': 0.002, 'name': f'z_angle_{i}'},
                {'params': [self.obj_transl_params[i]], 'lr': 0.001, 'name': f'transl_{i}'},
            ]
            
            params_list.extend(frame_params)                                                                
        self.optimizer = optim.Adam(params_list, lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def get_body_points(self, frame_idx=None, sampled=False):  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        body_pose = self.body_pose_params[frame_idx].reshape(1, -1).cuda()  
        body_pose_save = body_pose.clone().detach().cpu().numpy()
        shape = self.shape_params[frame_idx].reshape(1, -1).cuda()  
        global_orient = self.global_orient[frame_idx].reshape(1, 3).cuda()
        left_hand_pose = self.left_hand_params[frame_idx].reshape(1, -1).cuda()
        right_hand_pose = self.right_hand_params[frame_idx].reshape(1, -1).cuda()
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1).cuda()
        transl = self.transl[frame_idx].reshape(1, -1).cuda()
        output = self.smpl_model(betas=shape,   
                                body_pose=body_pose,
                                left_hand_pose=left_hand_pose,   
                                right_hand_pose=right_hand_pose,   
                                jaw_pose=zero_pose,   
                                leye_pose=zero_pose,
                                reye_pose=zero_pose,
                                global_orient=global_orient,
                                expression=torch.zeros((1, 10)).float().cuda(),
                                transl=transl)
                                                        
        xyz = output.vertices[0]
        if sampled:
            xyz = torch.einsum('vw,wc->vc', D_torch, xyz)
        return xyz
    def get_body_faces(self, sampled=False):  
        body_faces = self.smpl_model.faces  
        if sampled:
            body_faces = faces_ds
        return body_faces  
    
    def get_object_faces(self, frame_idx=None, sampled=False):
        if frame_idx is None:
            frame_idx = self.current_frame
        if sampled:
            object_mesh = self.sampled_obj_meshes[frame_idx]
        else:
            object_mesh = self.object_meshes[frame_idx]
        object_faces = object_mesh.triangles
        return np.asarray(object_faces).astype(np.int64)
    def get_object_points(self, frame_idx=None, sampled=False):
        if frame_idx is None:
            frame_idx = self.current_frame
        if sampled:
            object_mesh = self.sampled_obj_meshes[frame_idx]
        else:
            object_mesh = self.object_meshes[frame_idx]
        R_final, t_final = self.get_object_params(frame_idx)
        object_points = torch.tensor(np.asarray(object_mesh.vertices), 
                                   dtype=torch.float32, device=R_final.device)
        object_points = torch.mm(object_points, R_final.T) + t_final
        return object_points    
    def get_object_transform(self, frame_idx=None):  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        
        x_angle = torch.deg2rad(self.obj_x_params[frame_idx])  
        y_angle = torch.deg2rad(self.obj_y_params[frame_idx])  
        z_angle = torch.deg2rad(self.obj_z_params[frame_idx]) 
        
        RX = torch.stack([  
            torch.tensor([1.0, 0.0, 0.0], device=x_angle.device, dtype=torch.float32),  
            torch.stack([torch.tensor(0.0, device=x_angle.device, dtype=torch.float32), torch.cos(x_angle), -torch.sin(x_angle)]),  
            torch.stack([torch.tensor(0.0, device=x_angle.device, dtype=torch.float32), torch.sin(x_angle), torch.cos(x_angle)])  
        ])  
        
        RY = torch.stack([  
            torch.stack([torch.cos(y_angle), torch.tensor(0.0, device=y_angle.device, dtype=torch.float32), torch.sin(y_angle)]),  
            torch.tensor([0.0, 1.0, 0.0], device=y_angle.device, dtype=torch.float32),  
            torch.stack([-torch.sin(y_angle), torch.tensor(0.0, device=y_angle.device, dtype=torch.float32), torch.cos(y_angle)])  
        ])  
        
        RZ = torch.stack([  
            torch.stack([torch.cos(z_angle), -torch.sin(z_angle), torch.tensor(0.0, device=z_angle.device, dtype=torch.float32)]),  
            torch.stack([torch.sin(z_angle), torch.cos(z_angle), torch.tensor(0.0, device=z_angle.device, dtype=torch.float32)]),  
            torch.tensor([0.0, 0.0, 1.0], device=z_angle.device, dtype=torch.float32)  
        ])    
        R = torch.mm(torch.mm(RZ, RY), RX)  
        return R  
    
    def get_optimized_parameters(self):
        human_params = {
            'body_pose': [],
            'betas': [],
            'global_orient': [],
            'transl': [],
            'left_hand_pose': [],
            'right_hand_pose': [],
        }
        
        object_params = {
            'poses': [], 
            'centers': [] 
        }
        for frame_idx in range(self.seq_length):
                  
            body_pose = self.body_pose_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            shape = self.shape_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            global_orient = self.global_orient[frame_idx].reshape(1, 3).cpu().detach().numpy()
            left_hand_pose = self.left_hand_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            right_hand_pose = self.right_hand_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            transl = self.transl[frame_idx].reshape(1, -1).cpu().detach().numpy()
            
            human_params['body_pose'].append(body_pose.tolist())
            human_params['betas'].append(shape.tolist())
            human_params['global_orient'].append(global_orient.tolist())
            human_params['transl'].append(transl.tolist())
            human_params['left_hand_pose'].append(left_hand_pose.tolist())
            human_params['right_hand_pose'].append(right_hand_pose.tolist())
            
                                      
            if self.is_static_object:
                obj_frame_idx = self.best_frame
            else:
                obj_frame_idx = frame_idx
                
            R_final, t_final = self.get_object_params(obj_frame_idx)
            
            object_params['poses'].append(R_final.cpu().detach().numpy().tolist())
            object_params['centers'].append(t_final.cpu().detach().numpy().tolist())
        return {
            'human_params': human_params,
            'object_params': object_params,
            'frame_range': {
                'start_frame': self.start_frame,
                'end_frame': self.end_frame
            }
        }
    def get_corresponding_point(self, frame_idx=None):  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        object_points_idx =  self.object_points_idx[frame_idx]
        body_points_idx = np.asarray(self.body_points_idx[frame_idx])
        interacting_indices = object_points_idx[:, 1] != 0  
        interacting_body_indices = body_points_idx[interacting_indices]  
        body_points = self.get_body_points(frame_idx)[interacting_body_indices]  
        object_points = self.get_object_points(frame_idx)
        obj_index = object_points_idx[interacting_indices][:, 0]  
        interactiong_obj = object_points[obj_index]
        corresponding_points = {  
            'body_points': body_points,  
            'object_points': interactiong_obj 
        }  
        return corresponding_points  
    def _has_contact(self, frame_idx):
        object_points_idx = self.object_points_idx[frame_idx]
        if object_points_idx is None:
            return False
        interacting_indices = object_points_idx[:, 1] != 0
        return bool(np.any(interacting_indices))
    def optimize(self,   
                steps=100,   
                print_every=10,   
                contact_weight=50,   
                collision_weight=8, 
                mask_weight=0.05,
                project_2d_weight=3.5e-3,
                optimize_interval=3,
                smoothing_alpha=0.25,
                smoothing_beta=0.25,
                smoothing_window=7,
                smoothing_passes=2,
                smoothing_method='ema_box',
                smoothing_cutoff=0.08,
                smoothing_order=4,
                smoothing_fs=1.0):
        self.training_setup()   
                  
        self.leave_hand = False
        self.leave_hand_begin, self.leave_hand_end = None, None
        self.leave_hand_pairs = []

                               
        pre_contact_mode = False
        self.pre_contact_best_frame = None
        if not self.is_static_object and not self._has_contact(0):
            for idx in range(self.seq_length):
                if self._has_contact(idx):
                    self.pre_contact_best_frame = idx
                    pre_contact_mode = True
                    break
        if self.is_static_object and getattr(self, 'best_frame', None) is not None:
            frames_to_optimize = [self.best_frame]
        else:
            frames_to_optimize = list(range(0, self.seq_length, optimize_interval))
            if frames_to_optimize[-1] != self.seq_length:
                frames_to_optimize.append(self.seq_length-1)             

        if pre_contact_mode:
            if self.pre_contact_best_frame is None:
                frames_to_optimize = []
            else:
                frames_to_optimize = [f for f in frames_to_optimize if f >= self.pre_contact_best_frame]
                if self.pre_contact_best_frame not in frames_to_optimize:
                    frames_to_optimize.append(self.pre_contact_best_frame)
                frames_to_optimize = sorted(set(frames_to_optimize))
                                                                   
        self.frames_optimized = frames_to_optimize
        for frame_idx in tqdm(frames_to_optimize):
            self.current_frame = frame_idx
            corresponding_points= self.get_corresponding_point(frame_idx)
            if corresponding_points['body_points'].numel() == 0 and not self.leave_hand:
                self.leave_hand = True
                self.leave_hand_begin = max(frame_idx - optimize_interval, 0)
                         
            if corresponding_points['body_points'].numel() > 0 and self.leave_hand:
                self.leave_hand_end = frame_idx
                self.leave_hand_pairs.append((self.leave_hand_begin, self.leave_hand_end))
                self.leave_hand = False
                self.leave_hand_begin, self.leave_hand_end = None, None
            for step in range(steps):
                self.optimizer.zero_grad() if self.optimizer is not None else None
                corresponding_points= self.get_corresponding_point(frame_idx)
                contact_loss = compute_contact_loss(corresponding_points)   
                hverts = self.get_body_points(frame_idx, sampled=True).unsqueeze(0)
                overts = self.get_object_points(frame_idx, sampled=True).unsqueeze(0)
                hfaces = self.get_body_faces(sampled=True)
                ofaces = self.get_object_faces(frame_idx, sampled=True)
                collision_loss = compute_collision_loss(hverts, overts, hfaces, ofaces, h_weight=10.0, threshold=0) + 1e-5
                mask_loss= compute_mask_loss(self.width, self.height, self.video_dir, hverts, overts, hfaces, ofaces, mask_weight=1.5, edge_weight=1e-3, frame_idx=frame_idx + self.start_frame)
        
                         
        
                loss = ( contact_weight * contact_loss
                        + collision_weight * collision_loss
                        + mask_weight * mask_loss
                        )

                param_idx = self.best_frame if self.is_static_object else frame_idx
                if torch.any(torch.abs(self.obj_transl_params[param_idx]) > self.obj_transl_limit):
                    limit_mask = torch.abs(self.obj_transl_params[param_idx]) > self.obj_transl_limit
                    total_loss = loss + 1e6*F.mse_loss(self.obj_transl_params[param_idx][limit_mask], self.obj_transl_limit[limit_mask])
                else:
                    total_loss = loss
                if total_loss.requires_grad and total_loss.grad_fn is not None:
                    total_loss.backward()
                joint_mask_parameters(self.smpl_model, self.optimizer, frame_idx, self.body_kp_name, joint_sim)
                self.optimizer.step()     
                if step % print_every == 0:
                    tqdm.write(f"Frame {frame_idx}, Step {step}: Loss = {loss.item():.4f}, "
                            f"Contact = {contact_loss.item():.4f}, "
                                                                                
                            f"Collision = {collision_loss.item():.4f}, "
                            f"Mask = {mask_loss.item():.4f}"
                                )

        if self.is_static_object:
            with torch.no_grad():
                R_best_incam, T_best_incam  = self.get_object_params(self.best_frame)
                R_best_global, T_best_global = transform_to_global(
                    R_best_incam, 
                    T_best_incam, 
                    self.incam_orient_o[self.best_frame], 
                    self.global_orient_o[self.best_frame], 
                    self.incam_pelvis[self.best_frame], 
                    self.global_pelvis[self.best_frame],
                )
                for frame_idx in range(self.seq_length):
                    if frame_idx == self.best_frame:
                        self.R_incam_static.append(R_best_incam)
                        self.T_incam_static.append(T_best_incam)
                    else:
                        R_incam_final, T_incam_final = inverse_transform_to_incam(
                            R_best_global, 
                            T_best_global,
                            self.incam_orient_o[frame_idx], 
                            self.global_orient_o[frame_idx], 
                            self.incam_pelvis[frame_idx], 
                            self.global_pelvis[frame_idx],
                        ) 
                        self.R_incam_static.append(R_incam_final)
                        self.T_incam_static.append(T_incam_final)            
                                                                                                
        if pre_contact_mode and self.pre_contact_best_frame is not None:
            with torch.no_grad():
                                                                        
                R_best_incam, T_best_incam  = self.get_object_params(self.pre_contact_best_frame)
                R_best_global, T_best_global = transform_to_global(
                    R_best_incam,
                    T_best_incam,
                    self.incam_orient_o[self.pre_contact_best_frame],
                    self.global_orient_o[self.pre_contact_best_frame],
                    self.incam_pelvis[self.pre_contact_best_frame],
                    self.global_pelvis[self.pre_contact_best_frame],
                )
                for frame_idx in range(self.pre_contact_best_frame):
                    R_incam_final, T_incam_final = inverse_transform_to_incam(
                        R_best_global,
                        T_best_global,
                        self.incam_orient_o[frame_idx],
                        self.global_orient_o[frame_idx],
                        self.incam_pelvis[frame_idx],
                        self.global_pelvis[frame_idx],
                    )
                    self.R_incam_pre_contact[frame_idx] = R_incam_final
                    self.T_incam_pre_contact[frame_idx] = T_incam_final
        
        if optimize_interval > 1 :
                                                                                                
            self._interpolate_frames(optimize_interval)
                                                                                
        self._interpolate_object_pose_slerp(optimize_interval)
        for begin_frame, end_frame in self.leave_hand_pairs:
            self._interpolate_depths(begin_frame, end_frame)

                                                                                 
        self._lowpass_smooth_all(
            alpha=smoothing_alpha,
            beta_quat=smoothing_beta,
            bidirectional=True,
            ema_passes=smoothing_passes,
            window_size=smoothing_window,
            method=smoothing_method,
            cutoff=smoothing_cutoff,
            butter_order=smoothing_order,
            fs=smoothing_fs,
        )
    def _interpolate_depths(self, begin_frame, end_frame):
        with torch.no_grad():
            obj_center_begin = self.get_object_points(begin_frame, sampled=True).mean(dim=0)
            obj_center_end = self.get_object_points(end_frame, sampled=True).mean(dim=0)
            depth_begin = obj_center_begin[2]
            depth_end = obj_center_end[2]
            for frame_idx in range(begin_frame, end_frame):
                alpha = (frame_idx - begin_frame) / (end_frame - begin_frame)
                interpolate_depth = (1 - alpha) * depth_begin + alpha * depth_end
                frame_obj_center = self.get_object_points(frame_idx, sampled=True).mean(dim=0)
                oringin_depth = frame_obj_center[2]
                if self.T_total_frames is not None and frame_idx < len(self.T_total_frames) and self.T_total_frames[frame_idx] is not None:
                    self.T_total_frames[frame_idx][2] = self.T_total_frames[frame_idx][2] + interpolate_depth - oringin_depth
                else:
                    self.obj_transl_params[frame_idx][2] = self.obj_transl_params[frame_idx][2] + interpolate_depth - oringin_depth

    def _interpolate_frames(self, interval):
        with torch.no_grad():
            for i in range(0, self.seq_length, interval):
                start_frame = i
                end_frame = min(i + interval, self.seq_length - 1)
                if start_frame >= end_frame:
                    continue
                for mid_frame in range(start_frame, end_frame):
                    alpha = (mid_frame - start_frame) / (end_frame - start_frame)
                    
                    self.body_pose_params[mid_frame].copy_(
                        (1 - alpha) * self.body_pose_params[start_frame] + 
                        alpha * self.body_pose_params[end_frame]
                    )
                    self.shape_params[mid_frame].copy_(
                        (1 - alpha) * self.shape_params[start_frame] + 
                        alpha * self.shape_params[end_frame]
                    )
                    self.left_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.left_hand_params[start_frame] + 
                        alpha * self.left_hand_params[end_frame]
                    )
                    self.right_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.right_hand_params[start_frame] + 
                        alpha * self.right_hand_params[end_frame]
                    )
    def _interpolate_object_pose_slerp(self, interval):
        with torch.no_grad():
                        
            self.R_total_frames = [None for _ in range(self.seq_length)]
            self.T_total_frames = [None for _ in range(self.seq_length)]

                                                                                                    
            if self.is_static_object and len(self.R_incam_static) == self.seq_length and len(self.T_incam_static) == self.seq_length:
                for f in range(self.seq_length):
                    self.R_total_frames[f] = self.R_incam_static[f]
                    self.T_total_frames[f] = self.T_incam_static[f]
                return

            for f in self.R_incam_pre_contact.keys():
                if 0 <= f < self.seq_length:
                    self.R_total_frames[f] = self.R_incam_pre_contact[f]
                    self.T_total_frames[f] = self.T_incam_pre_contact[f]

                               
            anchors = [] if self.frames_optimized is None else list(self.frames_optimized)
            anchors = sorted(set(int(x) for x in anchors))
            if 0 not in anchors:
                anchors = [0] + anchors
            if (self.seq_length - 1) not in anchors:
                anchors = anchors + [self.seq_length - 1]

                                                                   
            anchor_quats = {}
            anchor_trans = {}
            for idx in anchors:
                Rk, Tk = self._compute_object_params_no_cache(idx)
                anchor_quats[idx] = rotation_matrix_to_quaternion(Rk)
                anchor_trans[idx] = Tk

                                                                                             
            for a, b in zip(anchors[:-1], anchors[1:]):
                qa, qb = anchor_quats[a], anchor_quats[b]
                Ta, Tb = anchor_trans[a], anchor_trans[b]
                length = max(b - a, 1)
                for f in range(a, b + 1):
                                                                   
                    if self.R_total_frames[f] is not None and self.T_total_frames[f] is not None:
                        continue
                    if f == a:
                        alpha = 0.0
                    elif f == b:
                        alpha = 1.0
                    else:
                        alpha = (f - a) / float(length)
                    qf = quaternion_slerp(qa, qb, torch.tensor(alpha, dtype=torch.float32, device=qa.device))
                    Rf = quaternion_to_rotation_matrix(qf)
                    Tf = (1 - alpha) * Ta + alpha * Tb
                    self.R_total_frames[f] = Rf
                    self.T_total_frames[f] = Tf

                                                     
            for f in range(self.seq_length):
                if self.R_total_frames[f] is None or self.T_total_frames[f] is None:
                    Rf, Tf = self._compute_object_params_no_cache(f)
                    self.R_total_frames[f] = Rf
                    self.T_total_frames[f] = Tf

    def _compute_object_params_no_cache(self, frame_idx):
                
        if self.is_static_object and len(self.R_incam_static) > 0 and len(self.T_incam_static) > 0:
            return self.R_incam_static[frame_idx], self.T_incam_static[frame_idx]
                     
        if frame_idx in self.R_incam_pre_contact and frame_idx in self.T_incam_pre_contact:
            return self.R_incam_pre_contact[frame_idx], self.T_incam_pre_contact[frame_idx]
        R_residual = self.get_object_transform(frame_idx)
        R_icp = self.icp_obj_R[frame_idx]
        R_final = torch.mm(R_residual, R_icp)
        t_residual = self.obj_transl_params[frame_idx]
        t_icp = self.icp_obj_transl[frame_idx]
        abs_idx = frame_idx + self.start_frame
        depth_centers = torch.tensor(self.centers_depth[abs_idx], dtype=torch.float32, device=R_final.device)
        t_final = torch.mv(R_final, depth_centers) + torch.mv(R_residual, t_icp) + t_residual
        return R_final, t_final

    def _ema_smooth_series(self, tensor_list, alpha=0.5, bidirectional=True):
        if len(tensor_list) == 0:
            return []
        device = tensor_list[0].device
        dtype = tensor_list[0].dtype
                         
        data = torch.stack([t.to(device=device, dtype=dtype) for t in tensor_list], dim=0)
        T = data.shape[0]
                 
        y_fwd = data.clone()
        for t in range(1, T):
            y_fwd[t] = alpha * data[t] + (1.0 - alpha) * y_fwd[t - 1]
        if not bidirectional:
            return [y_fwd[t] for t in range(T)]
                  
        y_bwd = data.clone()
        for t in range(T - 2, -1, -1):
            y_bwd[t] = alpha * data[t] + (1.0 - alpha) * y_bwd[t + 1]
        y = 0.5 * (y_fwd + y_bwd)
        return [y[t] for t in range(T)]

    def _smooth_quaternion_sequence(self, q_list, beta=0.5, bidirectional=True):
        if len(q_list) == 0:
            return []
                              
        qn = [q / (torch.linalg.norm(q) + 1e-8) for q in q_list]
        T = len(qn)
        q_fwd = [None] * T
        q_fwd[0] = qn[0]
        for t in range(1, T):
            q_fwd[t] = quaternion_slerp(q_fwd[t - 1], qn[t], torch.tensor(beta, dtype=qn[t].dtype, device=qn[t].device))
        if not bidirectional:
            return q_fwd
        q_bwd = [None] * T
        q_bwd[-1] = qn[-1]
        for t in range(T - 2, -1, -1):
            q_bwd[t] = quaternion_slerp(q_bwd[t + 1], qn[t], torch.tensor(beta, dtype=qn[t].dtype, device=qn[t].device))
        q_out = []
        for t in range(T):
            q_out.append(quaternion_slerp(q_fwd[t], q_bwd[t], torch.tensor(0.5, dtype=qn[t].dtype, device=qn[t].device)))
        return q_out

    def _box_smooth_series(self, tensor_list, window_size: int):
        if len(tensor_list) == 0 or window_size <= 1:
            return tensor_list
        k = max(1, int(window_size))
        if k % 2 == 0:
            k += 1
        half = k // 2
        T = len(tensor_list)
        out = []
        for t in range(T):
            s = max(0, t - half)
            e = min(T, t + half + 1)
            vals = [tensor_list[i] for i in range(s, e)]
            out.append(torch.mean(torch.stack(vals, dim=0), dim=0))
        return out

    def _box_smooth_quaternion_sequence(self, q_list, window_size: int):
        if len(q_list) == 0 or window_size <= 1:
            return q_list
        k = max(1, int(window_size))
        if k % 2 == 0:
            k += 1
        half = k // 2
        T = len(q_list)
        out = []
        for t in range(T):
            s = max(0, t - half)
            e = min(T, t + half + 1)
                                  
            q_mean = q_list[s]
            for i in range(s + 1, e):
                                                  
                n = i - s + 1
                w = 1.0 / n
                q_mean = quaternion_slerp(q_mean, q_list[i], torch.tensor(w, dtype=q_mean.dtype, device=q_mean.device))
            out.append(q_mean / (torch.linalg.norm(q_mean) + 1e-8))
        return out

    def _gaussian_kernel(self, window_size: int, sigma: float, device, dtype):
        k = max(1, int(window_size))
        if k % 2 == 0:
            k += 1
        center = k // 2
        idx = torch.arange(k, device=device, dtype=dtype)
        if sigma <= 0:
            w = torch.ones(k, device=device, dtype=dtype)
        else:
            w = torch.exp(-0.5 * ((idx - center) / sigma) ** 2)
        w = w / (w.sum() + 1e-8)
        return w

    def _gaussian_smooth_series(self, tensor_list, window_size: int, sigma: float = None):
        if len(tensor_list) == 0 or window_size <= 1:
            return tensor_list
        device = tensor_list[0].device
        dtype = tensor_list[0].dtype
        k = max(1, int(window_size))
        if k % 2 == 0:
            k += 1
        if sigma is None:
            sigma = max(1.0, k / 3.0)
        half = k // 2
        kernel = self._gaussian_kernel(k, sigma, device, dtype)
        T = len(tensor_list)
        out = []
        for t in range(T):
            s = max(0, t - half)
            e = min(T, t + half + 1)
                                          
            ks = half - (t - s)
            ke = ks + (e - s)
            kw = kernel[ks:ke]
            vals = [tensor_list[i] * kw[i - s] for i in range(s, e)]
            denom = kw.sum() + 1e-8
            out.append(torch.stack(vals, dim=0).sum(dim=0) / denom)
        return out

    def _gaussian_smooth_quaternion_sequence(self, q_list, window_size: int, sigma: float = None):
        if len(q_list) == 0 or window_size <= 1:
            return q_list
        k = max(1, int(window_size))
        if k % 2 == 0:
            k += 1
        if sigma is None:
            sigma = max(1.0, k / 3.0)
                                                      
        device = q_list[0].device
        dtype = q_list[0].dtype
        kernel = self._gaussian_kernel(k, sigma, device, dtype)
        half = k // 2
        T = len(q_list)
        out = []
        for t in range(T):
            s = max(0, t - half)
            e = min(T, t + half + 1)
            ks = half - (t - s)
            ke = ks + (e - s)
            kw = kernel[ks:ke]
                                 
            q_mean = q_list[s]
            w_sum = kw[0]
            for i in range(s + 1, e):
                w_i = kw[i - s]
                gamma = float(w_i / (w_sum + w_i + 1e-8))
                q_mean = quaternion_slerp(q_mean, q_list[i], torch.tensor(gamma, dtype=dtype, device=device))
                w_sum = w_sum + w_i
            out.append(q_mean / (torch.linalg.norm(q_mean) + 1e-8))
        return out

    def _lowpass_smooth_all(self, alpha=0.5, beta_quat=0.5, bidirectional=True, ema_passes=1, window_size=1, method='ema_box', cutoff=0.08, butter_order=4, fs=1.0):
        with torch.no_grad():
                                     
            if len(self.body_pose_params) > 0:
                series = [p.data for p in self.body_pose_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = self._ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = self._box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = self._gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.body_pose_params)):
                    self.body_pose_params[i].copy_(smoothed[i])
            if len(self.shape_params) > 0:
                series = [p.data for p in self.shape_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = self._ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = self._box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = self._gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.shape_params)):
                    self.shape_params[i].copy_(smoothed[i])
            if len(self.left_hand_params) > 0:
                series = [p.data for p in self.left_hand_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = self._ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = self._box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = self._gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.left_hand_params)):
                    self.left_hand_params[i].copy_(smoothed[i])
            if len(self.right_hand_params) > 0:
                series = [p.data for p in self.right_hand_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = self._ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = self._box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = self._gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.right_hand_params)):
                    self.right_hand_params[i].copy_(smoothed[i])                         
            if self.R_total_frames is not None and self.T_total_frames is not None:
                                    
                q_list = []
                idx_map = []
                for i, R in enumerate(self.R_total_frames):
                    if R is None:
                        q_list.append(None)
                    else:
                        q_list.append(rotation_matrix_to_quaternion(R))
                    idx_map.append(i)
                                                                  
                for i in range(len(q_list)):
                    if q_list[i] is None:
                        R_fallback, _ = self._compute_object_params_no_cache(i)
                        q_list[i] = rotation_matrix_to_quaternion(R_fallback)
                qs = q_list
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        qs = self._smooth_quaternion_sequence(qs, beta=beta_quat, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        qs = self._box_smooth_quaternion_sequence(qs, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        qs = self._gaussian_smooth_quaternion_sequence(qs, window_size)
                q_sm = qs
                for i in range(len(q_sm)):
                    self.R_total_frames[i] = quaternion_to_rotation_matrix(q_sm[i])

                                            
                t_list = [t if t is not None else self._compute_object_params_no_cache(i)[1] for i, t in enumerate(self.T_total_frames)]
                ts = t_list
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        ts = self._ema_smooth_series(ts, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        ts = self._box_smooth_series(ts, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        ts = self._gaussian_smooth_series(ts, window_size)
                t_sm = ts
                for i in range(len(t_sm)):
                    self.T_total_frames[i] = t_sm[i]
    def save_sequence(self, output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
        
        for i in range(self.seq_length):
            frame_dir = os.path.join(output_dir, f'frame_{i + self.start_frame:04d}')  
            os.makedirs(frame_dir, exist_ok=True)   
            self.current_frame = i 
            human_faces = self.get_body_faces(sampled=False)  
            human_verts = self.get_body_points(i, sampled=False).detach().cpu().numpy()
            object_vertices = self.get_object_points(i, sampled=False).detach().cpu().numpy()
            object_vertices_ = self.get_object_points(i, sampled=True).detach().cpu().numpy()
            incam_params = (self.global_orient[i], self.transl[i])
            global_params = (self.global_body_params["global_orient"][i], self.global_body_params["transl"][i])
            h_mesh = o3d.geometry.TriangleMesh()  
            h_mesh.vertices = o3d.utility.Vector3dVector(human_verts)  
            h_mesh.triangles = o3d.utility.Vector3iVector(human_faces)  
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'human.obj'), h_mesh)  
            obj_mesh = o3d.geometry.TriangleMesh()  
            obj_mesh.vertices = o3d.utility.Vector3dVector(object_vertices)  
            obj_mesh.triangles = o3d.utility.Vector3iVector(self.get_object_faces(i, sampled=False))
            obj_mesh_ = o3d.geometry.TriangleMesh()
            obj_mesh_.vertices = o3d.utility.Vector3dVector(object_vertices_)
            obj_mesh_.triangles = o3d.utility.Vector3iVector(self.get_object_faces(i, sampled=True))
                                                                                                      
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'object.obj'), obj_mesh)
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'object_sampled.obj'), obj_mesh_)
            
                     
            corresponding_points = self.get_corresponding_point(i)  
            body_points = corresponding_points['body_points'].detach().cpu().numpy()  
            object_points = corresponding_points['object_points'].detach().cpu().numpy()  
            
            lines = [[i, i + len(body_points)] for i in range(len(body_points))]  
            points = np.vstack((body_points, object_points))  
            colors = [[0, 1, 0] for _ in range(len(lines))]  
            
            line_set = o3d.geometry.LineSet(  
                points=o3d.utility.Vector3dVector(points),  
                lines=o3d.utility.Vector2iVector(lines),  
            )  
            line_set.colors = o3d.utility.Vector3dVector(colors)  
            o3d.io.write_line_set(os.path.join(frame_dir, 'contact_points.ply'), line_set)  
    

    def create_visualization_video(self, output_dir, K, video_path=None, fps=3, clear=True):
        if os.path.exists(output_dir) and clear:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        human_faces = np.array(self.get_body_faces(sampled=True), dtype=np.int32)
        obj_faces= np.array(self.sampled_obj_meshes[0].triangles, dtype=np.int32)
                                                                              
        renderer = Renderer(self.width, self.height, device="cuda",faces_human=human_faces,faces_obj=obj_faces,K=K)
        for i in tqdm(range(0, self.seq_length, 2), desc="rendering frames"):
            human_verts = self.get_body_points(i, sampled=True)
            object_mesh = self.sampled_obj_meshes[i]
            transform = self.get_object_transform(i).detach().cpu().numpy()
            object_vertices = self.get_object_points(i,sampled=True)
            img_raw = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255                                     
            if hasattr(object_mesh, 'vertex_colors'):
                object_color = np.asarray(object_mesh.vertex_colors)
            else:
                object_color = [0.3, 0.5, 0.7]
            if len(object_color) == 0:
                object_color = [0.3, 0.5, 0.7]

            img = renderer.render_mesh_hoi(human_verts, object_vertices, img_raw, [0.8, 0.8, 0.8],object_color)
            start_frame = (self.start_frame // 2) * 2
            frame_path = os.path.join(output_dir, f"frame_{i + start_frame:04d}.png")

            img = Image.fromarray(img)
            img.save(frame_path)


    def get_object_params(self, frame_idx=None):
        if frame_idx is None:
            frame_idx = self.current_frame
                                                           
        if self.R_total_frames is not None and self.T_total_frames is not None:
            if frame_idx < len(self.R_total_frames):
                Rt = self.R_total_frames[frame_idx]
                Tt = self.T_total_frames[frame_idx] if frame_idx < len(self.T_total_frames) else None
                if Rt is not None and Tt is not None:
                    return Rt, Tt
        if self.is_static_object and len(self.R_incam_static) > 0 and len(self.T_incam_static) > 0:
            R_final = self.R_incam_static[frame_idx]
            t_final = self.T_incam_static[frame_idx]
            return R_final, t_final
                                                                             
        if frame_idx in self.R_incam_pre_contact and frame_idx in self.T_incam_pre_contact:
            return self.R_incam_pre_contact[frame_idx], self.T_incam_pre_contact[frame_idx]
        R_residual = self.get_object_transform(frame_idx)
        R_icp = self.icp_obj_R[frame_idx]
        R_final = torch.mm(R_residual, R_icp)
        t_residual = self.obj_transl_params[frame_idx]
        t_icp = self.icp_obj_transl[frame_idx]
        abs_idx = frame_idx + self.start_frame
        depth_centers = torch.tensor(self.centers_depth[abs_idx], dtype=torch.float32, device=R_final.device)
        t_final = torch.mv(R_final, depth_centers) + torch.mv(R_residual, t_icp) + t_residual
        return R_final, t_final               
    def get_optimize_result(self):
        R_finals = []
        t_finals = []
                                                        
        print(self.start_frame,self.end_frame)
        body_pose_finals=[]
        betas_finals=[]
        global_orient_finals=[]
        transl_finals=[]
        hand_poses_finals=[]
        for i in range(self.seq_length):
            R_final, t_final = self.get_object_params(i)
            R_finals.append(R_final.detach().cpu().numpy())
            t_finals.append(t_final.detach().cpu().numpy())
            body_pose_finals.append(self.body_pose_params[i].reshape(1, -1).detach().cpu())
            betas_finals.append(self.shape_params[i].reshape(1, -1).detach().cpu())
            global_orient_finals.append(self.global_orient[i].reshape(1, 3).detach().cpu())
            transl_finals.append(self.transl[i].reshape(1, -1).detach().cpu())
            hand_poses_finals.append({
                "left_hand": self.left_hand_params[i].reshape(1, -1).detach().cpu().numpy(),
                "right_hand": self.right_hand_params[i].reshape(1, -1).detach().cpu().numpy()
            })                
        body_params_all={
            "body_pose": torch.cat(body_pose_finals, dim=0).numpy(),
            "betas": torch.cat(betas_finals, dim=0).numpy(),
            "global_orient": torch.cat(global_orient_finals, dim=0).numpy(),
            "transl": torch.cat(transl_finals, dim=0).numpy()
        }
        return body_params_all, hand_poses_finals, R_finals, t_finals
