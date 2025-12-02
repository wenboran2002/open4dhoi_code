from pykalman import KalmanFilter
import torch
import torch.nn as nn
import numpy as np
import gtsam
import os
import open3d as o3d

def _kalman_smoothing(transl):
        def apply_kalman(param_list, observation_weight=5.0, transition_cov=0.1, param_name=""):
            num_frames = len(param_list)
            if num_frames == 0: return
            param_shape = param_list[0].shape
            num_dims = param_list[0].numel()
            
            with torch.no_grad():
                observations = np.zeros((num_frames, num_dims), dtype=np.float32)
                for frame_idx in range(num_frames):
                    flat_param = param_list[frame_idx].view(-1).cpu().numpy()
                    observations[frame_idx] = flat_param
                
                data_variance = np.var(observations, axis=0)
                data_variance = np.maximum(data_variance, 1e-6)

                kf = KalmanFilter(
                    initial_state_mean=observations[0],
                    initial_state_covariance=np.diag(data_variance),
                    transition_matrices=np.eye(num_dims),
                    observation_matrices=np.eye(num_dims),
                    observation_covariance=np.diag(data_variance) * observation_weight,
                    transition_covariance=np.diag(data_variance) * transition_cov,
                )

                smooth_states, _ = kf.smooth(observations)
                device = param_list[0].device
                for frame_idx in range(num_frames):
                    smoothed_flat = torch.from_numpy(smooth_states[frame_idx]).float().to(device)
                    param_list[frame_idx].data = smoothed_flat.view(param_shape)
        transl = [nn.Parameter(p) for p in self.transl]
        apply_kalman(transl, observation_weight=1, transition_cov=0.1, param_name="transl")
        self.transl = [p.data for p in transl]
        apply_kalman(self.body_pose_params, observation_weight=0.05, transition_cov=0.1, param_name="body_pose")
        apply_kalman(self.shape_params, observation_weight=0.1, transition_cov=0.1, param_name="shape")
        apply_kalman(self.left_hand_params, observation_weight=0.1, transition_cov=0.1, param_name="left_hand")
        apply_kalman(self.right_hand_params, observation_weight=0.1, transition_cov=0.1, param_name="right_hand")
 
        if not self.is_static_object:
            # Smoothing via Lie Algebra (se3) using GTSAM for robustness.
            with torch.no_grad():
                pose_seq_gtsam = []
                # Move tensors to CPU and convert to numpy for gtsam
                final_R_list_cpu = [self.get_object_transform(i).cpu().numpy() for i in range(self.seq_length)]
                t_base_list_cpu = [t.cpu().numpy() for t in self.base_obj_transl]
                t_residual_list_cpu = [t.cpu().numpy() for t in self.obj_transl_params]
                centers_cpu = np.array(self.centers)

                for i in range(self.seq_length):
                    R_base = self.base_obj_R[i].cpu().numpy()
                    R_residual = final_R_list_cpu[i]
                    R_final_no_initial = R_residual @ R_base
                    R_final = R_final_no_initial @ self.initial_R[i]

                    t_base = t_base_list_cpu[i]
                    t_residual = t_residual_list_cpu[i]
                    
                    center = centers_cpu[i]
                    effective_translation = R_final_no_initial @ center + R_residual @ t_base + t_residual

                    pose_gtsam = gtsam.Pose3(gtsam.Rot3(R_final), effective_translation)
                    pose_seq_gtsam.append(pose_gtsam)
                twists = []
                for i in range(1, self.seq_length):
                    pose_i_minus_1 = pose_seq_gtsam[i-1]
                    pose_i = pose_seq_gtsam[i]
                    relative_pose = pose_i_minus_1.inverse().compose(pose_i)
                    twist = gtsam.Pose3.Logmap(relative_pose) # 6D se3 vector
                    twists.append(nn.Parameter(torch.from_numpy(twist).float()))
                
                if not twists: 
                    return
                apply_kalman(twists, observation_weight=3, transition_cov=0.1, param_name="twists")
                smoothed_poses_gtsam = [pose_seq_gtsam[0]]
                for i in range(len(twists)):
                    smoothed_twist_np = twists[i].data.numpy()
                    relative_pose_smoothed = gtsam.Pose3.Expmap(smoothed_twist_np)
                    new_pose = smoothed_poses_gtsam[i].compose(relative_pose_smoothed)
                    smoothed_poses_gtsam.append(new_pose)

                smoothed_final_R_list = [p.rotation().matrix() for p in smoothed_poses_gtsam]
                smoothed_effective_transl_list = [p.translation() for p in smoothed_poses_gtsam]
                
                device = self.obj_x_params[0].device
                smoothed_final_R_stack = torch.from_numpy(np.array(smoothed_final_R_list)).float().to(device)
                smoothed_effective_transl_list = [p.translation() for p in smoothed_poses_gtsam]
                
                device = self.obj_x_params[0].device
                smoothed_final_R_stack = torch.from_numpy(np.array(smoothed_final_R_list)).float().to(device)
                smoothed_effective_transl = torch.from_numpy(np.array(smoothed_effective_transl_list)).float().to(device)

                base_R_inv_stack = torch.stack(self.base_obj_R).transpose(1, 2)
                initial_R_stack = torch.from_numpy(np.array(self.initial_R)).float().to(device)
                initial_R_inv_tensor = initial_R_stack.transpose(1, 2)
                
                smoothed_final_R_no_initial_stack = torch.matmul(smoothed_final_R_stack, initial_R_inv_tensor)
                new_residual_R_stack = torch.matmul(smoothed_final_R_no_initial_stack, base_R_inv_stack)
                new_residual_euler_rad = matrix_to_euler_angles(new_residual_R_stack, "ZYX")
                new_residual_euler_deg = torch.rad2deg(new_residual_euler_rad)

                self.obj_z_params = [nn.Parameter(p) for p in new_residual_euler_deg[:, 0]]
                self.obj_y_params = [nn.Parameter(p) for p in new_residual_euler_deg[:, 1]]
                self.obj_x_params = [nn.Parameter(p) for p in new_residual_euler_deg[:, 2]]

                centers_stack = torch.tensor(self.centers, device=device, dtype=torch.float32)
                base_t_stack = torch.stack(self.base_obj_transl)

                rotated_centers = torch.bmm(smoothed_final_R_no_initial_stack, centers_stack.unsqueeze(-1)).squeeze(-1)
                new_t_final = smoothed_effective_transl - rotated_centers
                
                rotated_base_t = torch.bmm(new_residual_R_stack, base_t_stack.unsqueeze(-1)).squeeze(-1)
                new_residual_transl = new_t_final - rotated_base_t
                self.obj_transl_params = [nn.Parameter(p) for p in new_residual_transl]
 
