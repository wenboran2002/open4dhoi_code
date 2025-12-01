import torch
import smplx
import numpy as np
import open3d as o3d
import json
import os
import sys
import trimesh
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .utils.hoi_utils import load_transformation_matrix
from copy import deepcopy
from .utils.icppnp import solve_weighted_priority
from .utils.camera_utils import transform_to_global

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class HOISolver:
    def __init__(self, model_folder, device=None):
        """
        初始化HOI求解器
        Args:
            model_folder: SMPL-X模型文件路径
            device: 计算设备，默认自动选择
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化SMPL-X模型
        self.model = smplx.create(model_folder, model_type='smplx',
                                  gender='neutral',
                                  num_betas=10,
                                  flat_hand_mean=True,
                                  use_pca=False,
                                  num_expression_coeffs=10).to(self.device)

        # 定义四肢关节名称到SMPL-X关节索引的映射
        self.limb_joint_names_to_idx = {
            'left_foot': 7,  # 左脚踝
            'right_foot': 8,  # 右脚踝
            'left_wrist': 20,  # 左手腕
            'right_wrist': 21  # 右手腕
        }

        print(f"HOI Solver initialized on device: {self.device}")

    def save_mesh_as_obj(self, vertices, faces, filename):
        """保存网格为OBJ文件"""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename, mesh)
        print(f"Saved mesh to {filename}")

    def apply_transform_to_model(self, vertices, transform_matrix):
        """对顶点应用变换矩阵"""
        homogenous_verts = np.hstack([vertices, np.ones((len(vertices), 1))])
        transformed = (transform_matrix @ homogenous_verts.T).T
        return transformed[:, :3] / transformed[:, [3]]


    def get_corresponding_point(self, object_points_idx, body_points_idx, body_points, object_mesh):
        """获取人体和物体的对应点"""
        interacting_indices = object_points_idx[:, 1] != 0
        interacting_body_indices = np.asarray(body_points_idx)[interacting_indices]

        body_points = body_points[interacting_body_indices]

        object_points = torch.tensor(np.array(object_mesh.vertices),
                                     device=body_points.device).float()
        obj_index = object_points_idx[interacting_indices][:, 0]
        interactiong_obj = object_points[obj_index]

        corresponding_points = {
            'body_points': body_points.numpy(),
            'object_points': interactiong_obj,
            'body_indices': interacting_body_indices,
            'obj_indices': obj_index
        }

        return corresponding_points

    def rigid_transform_svd_with_corr(self, A, B):
        """使用SVD计算刚体变换"""
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T

        if np.linalg.det(R_mat) < 0:
            Vt[2, :] *= -1
            R_mat = Vt.T @ U.T

        t = centroid_B - R_mat @ centroid_A
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T

    def residuals_with_corr(self, x, A, B):
        """用于最小二乘优化的残差函数"""
        rot_vec = x[:3]
        t = x[3:]
        R_mat = R.from_rotvec(rot_vec).as_matrix()
        A_trans = (R_mat @ A.T).T + t
        return (A_trans - B).ravel()

    def refine_rigid_with_corr(self, A, B, x0=None):
        """使用最小二乘法精化刚体变换"""
        if x0 is None:
            T0 = self.rigid_transform_svd_with_corr(A, B)
            rot0 = R.from_matrix(T0[:3, :3]).as_rotvec()
            t0 = T0[:3, 3]
            x0 = np.hstack([rot0, t0])

        res = least_squares(self.residuals_with_corr, x0, args=(A, B))
        R_opt = R.from_rotvec(res.x[:3]).as_matrix()
        t_opt = res.x[3:]
        T_opt = np.eye(4)
        T_opt[:3, :3] = R_opt
        T_opt[:3, 3] = t_opt
        return T_opt

    def jacobian_ik_step_selective(self, global_orient, body_pose, betas, transl,
                                   target_joint_idxs, target_positions,
                                   constraint_joint_idxs, constraint_positions,
                                   lr=1.0):
        """选择性IK优化步骤"""
        global_orient = global_orient.clone().detach().requires_grad_(True)
        body_pose = body_pose.clone().detach().requires_grad_(True)

        output = self.model(global_orient=global_orient,
                            body_pose=body_pose,
                            betas=betas,
                            transl=transl,
                            return_full_pose=True)
        joints = output.joints[0]

        total_loss = 0
        loss_count = 0

        # 目标关节的损失
        for i, joint_idx in enumerate(target_joint_idxs):
            joint_pred = joints[joint_idx]
            target_pos = target_positions[i]
            total_loss += torch.nn.functional.mse_loss(joint_pred, target_pos)
            loss_count += 1

        # 约束关节的损失
        for i, joint_idx in enumerate(constraint_joint_idxs):
            joint_pred = joints[joint_idx]
            constraint_pos = constraint_positions[i]
            total_loss += torch.nn.functional.mse_loss(joint_pred, constraint_pos)
            loss_count += 1

        if loss_count > 0:
            total_loss = total_loss / loss_count

        total_loss.backward()

        with torch.no_grad():
            body_pose_new = body_pose - lr * body_pose.grad
            global_orient_new = global_orient - lr * global_orient.grad

        return global_orient_new.detach(), body_pose_new.detach(), total_loss.item()

    def run_joint_ik(self, global_orient, body_pose, betas, transl,
                     target_joints_dict, constraint_joints_list,
                     max_iter=40, lr=1.5):
        """运行关节特定IK优化"""
        target_joint_idxs = list(target_joints_dict.keys())
        target_offsets = list(target_joints_dict.values())

        # 获取初始约束关节相对位置
        with torch.no_grad():
            output = self.model(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas,
                                transl=transl,
                                return_full_pose=True)
            joints = output.joints[0]
            pelvis = joints[0]
            constraint_offsets = [joints[idx] - pelvis for idx in constraint_joints_list]

        for i in range(max_iter):
            with torch.no_grad():
                output = self.model(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas,
                                    transl=transl,
                                    return_full_pose=True)
                joints = output.joints[0]
                pelvis = joints[0]

                target_positions = [pelvis + offset for offset in target_offsets]
                constraint_positions = [pelvis + offset for offset in constraint_offsets]

            global_orient, body_pose, loss = self.jacobian_ik_step_selective(
                global_orient, body_pose, betas, transl,
                target_joint_idxs, target_positions,
                constraint_joints_list, constraint_positions,
                lr=lr
            )

            # print(f"[Iter {i:02d}] Joint IK Loss: {loss:.6f}")
            if loss < 1e-5:
                break

        return global_orient, body_pose

    def check_limb_joints_in_corresp(self, corresp, body_points_idx, joint_mapping, part_kp_file):
        """检查对应关系中是否包含四肢关节"""
        with open(part_kp_file, 'r') as f:
            human_part = json.load(f)

        body_indices = corresp['body_indices']
        target_joints = {}
        involved_limb_joints = set()

        for joint_name, part_kp_name in joint_mapping.items():
            if joint_name in self.limb_joint_names_to_idx:
                if part_kp_name in human_part:
                    part_kp_index = human_part[part_kp_name]['index']
                    if part_kp_index in body_indices:
                        smplx_joint_idx = self.limb_joint_names_to_idx[joint_name]
                        involved_limb_joints.add(smplx_joint_idx)
                        corresp_position = np.where(body_indices == part_kp_index)[0]
                        if len(corresp_position) > 0:
                            target_joints[smplx_joint_idx] = corresp_position[0]

        all_limb_indices = set(self.limb_joint_names_to_idx.values())
        constraint_joints = list(all_limb_indices - involved_limb_joints)

        return target_joints, constraint_joints

    def solve_hoi(self, obj_init, obj_sample_init, body_params, global_body_params, i, start_frame, end_frame, hand_poses,
                  object_points_idx, body_points_idx, object_points, image_points, joint_mapping, K=None,
                  part_kp_file=resource_path("video_optimizer/data/part_kp.json"), save_meshes=False, all_mutiview_info=None, is_multiview=False):
        """
        直接求解HOI，输入预处理的物体和人体参数
        Args:
            obj_init: 原始物体网格
            obj_sample_init: 采样的物体网格
            body_params: 包含多帧信息的人体参数字典
            i: 当前帧在序列中的索引
            start_frame, end_frame: 帧范围
            hand_poses: 手部姿态
            object_points_idx: 物体点索引
            body_points_idx: 人体点索引
            object_points: 用于2D-3D对应的物体点
            image_points: 用于2D-3D对应的图像点
            K: 相机内参
            joint_mapping: 关节映射字典
            part_kp_file: 人体关键点文件路径
            save_meshes: 是否保存网格文件
            is_multiview: 是否为多视角优化
            cam_params: 多视角相机的参数 (K, R, T)
        """
        print("Starting HOI solving with direct inputs...")

        # 从human_params中提取参数
        body_pose = body_params["body_pose"][i + start_frame].reshape(1, -1).cuda()
        global_orient = body_params["global_orient"][i + start_frame].reshape(1, 3).cuda()
        shape = body_params["betas"][i + start_frame].reshape(1, -1).cuda()
        transl = body_params["transl"][i + start_frame].reshape(1, -1).cuda()
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1).cuda()
        left_hand_pose = np.array(hand_poses[str(i + start_frame)]["left_hand"])
        right_hand_pose = np.array(hand_poses[str(i + start_frame)]["right_hand"])

        # 生成初始人体网格
        output = self.model(betas=shape,
                         body_pose=body_pose,
                         left_hand_pose=torch.from_numpy(left_hand_pose).float().cuda(),
                         right_hand_pose=torch.from_numpy(right_hand_pose).float().cuda(),
                         jaw_pose=zero_pose,
                         leye_pose=zero_pose,
                         reye_pose=zero_pose,
                         global_orient=global_orient,
                         expression=torch.zeros((1, 10)).float().cuda(),
                         transl=transl
                         )
        hpoints = output.vertices[0].detach().cpu()

        if save_meshes:
            self.save_mesh_as_obj(hpoints, self.model.faces, "human_before_ik.obj")

        # Step 1: ICP对齐
        object_points_idx = object_points_idx[i]
        body_points_idx = body_points_idx[i]

        object_points = object_points[i].reshape(-1,3)
        image_points = image_points[i].reshape(-1, 2)


        # time1 = time.time()
        print("Starting ICP alignment...")
        corresp = self.get_corresponding_point(object_points_idx, body_points_idx, hpoints, obj_init)
        print(f"Correspondence points shape: {corresp['body_points'].shape}")

        source_points_3d = np.asarray(corresp['object_points'])
        target_points_3d = np.asarray(corresp['body_points'])

        if is_multiview:
            incam_params = (body_params["global_orient"][i], body_params["transl"][i])
            global_params = (global_body_params["global_orient"][i], global_body_params["transl"][i])
        else:
            incam_params = None
            global_params = None
        R_opt, t_opt = solve_weighted_priority(incam_params, global_params, source_points_3d, target_points_3d, object_points, image_points, K, all_mutiview_info, weight_3d=900.0, weight_2d=2.0)
        # 对物体应用变换
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_opt
        transform_matrix[:3, 3] = t_opt.flatten()

        # if save_meshes:
        # print('transf',R_opt,t_opt)
        # overts=deepcopy(np.asarray(obj_init.vertices))
        # overts_transformed = self.apply_transform_to_model(overts, transform_matrix)
        # obj_oo=o3d.geometry.TriangleMesh()
        # obj_oo.vertices=o3d.utility.Vector3dVector(overts_transformed)
        # obj_oo.triangles=o3d.utility.Vector3iVector(np.asarray(obj_init.triangles).astype(np.int32))
        # h_pcd=o3d.geometry.TriangleMesh()
        # h_pcd.vertices=o3d.utility.Vector3dVector(hpoints)
        # h_pcd.triangles=o3d.utility.Vector3iVector(self.model.faces.astype(np.int32))
        # o3d.io.write_triangle_mesh("object_after_icp.obj", obj_oo+h_pcd)
        # print("ICP alignment completed.")
        # time2 = time.time()
        # print(f"ICP alignment time: {time2 - time1}")
        # Step 2: 检查是否需要IK优化
        # time3 = time.time()
        # target_joints, constraint_joints = self.check_limb_joints_in_corresp(
        #     corresp, body_points_idx, joint_mapping, part_kp_file)

        # if target_joints:
        #     print(f"Found limb joints to optimize: {list(target_joints.keys())}")
        #     print(f"Constraint joints: {constraint_joints}")

        #     # 获取ICP后的物体对应点位置
        #     transformed_obj_points = (transform_matrix @ np.hstack([source_points_3d, np.ones((source_points_3d.shape[0], 1))]).T).T[:,
        #                              :3]

        #     # 获取当前人体关节位置
        #     with torch.no_grad():
        #         output = self.model(betas=shape,
        #                             body_pose=body_pose,
        #                             jaw_pose=zero_pose,
        #                             leye_pose=zero_pose,
        #                             reye_pose=zero_pose,
        #                             global_orient=global_orient,
        #                             expression=torch.zeros((1, 10)).float().to(self.device),
        #                             transl=transl)
        #         joints = output.joints[0]
        #         pelvis = joints[0]

        #     # 为每个目标关节设置新的目标位置
        #     target_joints_dict = {}
        #     for joint_idx, corresp_pos in target_joints.items():
        #         target_world_pos = torch.tensor(transformed_obj_points[corresp_pos],
        #                                         device=self.device, dtype=torch.float32)
        #         target_offset = target_world_pos - pelvis
        #         target_joints_dict[joint_idx] = target_offset

        #     # 执行IK优化
        #     print("Starting IK optimization...")
        #     global_orient_new, body_pose_new = self.run_joint_ik(
        #         global_orient, body_pose, shape, transl,
        #         target_joints_dict=target_joints_dict,
        #         constraint_joints_list=constraint_joints,
        #         max_iter=10, lr=1.0
        #     )

        #     # 保存IK后的人体网格
        #     if save_meshes:
        #         output = self.model(betas=shape,
        #                             body_pose=body_pose_new,
        #                             jaw_pose=zero_pose,
        #                             leye_pose=zero_pose,
        #                             reye_pose=zero_pose,
        #                             global_orient=global_orient_new,
        #                             expression=torch.zeros((1, 10)).float().to(self.device),
        #                             transl=transl)

        #         vertices_after_ik = output.vertices[0].detach().cpu().numpy()
        #         self.save_mesh_as_obj(vertices_after_ik, self.model.faces, "human_after_ik.obj")

        #     print("HOI solving completed with IK optimization!")
        #     time4 = time.time()
        #     print(f"IK optimization time: {time4 - time3}")
        #     return {
        #         'global_orient': global_orient_new,
        #         'body_pose': body_pose_new,
        #         'icp_transform_matrix': transform_matrix,
        #         'optimized_joints': list(target_joints.keys()),
        #     }
        # else:
            # print("No limb joints found for IK optimization. Only ICP alignment performed.")
        return {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'icp_transform_matrix': transform_matrix,
            'optimized_joints': [],
        }
        
