from sdf import SDF
import torch
import json
import os
import sys
import numpy as np
import trimesh
import copy
import argparse
import time
from .optimizer_part import VideoBodyObjectOptimizer
import open3d as o3d
import smplx
from .utils.hoi_utils import load_transformation_matrix, update_hand_pose,icp_process
from copy import deepcopy
from probreg import cpd
from tqdm import tqdm
from .hoi_solver import HOISolver
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
def resource_path(relative_path):
    try:
                               
        base_path = sys._MEIPASS
    except Exception:
                  
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)                                                                                  
model_type = 'smplx'
model_folder = resource_path("video_optimizer/smpl_models/SMPLX_NEUTRAL.npz")
model = smplx.create(model_folder, model_type=model_type,
                            gender='neutral',
                            num_betas=10,
                            num_expression_coeffs=10,
                            use_pca=False, 
                            flat_hand_mean=True).cuda()
def apply_initial_transform_to_mesh(mesh,t):
    mesh_copy = copy.deepcopy(mesh)
    verts = np.asarray(mesh_copy.vertices)
    transformed_verts = verts + t
    mesh_copy.vertices = o3d.utility.Vector3dVector(transformed_verts)
    return mesh_copy

def apply_initial_transform_to_points(points, t):
    return points+ t



def kp_use(output, hand_poses, obj_orgs, sampled_orgs, initial_R,
           centers, human_part, K, start_frame, end_frame, joints_to_optimize, video_dir, is_static_object=False):        
    body_params = output["smpl_params_incam"]
    global_body_params = output["smpl_params_global"]
    kp_files = sorted(os.listdir(os.path.join(video_dir, 'kp_record')), key=lambda x: int(x.split('.')[0]))
    seq_length = end_frame - start_frame
    object_points_idx = []
    body_points_idx = []
    pairs_2d = []
    object_points = []
    image_points = []
    hoi_solver= HOISolver(model_folder=resource_path('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'))

    for i, file in tqdm(enumerate(kp_files[start_frame:end_frame])):
        annotation = json.load(open(os.path.join(video_dir, 'kp_record', file)))
        if annotation["2D_keypoint"]:
            current_idx = 0 if is_static_object else i
            
            point_indices = [p[0] for p in annotation["2D_keypoint"]]
            image_coords = [np.array(p[1]) for p in annotation["2D_keypoint"]]
            
            object_verts = np.array(obj_orgs[current_idx].vertices)[point_indices]
            transformed_verts = apply_initial_transform_to_points(object_verts, initial_R[current_idx], centers[current_idx])
            
            object_points.append(transformed_verts.astype(np.float32))
            image_points.append(np.array(image_coords, dtype=np.float32))
        else:
            object_points.append(np.array([]))
            image_points.append(np.array([]))

        object_idx = np.zeros((74, 2))
        for k, annot_index in annotation.items():
            if k=="2D_keypoint" or k=="multiview_2d_keypoints" or k=="multiview_cam_params":
                continue
            human_part_index = list(human_part.keys()).index(k)
            object_idx[human_part_index] = [annot_index, 1]
        pairs_2d.append(annotation["2D_keypoint"])  
        body_idx = [v['index'] for k, v in human_part.items()]
        object_points_idx.append(object_idx)
        body_points_idx.append(body_idx)
    hoi_interval = 1
    frames_to_optimize = list(range(0, seq_length, hoi_interval))
    if frames_to_optimize[-1] != seq_length:
        frames_to_optimize.append(seq_length-1) 
    optimized_results = {}
    transform_matrix = []
    joint_mapping=json.load(open(resource_path('video_optimizer/data/joint_reflect.json')))
    
    for i in frames_to_optimize:
        if is_static_object:
            i = 0
        obj_init = apply_initial_transform_to_mesh(obj_orgs[i], initial_R[i], centers[i])
        obj_init_sample = apply_initial_transform_to_mesh(sampled_orgs[i], initial_R[i], centers[i])
        
        result=hoi_solver.solve_hoi(obj_init,obj_init_sample,body_params, global_body_params, i, start_frame,end_frame,
                                    hand_poses,object_points_idx,body_points_idx,object_points,image_points,joint_mapping, K=K.cpu().numpy(), is_multiview=False)
        body_params['global_orient'][i+ start_frame] = result['global_orient'].detach().cpu()
        body_params['body_pose'][i+ start_frame] = result['body_pose'].detach().cpu()
        transform_matrix.append(result['transform_matrix'])

    if is_static_object:
        first_frame_obj = obj_orgs[0]
        first_frame_sampled = sampled_orgs[0]
        for i in range(seq_length):
            obj_orgs[i] = first_frame_obj
            sampled_orgs[i] = first_frame_sampled

                                 
    optimizer_args = {"body_params":body_params,
                 "global_body_params":global_body_params,
                 "hand_params":hand_poses,  
                 "object_points_idx":object_points_idx,   
                 "body_points_idx":body_points_idx, 
                 "pairs_2d":pairs_2d, 
                 "object_meshes":obj_orgs, 
                 "sampled_obj_meshes":sampled_orgs, 
                 "initial_R":initial_R,
                 "initial_t":centers,
                 "transform_matrix":transform_matrix,
                 "smpl_model":model,
                 "start_frame":start_frame,
                 "end_frame":end_frame,
                 "video_dir":video_dir,  
                 "lr":0.1,
                 "is_static_object":is_static_object,}
    optimizer = VideoBodyObjectOptimizer(**optimizer_args)
    joints_to_optimize = joints_to_optimize.split(',')
    optimizer.set_mask(joints_to_optimize)
    optimizer.optimize(steps=25, print_every=5, optimize_per_frame=True)
    optimized_params= optimizer.get_optimized_parameters()
    optimizer.create_visualization_video(os.path.join(video_dir, "optimized_frames"), K=K,
                                          video_path=os.path.join(video_dir, "optimize_video.mp4"), clear=False)
    body_params, hand_poses, R_finals, t_finals = optimizer.get_optimize_result()                           
    return body_params, hand_poses, R_finals, t_finals, optimized_params
def kp_use_multiview(output, hand_poses, obj_orgs, sampled_orgs, initial_R,
                     centers, human_part, K, start_frame, end_frame, 
                     joints_to_optimize, video_dir, is_static_object=False):

    body_params = output["smpl_params_incam"]
    global_body_params = output["smpl_params_global"]
    kp_files = sorted(os.listdir(os.path.join(video_dir, 'kp_record')), key=lambda x: int(x.split('.')[0]))
    seq_length = end_frame - start_frame
    object_points_idx = []
    body_points_idx = []
    pairs_2d = []
    object_points = []
    image_points = []
    all_mutiview_object_points = []
    all_mutiview_image_points = []
    all_mutiview_cam = []
    hoi_solver= HOISolver(model_folder=resource_path('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'))

    for i, file in tqdm(enumerate(kp_files[start_frame:end_frame])):
        annotation = json.load(open(os.path.join(video_dir, 'kp_record', file)))
        if annotation["2D_keypoint"]:
            current_idx = 0 if is_static_object else i
            
            point_indices = [p[0] for p in annotation["2D_keypoint"]]
            image_coords = [np.array(p[1]) for p in annotation["2D_keypoint"]]
            
            object_verts = np.array(obj_orgs[current_idx].vertices)[point_indices]
            transformed_verts = apply_initial_transform_to_points(object_verts, initial_R[current_idx], centers[current_idx])
            
            object_points.append(transformed_verts.astype(np.float32))
            image_points.append(np.array(image_coords, dtype=np.float32))


        mutiview_object_points = []
        mutiview_image_points = []
        mutiview_cam = []
        if "multiview_2d_keypoints" in annotation and annotation["multiview_2d_keypoints"]:
            for angle, data in annotation["multiview_2d_keypoints"].items():
                keypoints_for_this_angle = data.get("keypoints", [])
                cam_params_for_this_angle = data.get("cam_params")
                angle_3d_points = []
                angle_2d_points = []

                if keypoints_for_this_angle and cam_params_for_this_angle:
                    mutiview_cam.append(cam_params_for_this_angle)
                    
                    current_idx = 0 if is_static_object else i
                    
                    point_indices = [p[0] for p in keypoints_for_this_angle]
                    object_verts = np.array(obj_orgs[current_idx].vertices)[point_indices]
                    transformed_verts = apply_initial_transform_to_points(object_verts, initial_R[current_idx], centers[current_idx])
                    
                    angle_3d_points = list(transformed_verts)
                    angle_2d_points = [np.array(p[1]) for p in keypoints_for_this_angle]

                mutiview_object_points.append(angle_3d_points)
                mutiview_image_points.append(angle_2d_points)
        mutiview_object_points = np.asarray(mutiview_object_points)
        mutiview_image_points = np.asarray(mutiview_image_points)
        object_idx = np.zeros((74, 2))
        for k, annot_index in annotation.items():
            if k=="2D_keypoint" or k=="multiview_2d_keypoints":
                continue
            human_part_index = list(human_part.keys()).index(k)
            object_idx[human_part_index] = [annot_index, 1]
        all_mutiview_object_points.append(mutiview_object_points)
        all_mutiview_image_points.append(mutiview_image_points)
        all_mutiview_cam.append(mutiview_cam)
        pairs_2d.append(annotation["2D_keypoint"])  
        body_idx = [v['index'] for k, v in human_part.items()]
        object_points_idx.append(object_idx)
        body_points_idx.append(body_idx)

                                          
                                                             
                                                            
                                                                     
                 
    hoi_interval = 1
    frames_to_optimize = list(range(0, seq_length, hoi_interval))
    if frames_to_optimize[-1] != seq_length:
        frames_to_optimize.append(seq_length-1) 

    optimized_results = {}
    transform_matrix = []
    joint_mapping=json.load(open(resource_path('video_optimizer/data/joint_reflect.json')))
    
    for i in frames_to_optimize:
        if is_static_object:
            i = 0
    for i in frames_to_optimize:
        if is_static_object:
            i = 0
        obj_init = apply_initial_transform_to_mesh(obj_orgs[i], initial_R[i], centers[i])
        obj_init_sample = apply_initial_transform_to_mesh(sampled_orgs[i], initial_R[i], centers[i])
        
        all_mutiview_info = (all_mutiview_object_points[i], all_mutiview_image_points[i], all_mutiview_cam[i])
        result=hoi_solver.solve_hoi(obj_init,obj_init_sample,body_params, global_body_params, i,start_frame,end_frame,
                                    hand_poses,object_points_idx,body_points_idx,object_points,image_points, joint_mapping, K=None,
                                    all_mutiview_info=all_mutiview_info, is_multiview=True)
        body_params['global_orient'][i+ start_frame] = result['global_orient'].detach().cpu()
        body_params['body_pose'][i+ start_frame] = result['body_pose'].detach().cpu()
        transform_matrix.append(result['transform_matrix'])

    if is_static_object:
        first_frame_obj = obj_orgs[0]
        first_frame_sampled = sampled_orgs[0]
        for i in range(seq_length):
            obj_orgs[i] = first_frame_obj
            sampled_orgs[i] = first_frame_sampled

    optimizer_args = {"body_params":body_params,
                 "global_body_params":global_body_params,
                 "hand_params":hand_poses,  
                 "object_points_idx":object_points_idx,   
                 "body_points_idx":body_points_idx, 
                 "pairs_2d":pairs_2d, 
                 "object_meshes":obj_orgs, 
                 "sampled_obj_meshes":sampled_orgs, 
                 "initial_R":initial_R,
                 "initial_t":centers,
                 "transform_matrix":transform_matrix,
                 "smpl_model":model,
                 "start_frame":start_frame,
                 "end_frame":end_frame,
                 "video_dir":video_dir,  
                 "lr":0.1,
                 "is_static_object":is_static_object}
    
    temp_optimizer = VideoBodyObjectOptimizer(**optimizer_args)
    
    K_vis = torch.from_numpy(np.array(K)).float().cuda()

    temp_optimizer.create_visualization_video(
        os.path.join(video_dir, "optimized_frames"), K=K_vis,
        video_path=os.path.join(video_dir, "optimize_video.mp4"), clear=False
    )

    optimized_params= temp_optimizer.get_optimized_parameters()
    body_params, hand_poses, R_finals, t_finals = temp_optimizer.get_optimize_result()

    return body_params, hand_poses, R_finals, t_finals, optimized_params


                            
                                                                              
                                                                                                               
                                                                          
                                                                            
                                                                   
                                             
                                                                 
                                           
                                
                
