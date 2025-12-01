import json
import os
import numpy as np
from tqdm import tqdm

from .optimizer_part import VideoBodyObjectOptimizer
from .hoi_solver import HOISolver
from .kp_use import (
    model,
    resource_path,
    apply_initial_transform_to_mesh,
    apply_initial_transform_to_points,
)
from copy import deepcopy
def kp_use_new(
    output,
    hand_poses,
    body_poses,
    global_body_poses,
    obj_orgs,
    sampled_orgs,
    centers_depth,
    human_part,
    K,
    start_frame,
    end_frame,
    video_dir,
    is_static_object=False,
    kp_record_path: str = None
):
    if kp_record_path is None or not os.path.exists(kp_record_path):
        raise FileNotFoundError(f"kp_record_path not found: {kp_record_path}")
    with open(kp_record_path, "r", encoding="utf-8") as f:
        merged = json.load(f)
    body_params=body_poses
    global_body_params=global_body_poses
    seq_length = end_frame - start_frame
    best_frame = 0
    if is_static_object:
        max_count = -1
        for i in range(seq_length):
            frame_id = start_frame + i
            key = f"{frame_id:05d}"
            annotation = merged.get(key, {"2D_keypoint": []})
            num_2d = len(annotation.get("2D_keypoint", []))
            num_3d = 0
            for k in annotation.keys():
                if k in ("2D_keypoint", "multiview_2d_keypoints", "multiview_cam_params"):
                    continue
                num_3d += 1
            total = num_2d + num_3d
            if total > max_count:
                max_count = total
                best_frame = i

    object_points_idx = []
    body_points_idx = []
    pairs_2d = []
    object_points = []
    image_points = []
    body_kp_name = []
    hoi_solver = HOISolver(model_folder=resource_path('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'))

    for i in tqdm(range(seq_length)):
        frame_id = start_frame + i
        key = f"{frame_id:05d}"
        annotation = merged.get(key, {"2D_keypoint": []})

        if annotation.get("2D_keypoint"):
            current_idx = best_frame if is_static_object else i
            point_indices = [p[0] for p in annotation["2D_keypoint"]]
            image_coords = [np.array(p[1]) for p in annotation["2D_keypoint"]]
            object_verts = np.array(deepcopy(obj_orgs[current_idx].vertices))[point_indices]
            transformed_verts = apply_initial_transform_to_points(
                object_verts, centers_depth[current_idx + start_frame]
            )                              

            object_points.append(transformed_verts.astype(np.float32))
            image_points.append(np.array(image_coords, dtype=np.float32))
        else:
            object_points.append(np.array([]))
            image_points.append(np.array([]))
        object_idx = np.zeros((74, 2))
        for k, annot_index in annotation.items():
            if k in ("2D_keypoint", "multiview_2d_keypoints", "multiview_cam_params"):
                continue
            body_kp_name.append(k)
            human_part_index = list(human_part.keys()).index(k)
            object_idx[human_part_index] = [annot_index, 1]

        pairs_2d.append(annotation.get("2D_keypoint", []))
        body_idx = [v['index'] for v in human_part.values()]
        object_points_idx.append(object_idx)
        body_points_idx.append(body_idx)

    hoi_interval = 1
    if is_static_object:
        frames_to_optimize = [best_frame]
    else:
        frames_to_optimize = list(range(0, seq_length, hoi_interval))
        if frames_to_optimize[-1] != seq_length - 1:
            frames_to_optimize.append(seq_length - 1)

    optimized_results = {}
    icp_transform_matrix = []
    joint_mapping = json.load(open(resource_path('video_optimizer/data/joint_reflect.json')))



    for i in frames_to_optimize:
                                                                                                               
        obj_src_idx = best_frame if is_static_object else i
        obj_init = apply_initial_transform_to_mesh(
            obj_orgs[obj_src_idx], centers_depth[obj_src_idx + start_frame]
        )
        obj_init_sample = apply_initial_transform_to_mesh(
            sampled_orgs[obj_src_idx], centers_depth[obj_src_idx + start_frame]
        )
        result = hoi_solver.solve_hoi(
            obj_init,
            obj_init_sample,
            body_params,
            global_body_params,
            i,
            start_frame,
            end_frame,
            hand_poses,
            object_points_idx,
            body_points_idx,
            object_points,
            image_points,
            joint_mapping,
            K=K.cpu().numpy() if hasattr(K, "cpu") else K,
            is_multiview=False
        )
        body_params['global_orient'][i + start_frame] = result['global_orient'].detach().cpu()
        body_params['body_pose'][i + start_frame] = result['body_pose'].detach().cpu()
        icp_transform_matrix.append(result['icp_transform_matrix'])



    if is_static_object:
                                                              
        if len(icp_transform_matrix) > 0:
            icp_transform_matrix = [icp_transform_matrix[0] for _ in range(seq_length)]
        first_frame_obj = obj_orgs[best_frame]
        first_frame_sampled = sampled_orgs[best_frame]
        for i in range(seq_length):
            obj_orgs[i] = first_frame_obj
            sampled_orgs[i] = first_frame_sampled

    optimizer_args = {
        "body_params": body_params,
        "global_body_params": global_body_params,
        "hand_params": hand_poses,
        "object_points_idx": object_points_idx,
        "body_points_idx": body_points_idx,
        "body_kp_name" : body_kp_name,
        "pairs_2d": pairs_2d,
        "object_meshes": obj_orgs,
        "sampled_obj_meshes": sampled_orgs,
        "centers_depth": centers_depth,
        "icp_transform_matrix": icp_transform_matrix,
        "smpl_model": model,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "video_dir": video_dir,
        "lr": 0.1,
        "is_static_object": is_static_object,
        "best_frame": best_frame if is_static_object else None,
    }
    optimizer = VideoBodyObjectOptimizer(**optimizer_args)        
    optimizer.optimize(steps=30, print_every=5)
    optimized_params = optimizer.get_optimized_parameters()
    optimizer.create_visualization_video(
        os.path.join(video_dir, "optimized_frames"),
        K=K,
        video_path=os.path.join(video_dir, "optimize_video.mp4"),
        clear=False
    )
    body_params, hand_poses, R_finals, t_finals = optimizer.get_optimize_result()
    return body_params, hand_poses, R_finals, t_finals, optimized_params
