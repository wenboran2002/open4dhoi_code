import argparse
import os
import json
import sys
import cv2
import numpy as np
import torch
# torch.cuda.init()
import open3d as o3d
from copy import deepcopy

from video_optimizer.kp_use_new import kp_use_new
from video_optimizer.utils.hoi_utils import update_hand_pose
from video_optimizer.utils.parameter_transform import transform_and_save_parameters
from video_optimizer.utils.dataset_util import (
    get_records_by_annotation_progress,
    update_record_annotation_progress,
    get_static_flag_from_merged,
    validate_record_for_optimization
)
from pathlib import Path

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def preprocess_obj_sample(obj_org, object_poses, seq_length):
    centers = np.array(object_poses['center'])
    obj_orgs, center_objs = [], []
    for i in range(seq_length):
        obj_pcd = deepcopy(obj_org)
        new_overts = np.asarray(obj_pcd.vertices)
        if object_poses['scale'] == 0:
            object_poses['scale'] = 1.0
        new_overts *= object_poses['scale']
        center_objs.append(np.mean(deepcopy(new_overts), axis=0))
        new_overts = new_overts - np.mean(new_overts, axis=0)
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)
    return obj_orgs, centers, center_objs

def preprocess_obj(obj_org, object_poses, seq_length, center_obj):
    centers = np.array(object_poses['center'])
    obj_orgs = []
    for i in range(seq_length):
        obj_pcd = deepcopy(obj_org)

        new_overts = np.asarray(obj_pcd.vertices)
        if object_poses['scale'] == 0:
            object_poses['scale'] = 1.0
        new_overts *= object_poses['scale']
        new_overts = new_overts - center_obj[i]
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)
    return obj_orgs, centers

def find_last_annotated_frame(kp_dir: str) -> int:
    if not os.path.isdir(kp_dir):
        return -1
    frames = []
    for fname in os.listdir(kp_dir):
        if fname.endswith(".json") and fname[:5].isdigit():
            frames.append(int(fname[:5]))
    return max(frames) if frames else -1

def validate_dof(kp_dir: str, start_frame: int, end_frame: int) -> bool:
    invalid = []
    for frame_idx in range(start_frame, end_frame + 1):
        fpath = os.path.join(kp_dir, f"{frame_idx:05d}.json")
        if not os.path.exists(fpath):
            invalid.append((frame_idx, 0, 0, 0))
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            invalid.append((frame_idx, 0, 0, 0))
            continue
        num_2d = len(data.get("2D_keypoint", []) or [])
        num_3d = len([k for k in data.keys() if k != "2D_keypoint"])
        dof = 3 * num_3d + 2 * num_2d
        if dof < 6:
            invalid.append((frame_idx, dof, num_3d, num_2d))
    if invalid:
        print("Not enough DoF (>=6：")
        for frame_idx, dof, n3, n2 in invalid[:20]:
            print(f"  Frame {frame_idx}: DoF={dof} (3D={n3}×3, 2D={n2}×2)")
        if len(invalid) > 20:
            print(f"... and {len(invalid) - 20} frames")
        return False
    return True


def validate_dof_from_merged(merged: dict, start_frame: int, end_frame: int) -> bool:
    invalid = []
    for frame_idx in range(start_frame, end_frame + 1):
        key = f"{frame_idx:05d}"
        data = merged.get(key)
        if not isinstance(data, dict):
            invalid.append((frame_idx, 0, 0, 0))
            continue
        num_2d = len(data.get("2D_keypoint", []) or [])
        num_3d = len([k for k in data.keys() if k != "2D_keypoint"])
        dof = 3 * num_3d + 2 * num_2d
        if dof < 6:
            invalid.append((frame_idx, dof, num_3d, num_2d))
    if invalid:
        print("Not enough DoF (>=6：")
        for frame_idx, dof, n3, n2 in invalid[:20]:
            print(f"  Frame {frame_idx}: DoF={dof} (3D={n3}×3, 2D={n2}×2)")
        if len(invalid) > 20:
            print(f"... and {len(invalid) - 20} frames")
        return False
    return True

def assemble_preview_video(optimized_dir: str, output_path: str, fps: float = 18.0) -> None:
    frame_files = sorted(
        [os.path.join(optimized_dir, f) for f in os.listdir(optimized_dir) if f.endswith(".png")]
    )
    if not frame_files:
        print("No frames found for video")
        return
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print("Cannot read first frame, cancel video")
        return
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for ff in frame_files:
        fr = cv2.imread(ff)
        if fr is not None:
            out.write(fr)
    out.release()
    print(f"Preview video saved to: {output_path}")


def optimize_single_record(record):
    video_dir = record
    print(f"Video directory: {video_dir}")
    if os.path.exists(Path(video_dir, "video0.mp4")):
        video_path = Path(video_dir, "video0.mp4")
        cap = cv2.VideoCapture(str(video_path))
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        with open(Path(video_dir, "output", "obj_poses.json"), 'r', encoding='utf-8') as f:
            obj_poses = json.load(f)
        json_len = len(obj_poses['center'])
        if video_len != json_len:
            video_path = Path(video_dir, "video.mp4")
        else:
            video_path = Path(video_dir, "video0.mp4")
    else:
        video_path = Path(video_dir, "video.mp4")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    if total_frames <= 0:
        print("Cannot read video or video is empty")
        return False

    merged_path = os.path.join(video_dir, f"kp_record_merged.json")
    if not os.path.exists(merged_path):
        print(f"Cannot find merged annotation file: {merged_path}, please save it in the annotation tool first.")
        return False
    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)
    try:
        object_scale = float(merged.get("object_scale", 1.0))
    except Exception:
        object_scale = 1.0
    static_object = get_static_flag_from_merged(merged_path)
    print(f"Static object: {static_object}")
    try:
        keys_int = sorted(int(k) for k in merged.keys() if str(k).isdigit())
    except Exception:
        keys_int = []
    if not keys_int:
        print("Merged annotation is empty, terminate.")
        return False
    try:
        start_frame = merged["start_frame_index"]
    except:
        start_frame = 0
    end_frame = max(keys_int)
    print(f"Optimized frame range: {start_frame}..{end_frame}")
    if not validate_dof_from_merged(merged, start_frame, end_frame):
        print("Invalid annotation frames, terminate optimization")
        return False
    output = torch.load(os.path.join(video_dir, "motion", "result.pt"))
    body_params = output["smpl_params_incam"]
    global_body_params = output["smpl_params_global"]
    K = output['K_fullimg'][0]
    human_part = json.load(open(resource_path("video_optimizer/data/part_kp.json"), "r", encoding="utf-8"))
    hand_pose_path = os.path.join(video_dir, 'motion', 'hand_pose.json')
    if os.path.exists(hand_pose_path):
        hand_poses = json.load(open(hand_pose_path, "r", encoding="utf-8"))
    else:
        hand_poses = {}
    for i in range(total_frames):
        if str(i) not in hand_poses:
            hand_poses[str(i)] = {}
        body_params["body_pose"][i], hand_poses[str(i)]["left_hand"], hand_poses[str(i)]["right_hand"] = \
            update_hand_pose(hand_poses, body_params["global_orient"], body_params["body_pose"], i)
    print(len(hand_poses), "frames of hand poses loaded/updated.",len(body_params["body_pose"]))
    obj_path = os.path.join(video_dir, "obj_org.obj")
    if not os.path.exists(obj_path):
        print(f"Cannot find object model: {obj_path}")
        return False
    obj_org = o3d.io.read_triangle_mesh(obj_path)
    sampled_obj = obj_org.simplify_quadric_decimation(target_number_of_triangles=1000)
    obj_poses_path = os.path.join(video_dir, 'align', 'obj_poses.json')
    if not os.path.exists(obj_poses_path):
        print(f"Cannot find object poses: {obj_poses_path}")
        return False
    with open(obj_poses_path, "r", encoding="utf-8") as f:
        object_poses = json.load(f)
    seq_len=end_frame - start_frame + 1
    obj_orgs, t_finals, center_objs = preprocess_obj_sample(obj_org, object_poses, seq_length=seq_len)
    sampled_orgs, _, _ = preprocess_obj_sample(sampled_obj, object_poses, seq_length=seq_len)
    def _scale_mesh_inplace(meshes, s):
        for mesh in meshes:
            v = np.asarray(mesh.vertices)
            if v.size == 0:
                return
            c = v.mean(axis=0)
            v_scaled = (v - c) * s + c
            mesh.vertices = o3d.utility.Vector3dVector(v_scaled)
    _scale_mesh_inplace(obj_orgs, object_scale)
    _scale_mesh_inplace(sampled_orgs, object_scale)
    body_params_new, hand_poses_new, Rf_seg, tf_seg, optimized_params = kp_use_new(
        output=output,
        hand_poses=hand_poses,
        body_poses=body_params,
        global_body_poses=global_body_params,
        obj_orgs=obj_orgs,
        sampled_orgs=sampled_orgs,
        centers_depth=t_finals,
        human_part=human_part,
        K=K,
        start_frame=start_frame,
        end_frame=end_frame, 
        video_dir=video_dir,
        is_static_object=static_object,
        kp_record_path=merged_path
    )
    print("Optimize finished")
    optimized_dir = os.path.join(video_dir, "optimized_frames")
    preview_path = os.path.join(video_dir, "optimized_preview.mp4")
    assemble_preview_video(optimized_dir, preview_path, fps=18.0)
    from datetime import datetime
    save_dir = os.path.join(video_dir, "final_optimized_parameters")
    os.makedirs(save_dir, exist_ok=True)
    def _to_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().tolist()
        if hasattr(x, "tolist"):
            return x.tolist()
        return x

    final_human_params = {
        "body_pose": {},
        "betas": {},
        "global_orient": {},
        "transl": {},
        "left_hand_pose": {},
        "right_hand_pose": {},
    }
    hp = hand_poses_new
    print('check', len(body_params_new["body_pose"]), len(hp), start_frame, end_frame)
    for fi in range(start_frame, end_frame):
        idx = fi
        print('save', idx)
        final_human_params["body_pose"][str(fi)] = _to_list(body_params_new["body_pose"][fi-start_frame])
        final_human_params["betas"][str(fi)] = _to_list(body_params_new["betas"][fi-start_frame])
        final_human_params["global_orient"][str(fi)] = _to_list(body_params_new["global_orient"][fi-start_frame])
        final_human_params["transl"][str(fi)] = _to_list(body_params_new["transl"][fi-start_frame])
        final_human_params["left_hand_pose"][str(fi)] = _to_list(hp[fi-start_frame]["left_hand"])
        final_human_params["right_hand_pose"][str(fi)] = _to_list(hp[fi-start_frame]["right_hand"])
    final_object_params = {
        "poses": {},
        "centers": {},
        "scale": float(object_poses.get("scale", 1.0))
    }
    for local_i, fi in enumerate(range(start_frame, end_frame)):
        final_object_params["poses"][str(fi)] = np.asarray(Rf_seg[local_i]).tolist()
        final_object_params["centers"][str(fi)] = np.asarray(tf_seg[local_i]).tolist()
    incam_subset = {
        "global_orient": [],
        "transl": []
    }
    global_subset = {
        "global_orient": [],
        "transl": []
    }
    for fi in range(start_frame, end_frame):
        incam_subset["global_orient"].append(_to_list(output["smpl_params_incam"]["global_orient"][fi]))
        incam_subset["transl"].append(_to_list(output["smpl_params_incam"]["transl"][fi]))
        global_subset["global_orient"].append(_to_list(output["smpl_params_global"]["global_orient"][fi]))
        global_subset["transl"].append(_to_list(output["smpl_params_global"]["transl"][fi]))
    original_object_path = os.path.join(video_dir, "obj_org.obj")
    saved_files = transform_and_save_parameters(
        human_params_dict=deepcopy(final_human_params),
        org_params=deepcopy(final_object_params),
        camera_params=deepcopy(output),
        output_dir=save_dir,
        original_object_path=original_object_path,
        user_scale=float(object_scale)
    )
    transformed_json_path = None
    transformed_mesh_path = None
    for p in saved_files:
        if isinstance(p, str) and os.path.basename(p).startswith("transformed_parameters_") and p.endswith(".json"):
            transformed_json_path = p
        if isinstance(p, str) and p.endswith(".obj"):
            transformed_mesh_path = p
    transformed_summary = {}
    if transformed_json_path and os.path.isfile(transformed_json_path):
        with open(transformed_json_path, "r", encoding="utf-8") as tf:
            transformed_summary["parameters"] = json.load(tf)
    if transformed_mesh_path:
        transformed_summary["object_mesh_path"] = transformed_mesh_path
    kp_records_merged = {f"{fi:05d}": merged.get(f"{fi:05d}", {"2D_keypoint": []}) for fi in range(start_frame, end_frame)}
    combined_payload = {
        "metadata": {
            "description": "All parameters in one file: raw/pre-transform, incam/global subsets, kp_records, and transformed summary",
            "frame_range": {"start": start_frame, "end": end_frame},
            "user_scale": float(object_scale),
            "save_dir": save_dir
        },
        "human_params_raw": final_human_params,
        "object_params_raw": final_object_params,
        "smpl_params_incam_subset": incam_subset,
        "smpl_params_global_subset": global_subset,
        "kp_records": kp_records_merged,
        "transformed": transformed_summary
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(save_dir, f"all_parameters_{timestamp}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved parameters for rendering: {combined_path}")
    summary = {
        "frame_range": {"start": start_frame, "end": end_frame},
        "static_object": static_object,
        "optimized_params_available": bool(optimized_params),
    }
    with open(os.path.join(video_dir, "optimize_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='folder containing final_optimized_parameters')
    args = parser.parse_args()
    optimize_single_record(args.data_dir)

if __name__ == "__main__":
    main()

