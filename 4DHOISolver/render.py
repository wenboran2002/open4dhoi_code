import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # specify GPU id
import glob
import json
import numpy as np
import torch
import smplx
import argparse
from PIL import Image
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
# renderer & utilities from project
from video_optimizer.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from video_optimizer.utils.vis.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from video_optimizer.utils.camera_utils import transform_to_global
import trimesh
from video_optimizer.utils.parameter_transform import apply_transform_to_smpl_params
from pytorch3d.transforms import axis_angle_to_matrix

def find_latest_file(dirpath, pattern):
    files = glob.glob(os.path.join(dirpath, pattern))
    return max(files, key=os.path.getmtime) if files else None

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def ensure_cuda(t):
    return t.cuda() if torch.cuda.is_available() else t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='folder containing final_optimized_parameters')
    parser.add_argument('--smpl_model', default='video_optimizer/smpl_models/SMPLX_NEUTRAL.npz')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--out', default='output_render.mp4')
    args = parser.parse_args()

    fop_dir = os.path.join(args.data_dir, 'final_optimized_parameters')
    if not os.path.isdir(fop_dir):
        raise FileNotFoundError(f"Not found: {fop_dir}")

    combined_json = find_latest_file(fop_dir, 'all_parameters_*.json') or find_latest_file(fop_dir, 'transformed_parameters_*.json')
    if not combined_json:
        raise FileNotFoundError("No combined or transformed JSON found in final_optimized_parameters")

    with open(combined_json, 'r', encoding='utf-8') as f:
        combined = json.load(f)
    human_raw = combined['human_params_raw']
    object_raw = combined['object_params_raw']
    incam_subset = combined['smpl_params_incam_subset']
    global_subset = combined['smpl_params_global_subset']
    transformed_section = combined['transformed']
    print(f"Frames in human raw: {len(human_raw.get('body_pose', {}))}, object raw: {len(object_raw.get('poses', {}))}, incam subset: {len(incam_subset.get('global_orient', {}))}, global subset: {len(global_subset.get('global_orient', {}))}")
    transformed_object_params = transformed_section['parameters']['object_params']
    mesh_path = transformed_section['object_mesh_path']
    if not mesh_path or not os.path.isfile(mesh_path):
        raise FileNotFoundError("object mesh (.obj) not found")
 
    # SMPL model
    smpl_model = smplx.create(
        args.smpl_model,
        model_type='smplx',
        gender='neutral',
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True
    )
    if torch.cuda.is_available():
        smpl_model = smpl_model.cuda()
    frame_keys = sorted(human_raw.get('body_pose', {}).keys(), key=lambda k: int(k))
    if not frame_keys:
        raise ValueError("No frames found in human params")
    obj_tm = trimesh.load(mesh_path, process=False)
    obj_vertices = np.asarray(obj_tm.vertices, dtype=np.float32)
    obj_faces = np.asarray(obj_tm.faces, dtype=np.int32)
    obj_colors = np.asarray(obj_tm.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0 if obj_tm.visual.vertex_colors is not None else None


    num_frames = len(frame_keys)
    transformed_human_params = {}
    transformed_object_params = {}
    transformed_human_params['body_pose'] = {}
    transformed_human_params['betas'] = {}
    transformed_human_params['global_orient'] = {}
    transformed_human_params['transl'] = {}
    transformed_human_params['left_hand_pose'] = {}
    transformed_human_params['right_hand_pose'] = {}
    transformed_object_params['R_total'] = {}
    transformed_object_params['T_total'] = {}
    human_vertices = []
    pelvises=[]
    org_pelvises=[]
    faces_human = None
    joints=[]

    motion_output=torch.load(os.path.join(args.data_dir, "motion", "result.pt"))
    global_body_params = motion_output["smpl_params_global"]
    incam_body_params = motion_output["smpl_params_incam"]

    for idx,fk in tqdm(enumerate(frame_keys), desc='SMPL forward'):
        bp = np.asarray(human_raw['body_pose'][fk], dtype=np.float32)
        betas = np.asarray(human_raw['betas'][fk], dtype=np.float32)
        left_hand = np.asarray(human_raw['left_hand_pose'][fk], dtype=np.float32)
        right_hand = np.asarray(human_raw['right_hand_pose'][fk], dtype=np.float32)
        glob_orient_org = np.asarray(human_raw['global_orient'][fk], dtype=np.float32)
        transl_org = np.asarray(human_raw['transl'][fk], dtype=np.float32)

        glob_o = torch.tensor(global_subset['global_orient'][idx], dtype=torch.float32)
        glob_t = torch.tensor(global_subset['transl'][idx], dtype=torch.float32)
        with torch.no_grad():
            bpt = ensure_cuda(torch.tensor(bp).view(1, -1))
            betat = ensure_cuda(torch.tensor(betas).view(1, -1))
            glt = ensure_cuda(torch.tensor(glob_o).view(1, 3))
            trt = ensure_cuda(torch.tensor(glob_t).view(1, 3))
            lht = ensure_cuda(torch.tensor(left_hand).view(1, -1))
            rht = ensure_cuda(torch.tensor(right_hand).view(1, -1))
            out = smpl_model(
                betas=betat,
                body_pose=bpt,
                left_hand_pose=lht,
                right_hand_pose=rht,
                jaw_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                leye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                reye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                global_orient=glt,
                expression=torch.zeros((1,10)).cuda() if torch.cuda.is_available() else torch.zeros((1,10)),
                transl=trt
            )
        verts = out.vertices[0].cpu().numpy()
        human_vertices.append(verts)
        pelvis=out.joints[:, 0, :]
        pelvises.append(pelvis)
        joints.append(out.joints[0])

        with torch.no_grad():
            bpt = ensure_cuda(torch.tensor(bp).view(1, -1))
            betat = ensure_cuda(torch.tensor(betas).view(1, -1))
            glt = ensure_cuda(torch.tensor(glob_orient_org).view(1, 3))
            trt = ensure_cuda(torch.tensor(transl_org).view(1, 3))
            lht = ensure_cuda(torch.tensor(left_hand).view(1, -1))
            rht = ensure_cuda(torch.tensor(right_hand).view(1, -1))
            out_org = smpl_model(
                betas=betat,
                body_pose=bpt,
                left_hand_pose=lht,
                right_hand_pose=rht,
                jaw_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                leye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                reye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                global_orient=glt,
                expression=torch.zeros((1,10)).cuda() if torch.cuda.is_available() else torch.zeros((1,10)),
                transl=trt
            )
        org_pelvis=out_org.joints[:, 0, :]
        org_pelvises.append(org_pelvis)
        if faces_human is None:
            faces_human = np.asarray(smpl_model.faces, dtype=np.int32)
        transformed_human_params['body_pose'][fk] = bp.tolist()
        transformed_human_params['betas'][fk] = betas.tolist()
        transformed_human_params['global_orient'][fk] = glob_o.tolist()
        transformed_human_params['transl'][fk] = glob_t.tolist()
        transformed_human_params['left_hand_pose'][fk] = left_hand.tolist()
        transformed_human_params['right_hand_pose'][fk] = right_hand.tolist()
    human_vertices_transformed = []
    object_vertices_per_frame = []
    for idx, fk in enumerate(frame_keys):
        hverts = human_vertices[idx]
        # object default verts
        overts = obj_vertices.copy()
        pelvis=pelvises[idx]
        org_pelvis=org_pelvises[idx]
        R_ = np.asarray(object_raw['poses'][fk], dtype=np.float32)
        R_axis_angle = torch.tensor(Rotation.from_matrix(R_).as_rotvec(), dtype=torch.float32)
        T = torch.tensor(object_raw['centers'][fk], dtype=torch.float32)
        T_np = np.asarray(object_raw['centers'][fk], dtype=np.float32)   # (3,)
        incam_o = torch.tensor(incam_subset['global_orient'][idx], dtype=torch.float32)
        incam_t = torch.tensor(incam_subset['transl'][idx], dtype=torch.float32)
        glob_o = torch.tensor(global_subset['global_orient'][idx], dtype=torch.float32)
        glob_t = torch.tensor(global_subset['transl'][idx], dtype=torch.float32)
        incam_params = (incam_o, incam_t)
        global_params = (glob_o, glob_t)
        new_o, new_t = apply_transform_to_smpl_params(
            R_axis_angle, T,
            (incam_o, org_pelvis), (glob_o, pelvis)
        )
        R_old = axis_angle_to_matrix(incam_o).squeeze(0)  # (3,3)
        R_new = axis_angle_to_matrix(glob_o).squeeze(0)  # (3,3)
        T_old = org_pelvis.detach().cpu().squeeze(0)
        T_new = pelvis.detach().cpu().squeeze(0)

        R_delta = R_new @ R_old.T
        t_delta = T_new - (T_old @ R_delta.T)

        R_ind = R_delta
        t_ind = t_delta
        R_total = R_ind @ torch.from_numpy(R_).float()
        T_total = torch.from_numpy(T_np).float() @ R_ind.T + t_ind
        overts = (obj_vertices @ R_total.cpu().numpy().T) + T_total.cpu().numpy()
        transformed_object_params['R_total'][fk] = R_total.tolist()
        transformed_object_params['T_total'][fk] = T_total.tolist()
        human_vertices_transformed.append(hverts.astype(np.float32))
        object_vertices_per_frame.append(overts.astype(np.float32))
    save_dict={
        'human_params_transformed': transformed_human_params,
        'object_params_transformed': transformed_object_params
    }
    save_path=os.path.join(args.data_dir, 'transformed_parameters_final.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=4)

    verts_human_t = torch.tensor(np.stack(human_vertices_transformed, axis=0), dtype=torch.float32).cuda()
    verts_object_t = torch.tensor(np.stack(object_vertices_per_frame, axis=0), dtype=torch.float32).cuda()
    faces_human_t = faces_human
    faces_obj = obj_faces
    K = torch.tensor([
        [1200.0, 0.0, args.width / 2],
        [0.0, 1200.0, args.height / 2],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    if torch.cuda.is_available():
        K = K.cuda()

    renderer = Renderer(args.width, args.height, device="cuda" if torch.cuda.is_available() else "cpu",
                        faces_human=faces_human_t, faces_obj=faces_obj, K=K)
    combined_verts = torch.cat([verts_human_t, verts_object_t], dim=1)
    verts_glob = combined_verts
    

    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.5,
        cam_height_degree=20,
        target_center_height=1.0,
        vec_rot=180
    )

    joints_glob = verts_human_t.mean(dim=1, keepdim=True)  # (F,1,3)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob.cpu())
    renderer.set_ground(scale * 1.5, cx, cz)
    temp_dir = os.path.join(args.data_dir, 'final_optimized_parameters', 'temp_frames_render')
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []
    color = torch.ones(3).float().cuda() * 0.8
    for i in tqdm(range(num_frames), desc='Rendering frames'):
        cams = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground_hoi(verts_human_t[i], verts_object_t[i], cams, global_lights, [0.8, 0.8, 0.8],
                                             obj_colors)
        img = np.clip(img, 0, 255).astype(np.uint8)
        path = os.path.join(temp_dir, f'frame_{i:04d}.png')
        Image.fromarray(img).save(path, optimize=False)
        frame_paths.append(path)
    import imageio
    writer = imageio.get_writer(os.path.join(args.data_dir, args.out), fps=args.fps, codec='libx264', quality=9)
    for p in tqdm(frame_paths, desc='Writing video'):
        img = imageio.imread(p)
        writer.append_data(img)
    writer.close()

    print("Render complete. Video saved to:", os.path.join(args.data_dir, args.out))


if __name__ == "__main__":
    main()
