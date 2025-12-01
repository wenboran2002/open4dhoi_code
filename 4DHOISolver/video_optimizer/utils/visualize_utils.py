import numpy as np
import open3d as o3d
import smplx
import cv2
import pyrender
import trimesh
model_type='smplx'
model_folder="./video_optimizer/smpl_models/SMPLX_NEUTRAL.npz"
layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False,
             'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False,
             'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
smplx_model = smplx.create(model_folder, model_type=model_type,
                     gender='neutral',
                     num_betas=10,
                     num_expression_coeffs=10, use_pca=False, use_face_contour=True, **layer_arg)
mesh_faces=np.asarray(smplx_model.faces)

def visualize_human_mesh_contact(pcds, contact_scores,img_name):
    region_score = contact_scores.detach().cpu().numpy().reshape(-1)
    # print(np.sum(region_score[:10475]))
    # print(region_score.shape)
    colors = np.zeros((len(pcds), 3))
    colors[region_score[:10475]] = np.asarray([1, 0, 0])
    human=o3d.geometry.TriangleMesh()
    human.vertices = o3d.utility.Vector3dVector(pcds.reshape(-1, 3))
    human.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    human.triangles = o3d.utility.Vector3iVector(mesh_faces)
    o3d.io.write_triangle_mesh(
        f"./output/test_output/{img_name}/contact/{img_name}_contact_org.obj",
        human)
def visualize_obj_mesh_contact(pcds,mesh, contact_scores,img_name):
    region_score = contact_scores.detach().cpu().numpy().reshape(-1)
    # print(np.sum(region_score[:10475]))
    # print(region_score.shape)
    colors = np.zeros((len(pcds), 3))
    colors[region_score[10475:]] = np.asarray([1, 0, 0])
    human=o3d.geometry.TriangleMesh()
    human.vertices = o3d.utility.Vector3dVector(pcds.reshape(-1, 3))
    human.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    human.triangles = o3d.utility.Vector3iVector(mesh)
    o3d.io.write_triangle_mesh(
        f"./output/test_output/{img_name}/contact/{img_name}_contact_org_obj.obj",
        human)
def visualize_mesh_opacity(pcds,mesh, scores,img_name,sufix='h'):
    scores=scores.detach().cpu().numpy().reshape(-1)
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
    scores_normalized = np.asarray(scores_normalized).reshape(-1)
    colors = np.zeros((len(scores_normalized), 3))

    colors[:, 0] = scores_normalized  # Red channel (increases with score)
    colors[:, 1] = 1 - scores_normalized  # Green channel (decreases with score)
    colors[:, 2] = 0  # Blue channel (not used here)
    human=o3d.geometry.TriangleMesh()
    human.vertices = o3d.utility.Vector3dVector(pcds.reshape(-1, 3))
    human.vertex_colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    if mesh is None:
        mf=mesh_faces
    else:
        mf=mesh
    human.triangles = o3d.utility.Vector3iVector(mf)
    o3d.io.write_triangle_mesh(
        f"./output/test_output/{img_name}/contact/{img_name}_opacity_{sufix}.obj",
        human)

def visualize_space(human_verts,object_verts,object_faces,img_name):
    human=o3d.geometry.TriangleMesh()
    human.vertices = o3d.utility.Vector3dVector(human_verts.reshape(-1, 3))
    human.triangles = o3d.utility.Vector3iVector(mesh_faces)
    object=o3d.geometry.TriangleMesh()
    object.vertices = o3d.utility.Vector3dVector(object_verts.reshape(-1, 3))
    object.triangles = o3d.utility.Vector3iVector(object_faces)
    align=human+object
    o3d.io.write_triangle_mesh(f"./output/test_output/{img_name}/{img_name}_space.obj",align)
    
def visualize_pcd_space(human_verts,object_verts,img_name):
    human_pcd=o3d.geometry.PointCloud()
    human_pcd.points = o3d.utility.Vector3dVector(human_verts.reshape(-1, 3))
    human_pcd.paint_uniform_color([1, 0, 0])
    object_pcd=o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_verts.reshape(-1, 3))
    object_pcd.paint_uniform_color([0, 0, 1])
    align=human_pcd+object_pcd
    o3d.io.write_point_cloud(f"./output/test_output/{img_name}/pcd/{img_name}_space.ply",align)
    o3d.io.write_point_cloud(f"./output/test_output/{img_name}/pcd/{img_name}_h_.ply",human_pcd)
    o3d.io.write_point_cloud(f"./output/test_output/{img_name}/pcd/{img_name}_o_.ply",object_pcd)

def visualize_imgs(render_pkg,viewpoint_cam,mask,mask_o,mask_h):
    image, image_o, image_h=render_pkg["render"],render_pkg["render_o"], render_pkg["render_h"]
    image=render_pkg["render"]
    image_test = image.permute(1, 2, 0).detach().cpu().numpy() * 255
    image_test = cv2.cvtColor(image_test, cv2.COLOR_RGB2BGR)

    image_test_o = image_o.permute(1, 2, 0).detach().cpu().numpy() * 255
    image_test_o = cv2.cvtColor(image_test_o, cv2.COLOR_RGB2BGR)

    image_test_h = image_h.permute(1, 2, 0).detach().cpu().numpy() * 255
    image_test_h = cv2.cvtColor(image_test_h, cv2.COLOR_RGB2BGR)
    # print(image.shape)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_render.png',
        image_test)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_render_o.png',
        image_test_o)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_render_h.png',
        image_test_h)

    gt_image = viewpoint_cam.original_image.cuda()
    gt_image_h = viewpoint_cam.original_image_h.cuda()
    gt_image_o = viewpoint_cam.original_image_o.cuda()

    bgr_image = gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255
    bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)

    bgr_image_o = gt_image_o.permute(1, 2, 0).detach().cpu().numpy() * 255
    bgr_image_o = cv2.cvtColor(bgr_image_o, cv2.COLOR_RGB2BGR)

    bgr_image_h = gt_image_h.permute(1, 2, 0).detach().cpu().numpy() * 255
    bgr_image_h = cv2.cvtColor(bgr_image_h, cv2.COLOR_RGB2BGR)

    mask_image= mask.permute(1, 2, 0).detach().cpu().numpy() * 255
    mask_image_o= mask_o.permute(1, 2, 0).detach().cpu().numpy() * 255
    mask_image_h= mask_h.permute(1, 2, 0).detach().cpu().numpy() * 255

    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_gt.png',
        bgr_image)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_gt_o.png',
        bgr_image_o)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_gt_h.png',
        bgr_image_h)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_mask.png',
        mask_image)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_mask_o.png',
        mask_image_o)
    cv2.imwrite(
        f'./output/test_output/{viewpoint_cam.image_name}/render/{viewpoint_cam.image_name}_mask_h.png',
        mask_image_h)


def visualize_mesh_projection(img,mesh,face,cam_param,img_name):
    # mesh
    # mesh = mesh.squeeze(0)
    img = img * 255
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    mesh = trimesh.Trimesh(mesh, face)

    # tmp=mesh.export('output_ply/'+name[:-4]+'%d.ply'%i)
    # tmp = mesh.export(path)
    # print('mesh', type(tmp))
    mesh.apply_transform(rot)

    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

    scene.add(mesh, 'mesh')
    # # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # # save to image

    # print(rgb.shape,)
    img_ = rgb * valid_mask + img * (1 - valid_mask)
    img = rgb * valid_mask

    cv2.imwrite(f'./output/test_output/{img_name}/render/{img_name}_project.png', img_)
    # return img, img_



from scipy.spatial.transform import Rotation as R


def create_ellipsoid_mesh(xyz, scaling, quaternion, color, resolution=10):
    """
    Create a 3D ellipsoid mesh using Open3D with Quaternion rotation and apply color.

    Parameters:
    xyz (tuple): Center of the ellipsoid (x, y, z).
    scaling (tuple): Scaling factors (σx, σy, σz) along each axis.
    quaternion (tuple): Quaternion (qx, qy, qz, qw) for rotation.
    color (tuple): RGB color as a tuple (r, g, b), where each component is in [0, 1].
    resolution (int): Resolution for mesh generation (higher values for finer mesh).

    Returns:
    open3d.geometry.TriangleMesh: The created ellipsoid mesh with color.
    """

    # Create a sphere mesh (as a basis for the ellipsoid)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)

    # Apply scaling to convert the sphere into an ellipsoid
    scaling_matrix = np.diag([scaling[0], scaling[1], scaling[2], 1])
    sphere.transform(scaling_matrix)

    # Apply rotation using the quaternion
    rot = R.from_quat(quaternion)  # Quaternion to rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rot.as_matrix()  # Set the 3x3 rotation part
    sphere.transform(rotation_matrix)

    # Apply translation (move ellipsoid to center xyz)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = np.array(xyz)
    sphere.transform(translation_matrix)

    # Assign color to each vertex of the ellipsoid
    colors = np.tile(color, (len(sphere.vertices), 1))  # Same color for all vertices
    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)

    return sphere


def save_ellipsoid_as_obj(xyz, scaling, rotation, filename='gaussian_ellipsoid.obj', resolution=50):
    """
    Create and save the ellipsoid as a .obj file.

    Parameters:
    xyz (tuple): Center of the ellipsoid (x, y, z).
    scaling (tuple): Scaling factors (σx, σy, σz) along each axis.
    rotation (tuple): Rotation angles in degrees (rx, ry, rz) around x, y, z axes.
    filename (str): Name of the file to save the ellipsoid (default: 'gaussian_ellipsoid.obj').
    resolution (int): Resolution for mesh generation (higher values for finer mesh).
    """

    # Create the ellipsoid mesh
    ellipsoid_mesh = create_ellipsoid_mesh(xyz, scaling, rotation, resolution)

    # Save the mesh as an .obj file
    o3d.io.write_triangle_mesh(filename, ellipsoid_mesh)
    print(f"Ellipsoid saved as {filename}")

