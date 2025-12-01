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

import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
from pytorch3d.transforms import axis_angle_to_matrix
import math
WARNED = False
def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list
def apply_transform_to_model(vertices, transform_matrix):
    homogenous_verts = np.hstack([vertices, np.ones((len(vertices), 1))])
    transformed = (transform_matrix @ homogenous_verts.T).T
    return transformed[:, :3] / transformed[:, [3]]


def rotate_camera(rotation_angle, rotation_axis):
    camera_rotation_rad = math.radians(rotation_angle)
    if rotation_axis.lower() == 'y':
        camera_self_rotation = np.array([
            [math.cos(camera_rotation_rad), 0, math.sin(camera_rotation_rad)],
            [0, 1, 0],
            [-math.sin(camera_rotation_rad), 0, math.cos(camera_rotation_rad)]
        ])
    elif rotation_axis.lower() == 'x':
        camera_self_rotation = np.array([
            [1, 0, 0],
            [0, math.cos(camera_rotation_rad), -math.sin(camera_rotation_rad)],
            [0, math.sin(camera_rotation_rad), math.cos(camera_rotation_rad)]
        ])
    elif rotation_axis.lower() == 'z':
        camera_self_rotation = np.array([
            [math.cos(camera_rotation_rad), -math.sin(camera_rotation_rad), 0],
            [math.sin(camera_rotation_rad), math.cos(camera_rotation_rad), 0],
            [0, 0, 1]
        ])
    return camera_self_rotation


def transform_to_global(R_incam, T_incam, incam_orient, global_orient, incam_pelvis, global_pelvis):
    R_old = axis_angle_to_matrix(incam_orient).squeeze(0)
    R_new = axis_angle_to_matrix(global_orient).squeeze(0)
    T_old = incam_pelvis.squeeze(0)
    T_new = global_pelvis.squeeze(0)
    R_delta = R_new @ R_old.T
    t_delta = T_new - (T_old @ R_delta.T)

    R_ind = R_delta
    t_ind = t_delta
    R_total = R_ind @ R_incam.float()
    T_total = T_incam.float() @ R_ind.T + t_ind.float()
    return R_total, T_total


def inverse_transform_to_incam(R_best_global, T_best_global, incam_orient, global_orient, incam_pelvis, global_pelvis):
    R_old = axis_angle_to_matrix(incam_orient).squeeze(0)
    R_new = axis_angle_to_matrix(global_orient).squeeze(0)
    T_old = incam_pelvis.squeeze(0)
    T_new = global_pelvis.squeeze(0)

    R_delta = R_new @ R_old.T
    t_delta = T_new - (T_old @ R_delta.T)

    R_incam = R_delta.T @ R_best_global.float()
    T_incam = (T_best_global.float() - t_delta.float()) @ R_delta

    return R_incam, T_incam


def compute_bounding_box(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    return min_coords, max_coords, center, size


def compute_camera_position(bbox_center, bbox_size, ground_plane='xz', distance_factor=3.0):
    if ground_plane == 'xz':
        camera_y = bbox_center[1]
        max_dimension = max(bbox_size[0], bbox_size[2])
        distance = max_dimension * distance_factor
        camera_x = bbox_center[0] + distance
        camera_z = bbox_center[2]
        camera_position = np.array([camera_x, camera_y, camera_z])
    return camera_position


def compute_camera_extrinsics(camera_position, target_position):
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 1, 0])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    rotation_matrix = np.array([
        right,
        -up,
        forward
    ])

    extrinsics = np.zeros((3, 4))
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = -rotation_matrix @ camera_position

    return extrinsics, rotation_matrix, camera_position

def compute_camera_intrinsics(bbox_size, image_width, image_height, fov_degrees=60.0):
    max_object_size = max(bbox_size)
    fov_y = np.radians(fov_degrees)
    focal_length_y = image_height / (2 * np.tan(fov_y / 2))
    focal_length_x = focal_length_y
    cx = image_width / 2
    cy = image_height / 2

    intrinsics = np.array([
        [focal_length_x, 0, cx],
        [0, focal_length_y, cy],
        [0, 0, 1]
    ])
    return intrinsics

def create_camera_for_object(vertices, image_width=800, image_height=600, ground_plane='xz', distance_factor=3.0, fov_degrees=60.0):
    min_coords, max_coords, center, size = compute_bounding_box(vertices)
    camera_position = compute_camera_position(center, size, ground_plane, distance_factor)
    extrinsics, rotation_matrix, camera_pos = compute_camera_extrinsics(camera_position, center)
    intrinsics = compute_camera_intrinsics(size, image_width, image_height, fov_degrees=fov_degrees)

    return {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'camera_position': camera_pos,
        'rotation_matrix': rotation_matrix,
        'target_position': center,
        'bounding_box': {
            'min': min_coords,
            'max': max_coords,
            'center': center,
            'size': size
        }
    }

