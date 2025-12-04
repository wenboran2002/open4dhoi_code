import tkinter as tk
import argparse
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import numpy as np
import torch

torch.cuda.init()
import open3d as o3d
import os
import sys
import smplx
import trimesh
import shutil
import ttkbootstrap as ttk_boot
from ttkbootstrap.constants import *

from ttkbootstrap.constants import PRIMARY, SUCCESS, INFO, SECONDARY, WARNING, DANGER

import threading
import time
from types import SimpleNamespace
import glob



from copy import deepcopy

# CoTracker imports
try:
    from cotracker.predictor import CoTrackerOnlinePredictor
    from cotracker.utils.visualizer import Visualizer
    import imageio.v3 as iio

    COTRACKER_AVAILABLE = True
    print("CoTracker Online is available")
except ImportError as e:
    COTRACKER_AVAILABLE = False
    print(f"Warning: CoTracker not available: {e}")
    print("Will fall back to interpolation for 2D keypoints")

EMOJI_FONT = "mincho"
DEFAULT_FONT = (EMOJI_FONT, 10)
TITLE_FONT = (EMOJI_FONT, 16, "bold")
SUBTITLE_FONT = (EMOJI_FONT, 12, "bold")
BUTTON_FONT = (EMOJI_FONT, 9)
INFO_FONT = (EMOJI_FONT, 10, "bold")


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


model_type = 'smplx'
model_folder = resource_path("./data/SMPLX_NEUTRAL.npz")
model = smplx.create(model_folder, model_type=model_type,
                     gender='neutral',
                     num_betas=10,
                     num_expression_coeffs=10,
                     use_pca=False,
                     flat_hand_mean=True)
from scipy.spatial.transform import Rotation as R
def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    calculating joints' global rotation
    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22,3)
    Returns:
        np.array: (3, 3)
    """
    global_rotation = np.eye(3)
    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation
def update_hand_pose(hand_poses,global_orient,body_params,frame_idx):
    M = np.diag([-1, 1, 1])

    body_pose = body_params[frame_idx].detach().cpu().numpy().reshape(1, -1)
    global_orient = global_orient[frame_idx].detach().cpu().numpy().reshape(1, 3)
    try:
        handpose=hand_poses[str(frame_idx)]
    except:
        return torch.from_numpy(body_pose), np.zeros(45), np.zeros(45)
    full_body_pose = np.concatenate(
        [global_orient.reshape(1, 3), body_pose.reshape(21, 3)], axis=0)
    left_elbow_global_rot = compute_global_rotation(full_body_pose, 18)  # left elbow IDX: 18
    right_elbow_global_rot = compute_global_rotation(full_body_pose, 19)  # left elbow IDX: 19

    if 'left_hand' in handpose:
        global_orient_hand_left = np.asarray(handpose["left_global_orient"]).reshape(3, 3)
        left_wrist_global_rot = M @ global_orient_hand_left @ M  # mirror switch
        left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot
        left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()
        body_pose[:, 57:60] = left_wrist_pose_vec
    # global_orient_hand_left=np.asarray(hand_poses[str(frame_idx)]["global_orient"][0]).reshape(3,3)
    # print(global_orient_hand_left.shape)
    # exit(0)
    if 'right_hand' in handpose:
        global_orient_hand_right = np.asarray(handpose["right_global_orient"]).reshape(3, 3)
        right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ global_orient_hand_right
        right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()
        body_pose[:, 60:63] = right_wrist_pose_vec


    left_hand_pose = np.zeros(45)
    right_hand_pose = np.zeros(45)
    for i in range(15):
        if 'left_hand' in handpose:
            left_finger_pose = M @ np.asarray(hand_poses[str(frame_idx)]["left_hand"])[
                i] @ M
            left_finger_pose_vec = R.from_matrix(left_finger_pose).as_rotvec()
            left_hand_pose[i * 3: i * 3 + 3] = left_finger_pose_vec
        if 'right_hand' in handpose:
            right_finger_pose = np.asarray(hand_poses[str(frame_idx)]["right_hand"][i])
            right_finger_pose_vec = R.from_matrix(right_finger_pose).as_rotvec()
            right_hand_pose[i * 3: i * 3 + 3] = right_finger_pose_vec

    return torch.from_numpy(body_pose), left_hand_pose, right_hand_pose

def apply_transform_to_model(vertices, transform_matrix):
    homogenous_verts = np.hstack([vertices, np.ones((len(vertices), 1))])

    transformed = (transform_matrix @ homogenous_verts.T).T
    return transformed[:, :3] / transformed[:, [3]]


def preprocess_obj_sample(obj_org, object_poses, orient_path, seq_length):
    centers = np.array(object_poses['center'])
    obj_orgs, center_objs = [], []
    for i in range(seq_length):
        obj_pcd = deepcopy(obj_org)
        if 'rotation' in object_poses:
            rotation_matrix = object_poses['rotation'][i]
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            obj_pcd.transform(transform_matrix)
        new_overts = np.asarray(obj_pcd.vertices)
        new_overts *= object_poses['scale']
        new_overts = new_overts - np.mean(new_overts, axis=0)
        center_objs.append(np.mean(new_overts, axis=0))
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)
    return obj_orgs, centers, center_objs


def preprocess_obj(obj_org, object_poses, orient_path, seq_length, center_obj):
    centers = np.array(object_poses['center'])
    obj_orgs = []
    for i in range(seq_length):
        obj_pcd = deepcopy(obj_org)
        if 'rotation' in object_poses:
            rotation_matrix = object_poses['rotation'][i]
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            obj_pcd.transform(transform_matrix)
        new_overts = np.asarray(obj_pcd.vertices)
        new_overts *= object_poses['scale']
        new_overts = new_overts - center_obj[i]
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)
    return obj_orgs, centers


class KeyPointApp:
    def __init__(self, root, args):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.title("KeyPoint Annotation Tool")

        self.style = ttk_boot.Style(theme="superhero")
        self.configure_emoji_fonts()

        self.obj_point = None
        self.kp_pair = {"2D_keypoint": []}
        self.human_joint_positions = {}
        self.annotated_frames = set()
        self.annotated_frames_2D = set()
        self.render_key_frame = set()
        self.last_frame = None
        self.last_frame_2D = None
        self.no_annot = True
        self.no_annote_2D = True
        self.selected_human_kp = None
        self.selected_2d_point = None
        self.is_static_object = tk.BooleanVar(value=False)


        self.tracked_points = {}
        self.tracking_active = False
        self.video_frames = None

        self.frame_keypoints_cache = {}  # {frame_idx: {"2D_keypoint": [...], ...}}

        self.update_timer_id = None
        self.is_app_closing = False

        self.cotracker_model = None
        self.video_tensor = None

        self.obj_orgs_base_vertices = None
        self.sampled_orgs_base_vertices = None

        self.load_config_files()
        self.setup_ui()
        self.load_data(args)

    def configure_emoji_fonts(self):
        self.style.configure("TLabel", font=DEFAULT_FONT)
        self.style.configure("TButton", font=BUTTON_FONT)
        self.style.configure("TCheckbutton", font=DEFAULT_FONT)
        self.style.configure("TLabelframe.Label", font=SUBTITLE_FONT)

        self.style.configure("Title.TLabel", font=TITLE_FONT)
        self.style.configure("Info.TLabel", font=INFO_FONT)

    def load_config_files(self):
        with open(resource_path("./data/part_kp.json"), "r") as file:
            self.all_joint = json.load(file)
        for joint, value in self.all_joint.items():
            self.human_joint_positions[joint] = value['point']

        with open(resource_path("./data/main_joint.json"), "r") as file:
            self.main_joint_coord = json.load(file)
        with open(resource_path("./data/joint_tree.json"), "r") as file:
            self.joint_tree = json.load(file)
        with open(resource_path("./data/button_name.json"), "r", encoding='utf-8') as file:
            self.button_name = json.load(file)

    def setup_ui(self):
        self.main_container = ttk_boot.Frame(self.root, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.left_container = ttk_boot.Frame(self.main_container)
        self.left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.top_frame = ttk_boot.Frame(self.left_container)
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.video_frame = ttk_boot.LabelFrame(self.top_frame, text="Target video", padding=15, bootstyle=PRIMARY)
        self.plot_frame = ttk_boot.LabelFrame(self.top_frame, text="3D model", padding=15, bootstyle=SUCCESS)
        self.human_image_frame = ttk_boot.LabelFrame(self.top_frame, text="Human keypoints", padding=15, bootstyle=INFO)

        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(5, 5))
        self.human_image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(5, 0))

        self.bottom_frame = ttk_boot.LabelFrame(self.left_container, text="Reference view", padding=10,
                                                bootstyle=SECONDARY)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(10, 0))

        self.setup_video_panel()
        self.setup_3d_control_panel()
        self.setup_human_panel()

    def setup_video_panel(self):

        self.video_display_frame = ttk_boot.Frame(self.video_frame)
        self.video_display_frame.pack(fill=tk.BOTH, expand=True)

        self.progress_frame = ttk_boot.Frame(self.video_frame)
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk_boot.Scale(
            self.progress_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            command=self.seek_video,
            bootstyle=PRIMARY
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.frame_label = ttk_boot.Label(
            self.progress_frame,
            text="0/0",
            font=INFO_FONT,
            bootstyle=PRIMARY
        )
        self.frame_label.pack(side=tk.RIGHT)

        self.controls_frame = ttk_boot.Frame(self.video_frame)
        self.controls_frame.pack(fill=tk.X, pady=(10, 0))

        self.play_button = ttk_boot.Button(
            self.controls_frame,
            text="Play",
            command=self.toggle_video,
            bootstyle="success",
            width=15
        )
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))

        self.static_checkbox = ttk_boot.Checkbutton(
            self.controls_frame,
            text="Static object",
            variable=self.is_static_object,
            bootstyle="success"
        )
        self.static_checkbox.pack(side=tk.RIGHT)

    def setup_3d_control_panel(self):
        self.button_frame = ttk_boot.Frame(self.plot_frame)
        self.button_frame.pack(fill=tk.X, pady=(0, 10))

        reset_btn = ttk_boot.Button(
            self.button_frame,
            text="Reset keypoints",
            command=self.reset_keypoints,
            bootstyle="outline-danger",
            width=20
        )
        reset_btn.pack(fill=tk.X, pady=2)

        select_3d_btn = ttk_boot.Button(
            self.button_frame,
            text="Select 3D point",
            command=self.open_o3d_viewer,
            bootstyle="primary",
            width=20
        )
        select_3d_btn.pack(fill=tk.X, pady=2)

        # Add scale check button
        scale_check_btn = ttk_boot.Button(
            self.button_frame,
            text="Check",
            command=self.open_scale_check_viewer,
            bootstyle="outline-primary",
            width=20
        )
        scale_check_btn.pack(fill=tk.X, pady=2)
        self.annotation_buttons = []
        buttons = [
            ("Select 2D point", self.keypoint_2D, "outline-info"),
            ("Start tracking", self.start_2d_tracking, "success"),
            ("Re-track from current", self.restart_tracking_from_current, "outline-success")
        ]
        for i, (text, command, style) in enumerate(buttons):
            btn = ttk_boot.Button(
                self.button_frame,
                text=text,
                command=command,
                bootstyle=style,
                width=20
            )
            btn.pack(fill=tk.X, pady=2)
            self.annotation_buttons.append(btn)
        save_btn = ttk_boot.Button(
            self.button_frame,
            text="Save annotations",
            command=self.save_kp_record_merged,
            bootstyle="outline-danger",
            width=20
        )
        save_btn.pack(fill=tk.X, pady=2)
        delete_btn = ttk_boot.Button(
            self.button_frame,
            text="Delete",
            command=self.mark_delete_and_next,
            bootstyle="danger",
            width=20
        )
        delete_btn.pack(fill=tk.X, pady=2)

        self.manage_button = ttk_boot.Button(
            self.button_frame,
            text="Manage keypoints",
            command=self.manage_existing_keypoints,
            bootstyle="outline-warning",
            width=20
        )
        self.manage_button.pack(fill=tk.X, pady=2)

        self.scale_frame = ttk_boot.Frame(self.plot_frame)
        self.scale_frame.pack(fill=tk.X, pady=(10, 5))

        self.scale_label = ttk_boot.Label(
            self.scale_frame,
            text="Object Scale:",
            font=INFO_FONT,
            bootstyle=INFO
        )
        self.scale_label.pack(side=tk.LEFT)

        self.scale_var = tk.StringVar(value="1.0")
        self.scale_entry = ttk_boot.Entry(
            self.scale_frame,
            textvariable=self.scale_var,
            width=10,
            bootstyle=PRIMARY
        )
        self.scale_entry.pack(side=tk.LEFT, padx=(5, 5))

        self.scale_apply_btn = ttk_boot.Button(
            self.scale_frame,
            text="Apply",
            command=self.apply_object_scale,
            bootstyle="outline-primary",
            width=8
        )
        self.scale_apply_btn.pack(side=tk.LEFT)

        self.point_label = ttk_boot.Label(
            self.plot_frame,
            text="No point selected",
            font=INFO_FONT,
            bootstyle=INFO
        )
        self.point_label.pack(pady=(10, 5))
        self.info_frame = ttk_boot.Frame(self.plot_frame)
        self.info_frame.pack(fill=tk.BOTH, expand=True)
        self.text_frame = ttk_boot.Frame(self.info_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True)

        self.point_info = tk.Text(
            self.text_frame,
            height=12,
            width=35,
            font=(EMOJI_FONT, 9),
            bg="#2b3e50",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            selectbackground="#3498db",
            selectforeground="#ffffff",
            wrap=tk.WORD
        )

        self.scrollbar = ttk_boot.Scrollbar(
            self.text_frame,
            orient=tk.VERTICAL,
            command=self.point_info.yview,
            bootstyle=PRIMARY
        )
        self.point_info.config(yscrollcommand=self.scrollbar.set)

        self.point_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_human_panel(self):
        self.human_display_frame = ttk_boot.Frame(self.human_image_frame)
        self.human_display_frame.pack(fill=tk.BOTH, expand=True)

        self.human_status_label = ttk_boot.Label(
            self.human_image_frame,
            text="Click human keypoint to annotate",
            font=(EMOJI_FONT, 10, "italic"),
            bootstyle=INFO
        )
        self.human_status_label.pack(pady=(10, 0))

    def setup_bottom_images(self):
        sides_pic = list(os.walk(resource_path("display")))[0][2]
        self.images_container = ttk_boot.Frame(self.bottom_frame)
        self.images_container.pack(fill=tk.X, pady=5)
        for i, pic in enumerate(sides_pic):
            image_frame = ttk_boot.Frame(self.images_container, padding=5)
            image_frame.pack(side=tk.LEFT, padx=5)

            title_label = ttk_boot.Label(
                image_frame,
                text=pic.split(".")[0],
                font=INFO_FONT,
                bootstyle=SECONDARY
            )
            title_label.pack(pady=(0, 5))
            img = Image.open(os.path.join(resource_path("display"), pic))
            img = img.resize((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            setattr(self, pic.split(".")[0], photo)

            img_label = ttk_boot.Label(image_frame, image=photo, relief="solid", borderwidth=2)
            img_label.pack()

    def load_data(self, args):
        self.video_dir = args.video_dir
        # self.joint_to_optimize = args.joint_to_optimize
        self.cap = cv2.VideoCapture(f"{self.video_dir}/video.mp4")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.total_frames)
        self.frame_change(0)
        if os.path.exists(f"{self.video_dir}/kp_record"):
            shutil.rmtree(f"{self.video_dir}/kp_record")
        os.makedirs(f"{self.video_dir}/kp_record")

        for frame in range(self.current_frame, self.total_frames):
            save_name = f"{frame}".zfill(5)
            with open(f"{args.video_dir}/kp_record/{save_name}.json", "w") as file:
                json.dump(self.kp_pair, file, indent=4)

        self.progress_bar.config(to=self.total_frames - 1)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            self.original_width, self.original_height = frame.size
            self.original_max_dim = max(self.original_height, self.original_width)
            self.keypoint_window_size = min(self.original_max_dim, 800)
            standard_size = (480, 480)
            frame = frame.resize(standard_size, Image.Resampling.LANCZOS)

            frame = self.draw_tracking_points_on_frame(frame)

            self.img_width, self.img_height = frame.size
            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label = ttk_boot.Label(self.video_display_frame, image=self.obj_img)
            self.video_label.pack(pady=10)
            self.root.geometry(f"{2 * self.img_width + 360 + 560}x{self.img_height + 500}")
            self.is_playing = False
            if not self.is_app_closing:
                self.update_video_frame()

        if COTRACKER_AVAILABLE:
            try:
                self.cotracker_model = CoTrackerOnlinePredictor(
                    checkpoint='./co-tracker/checkpoints/scaled_online.pth').cuda()
                print("CoTracker model loaded successfully")
            except Exception as e:
                print(f"Failed to load CoTracker model: {e}")
                self.cotracker_model = None
        else:
            self.cotracker_model = None

        if self.cotracker_model is not None:
            try:
                video_path = f"{self.video_dir}/video.mp4"
                self.video_frames = []
                for frame in iio.imiter(video_path, plugin="FFMPEG"):
                    self.video_frames.append(frame)
                print(f"Loaded {len(self.video_frames)} frames for CoTracker")
            except Exception as e:
                print(f"Failed to load video frames for CoTracker: {e}")

        output = torch.load(f"{args.video_dir}/motion/result.pt")
        # print(output.keys())
        self.body_params = output["smpl_params_incam"]
        self.global_body_params = output["smpl_params_global"]

        self.hand_poses = json.load(open(os.path.join(args.video_dir, 'motion/hand_pose.json')))
        self.human_part = json.load(open(f"{resource_path('./data/part_kp.json')}"))
        self.K = output['K_fullimg'][0]
        self.output = output
        self.R = torch.eye(3, dtype=torch.float32)
        self.T = torch.zeros(3, dtype=torch.float32)

        for i in range(self.total_frames):
            if str(i) not in self.hand_poses:
                self.hand_poses[str(i)] = {}
            self.body_params["body_pose"][i], self.hand_poses[str(i)]["left_hand"], self.hand_poses[str(i)][
                "right_hand"] \
                = update_hand_pose(self.hand_poses, self.body_params["global_orient"], self.body_params["body_pose"], i)

        with open(f'{args.video_dir}/align/obj_poses.json') as f:
            self.object_poses = json.load(f)

        self.obj_org = mesh = o3d.io.read_triangle_mesh(f"{args.video_dir}/obj_org.obj")
        self.sampled_obj = self.obj_org.simplify_quadric_decimation(target_number_of_triangles=1000)

        self.obj_orgs, self.t_finals, center_objs = preprocess_obj_sample(self.obj_org, self.object_poses,
                                                                          os.path.join(args.video_dir, 'orient/'),
                                                                          self.total_frames)
        self.sampled_orgs, _ = preprocess_obj(self.sampled_obj, self.object_poses,
                                              os.path.join(args.video_dir, 'orient/'), self.total_frames, center_objs)

        self.obj_orgs_base_vertices = [np.asarray(obj.vertices).copy() for obj in self.obj_orgs]
        self.sampled_orgs_base_vertices = [np.asarray(obj.vertices).copy() for obj in
                                           self.sampled_orgs] if self.sampled_orgs is not None else []

        self.R_finals = [np.eye(3)] * self.total_frames

        if "rotation" not in self.object_poses:
            self.object_poses['rotation'] = []
            for frame in range(self.current_frame, self.total_frames):
                self.object_poses['rotation'].append(np.eye(3))


        self.unwrapped_body_params()

        self.setup_human_keypoints()
        self.setup_bottom_images()
        self.update_frame_counter()

    def unwrapped_body_params(self):
        self.body_pose_params = []
        self.shape_params = []
        self.left_hand_params = []
        self.right_hand_params = []
        self.global_orient = []
        self.transl = []

        for i in range(self.total_frames):
            self.body_pose_params.append(self.body_params["body_pose"][i].reshape(1, -1))
            self.shape_params.append(self.body_params['betas'][i].reshape(1, -1))
            handpose = self.hand_poses[str(i)]
            left_hand_pose = torch.from_numpy(np.asarray(handpose['left_hand']).reshape(-1, 3)).float()
            right_hand_pose = torch.from_numpy(np.asarray(handpose['right_hand']).reshape(-1, 3)).float()
            self.left_hand_params.append(left_hand_pose)
            self.right_hand_params.append(right_hand_pose)
            self.global_orient.append(self.body_params['global_orient'][i].reshape(1, 3))
            self.transl.append(self.body_params['transl'][i].reshape(1, -1))

    def setup_human_keypoints(self):
        obj_img_size = 480, 480
        img = Image.open(resource_path("./data/human_kp.png"))
        human_img_width, human_img_height = img.size
        img = img.resize(obj_img_size, Image.Resampling.LANCZOS)
        self.human_img = ImageTk.PhotoImage(img)
        self.human_img_label = ttk_boot.Label(self.human_display_frame, image=self.human_img)
        self.human_img_label.pack(pady=10)

        for main_joint, location in self.main_joint_coord.items():
            real_x, real_y = location
            real_x, real_y = real_x - 40, real_y + 70
            if main_joint == "leftNeck":
                real_x, real_y = real_x + 25, real_y
            if main_joint == "rightNeck":
                real_x, real_y = real_x - 25, real_y

            scale_x = real_x / human_img_width * obj_img_size[0]
            scale_y = real_y / human_img_height * obj_img_size[1] - 10

            button = ttk_boot.Button(
                self.human_display_frame,
                text=self.button_name[main_joint],
                command=lambda x=main_joint, y_coord=scale_y, x_coord=scale_x: self.show_menu(x, x_coord, y_coord),
                bootstyle="outline-info",
                width=6
            )
            button.place(x=scale_x, y=scale_y)

    def toggle_video(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause", bootstyle="outline-warning")
        else:
            self.play_button.config(text="Play", bootstyle="success")

    def seek_video(self, value):
        was_playing = self.is_playing
        self.is_playing = False
        frame_no = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        self.frame_change(frame_no)
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((480, 480), Image.Resampling.LANCZOS)

            frame = self.draw_tracking_points_on_frame(frame)

            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label.configure(image=self.obj_img)
            self.update_frame_counter()

        self.is_playing = was_playing

    def update_frame_counter(self):
        self.frame_label.config(text=f"{self.current_frame}/{self.total_frames - 1}")

    def draw_tracking_points_on_frame(self, frame):
        if not self.tracking_active or not self.tracked_points:
            return frame

        current_kp = self.frame_keypoints_cache.get(self.current_frame, {})

        if "2D_keypoint" in current_kp and current_kp["2D_keypoint"]:
            try:
                frame_np = np.array(frame)

                height, width = frame_np.shape[:2]
                # print(self.original_width, self.original_height)
                scale_x = width / self.original_width
                scale_y = height / self.original_height

                for obj_idx, img_point in current_kp["2D_keypoint"]:

                    x = int(img_point[0] * scale_x)
                    y = int(img_point[1] * scale_y)

                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame_np, (x, y), 6, (0, 255, 0), -1)
                        cv2.circle(frame_np, (x, y), 8, (255, 255, 255), 2)

                        cv2.putText(frame_np, f"Obj{obj_idx}", (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                frame = Image.fromarray(frame_np)
            except Exception as e:
                print(f"Error drawing tracking points: {e}")

        return frame

    def update_video_frame(self):
        if self.is_app_closing:
            return

        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.frame_change(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize((480, 480), Image.Resampling.LANCZOS)

                frame = self.draw_tracking_points_on_frame(frame)

                self.obj_img = ImageTk.PhotoImage(image=frame)
                self.video_label.configure(image=self.obj_img)
                self.progress_var.set(self.current_frame)
                self.update_frame_counter()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_change(0)
                self.progress_var.set(0)
                self.update_frame_counter()
                self.is_playing = False
                self.play_button.config(text="Play", bootstyle="success")

        if not self.is_app_closing:
            try:
                self.update_timer_id = self.root.after(30, self.update_video_frame)
            except tk.TclError:
                pass

    def show_menu(self, main_joint, x, y):
        menu = tk.Menu(self.human_image_frame, tearoff=0, bg="#2b3e50", fg="#ecf0f1", activebackground="#3498db")
        for sub_joint in self.joint_tree[main_joint]:
            menu.add_command(label=sub_joint, command=lambda sj=sub_joint: self.option_selected(sj))
        menu.add_separator()
        menu.add_command(label="Cancel", command=lambda: self.option_selected(None))
        menu.post(self.human_image_frame.winfo_rootx() + int(x), self.human_image_frame.winfo_rooty() + int(y))

    def option_selected(self, option):
        if option == "exit" or option is None:
            return
        self.selected_2d_point = None
        self.selected_human_kp = option
        self.human_status_label.config(text=f"Selected: {option}")
        print(f"Selected: {option}")
        self.update_plot()

    def _select_2d_point_on_frame(self, window_title="2D keypoint selection", image=None):
        if image is None:
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            if not ret:
                messagebox.showerror("Error", "Could not read video frame.")
                return None
        else:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(frame.shape)
        display_img = frame.copy()
        height, width = display_img.shape[:2]
        max_size = 800
        max_dim = max(height, width)

        scale = 1.0
        if max_dim > max_size:
            scale = max_size / max_dim
            new_h, new_w = int(height * scale), int(width * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))

        cv2.namedWindow(window_title)

        clicked_point = [None]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point[0] = (x, y)
                img_copy = display_img.copy()
                cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(img_copy, f"({x}, {y})", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window_title, img_copy)

        cv2.setMouseCallback(window_title, mouse_callback)
        cv2.imshow(window_title, display_img)

        while clicked_point[0] is None:
            if cv2.waitKey(30) & 0xFF == 27:  # Allow exit with ESC
                break
            try:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        cv2.destroyWindow(window_title)
        if clicked_point[0] is not None:
            x, y = clicked_point[0]
            x /= scale
            y /= scale
            return (x, y)
        return None

    def update_plot(self):
        # update kp_pair
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if os.path.exists(current_file):
            with open(current_file, "r") as file:
                current_kp = json.load(file)
            self.kp_pair = current_kp

        if self.selected_2d_point is not None and self.obj_point is not None:
            self.kp_pair["2D_keypoint"].append((int(self.obj_point), self.selected_2d_point))
            self.point_info.insert(tk.END, f"2D keypoint: {self.selected_2d_point}\n")
            self.point_info.insert(tk.END, f"Object point index: {self.obj_point}\n")
            self.point_info.insert(tk.END, f"Current frame: {self.current_frame}\n")
            self.point_info.insert(tk.END, "─" * 30 + "\n")
            self.point_info.see(tk.END)

            self.no_annot = False
            self.annotated_frames.add(self.current_frame)
            self.annotated_frames_2D.add(self.current_frame)
            if len(self.annotated_frames) > 1:
                self.last_frame = sorted(list(self.annotated_frames))[-2]
            else:
                self.last_frame = None

            if len(self.annotated_frames_2D) > 1:
                self.last_frame_2D = sorted(list(self.annotated_frames_2D))[-2]
            else:
                self.last_frame_2D = None

            if not self.tracking_active and len(self.kp_pair["2D_keypoint"]) > 0:
                # self.render_status_label.config(text="Click 'Start tracking' button to track points")
                save_name = f"{self.current_frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
                self.frame_keypoints_cache[self.current_frame] = self.kp_pair.copy()
            else:

                print('r', self.kp_pair)

                save_name = f"{self.current_frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
                self.frame_keypoints_cache[self.current_frame] = self.kp_pair.copy()

                self.refresh_current_frame()

            return

        if self.selected_human_kp is None or self.obj_point is None:
            return

        self.no_annot = False
        self.no_annote_2D = True
        self.annotated_frames.add(self.current_frame)

        if len(self.annotated_frames) > 1:
            self.last_frame = sorted(list(self.annotated_frames))[-2]
        else:
            self.last_frame = None

        self.kp_pair[self.selected_human_kp] = int(self.obj_point)

        self.point_info.insert(tk.END, f"Current frame: {self.current_frame}\n")
        self.point_info.insert(tk.END, f"Object point index: {self.obj_point}\n")
        self.point_info.insert(tk.END, f"Human keypoint: {self.selected_human_kp}\n")
        self.point_info.insert(tk.END, "─" * 30 + "\n")
        self.point_info.see(tk.END)

        for frame in range(self.current_frame, self.total_frames):
            save_name = f"{frame}".zfill(5)
            frame_file = f"{self.video_dir}/kp_record/{save_name}.json"

            if os.path.exists(frame_file):
                with open(frame_file, "r") as file:
                    existing_data = json.load(file)
                if "2D_keypoint" in existing_data:
                    self.kp_pair["2D_keypoint"] = existing_data["2D_keypoint"]

            with open(frame_file, "w") as file:
                json.dump(self.kp_pair, file, indent=4)
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if os.path.exists(current_file):
            with open(current_file, "r") as file:
                current_kp = json.load(file)
            self.kp_pair = current_kp

        self.point_label.config(text="No point selected")
        self.human_status_label.config(text="Click human keypoint to annotate")

    def keypoint_2D(self):
        point = self._select_2d_point_on_frame()
        if point:
            self.selected_2d_point = point
            self.update_plot()

    def get_body_points(self):
        body_pose = self.body_pose_params[self.current_frame]
        shape = self.shape_params[self.current_frame]
        global_orient = self.global_orient[self.current_frame]
        left_hand_pose = self.left_hand_params[self.current_frame]
        right_hand_pose = self.right_hand_params[self.current_frame]
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1)
        transl = self.transl[self.current_frame]

        output = model(betas=shape,
                       body_pose=body_pose,
                       left_hand_pose=left_hand_pose,
                       right_hand_pose=right_hand_pose,
                       jaw_pose=zero_pose,
                       leye_pose=zero_pose,
                       reye_pose=zero_pose,
                       global_orient=global_orient,
                       expression=torch.zeros((1, 10)).float(),
                       transl=transl)
        print('transl', transl, 'global_orient', global_orient)
        return output.vertices[0]

    def get_object_points(self):
        # print('current frame', self.current_frame)
        verts = np.asarray(self.obj_orgs[self.current_frame].vertices, dtype=np.float32)
        R = self.R_finals[self.current_frame]
        t = self.t_finals[self.current_frame]
        return np.matmul(verts, R.T) + t

    def reset_keypoints(self):
        self.kp_pair = {"2D_keypoint": []}
        self.tracked_points = {}
        self.tracking_active = False
        self.point_info.delete('1.0', tk.END)
        self.point_info.insert(tk.END, "Keypoints reset\n")
        self.point_label.config(text="No point selected")
        self.human_status_label.config(text="Click human keypoint to annotate")
        self.refresh_current_frame()
        with open(f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json", "w") as file:
            json.dump(self.kp_pair, file, indent=4)

    def apply_object_scale(self):
        try:
            scale_factor = float(self.scale_var.get())
            if scale_factor <= 0:
                messagebox.showerror("Error", "Scale factor must be greater than 0")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            return
        if self.obj_orgs_base_vertices is None or len(self.obj_orgs_base_vertices) != len(self.obj_orgs):
            self.obj_orgs_base_vertices = [np.asarray(obj.vertices).copy() for obj in self.obj_orgs]
        if self.sampled_orgs is not None:
            if self.sampled_orgs_base_vertices is None or len(self.sampled_orgs_base_vertices) != len(
                    self.sampled_orgs):
                self.sampled_orgs_base_vertices = [np.asarray(obj.vertices).copy() for obj in self.sampled_orgs]
        else:
            self.sampled_orgs_base_vertices = []

        
        result = messagebox.askyesno("Confirm Scale",
                                     f"Will apply {scale_factor}x scale to objects in all frames.\n"
                                     f"This operation recalculates from the original backup each time and does not stack scaling.\n\nAre you sure you want to continue?")

        if not result:
            return

        n_frames = min(self.total_frames, len(self.obj_orgs_base_vertices), len(self.obj_orgs))
        for frame_idx in range(n_frames):
            base_vertices = self.obj_orgs_base_vertices[frame_idx]
            center = np.mean(base_vertices, axis=0)
            vertices_final = (base_vertices - center) * scale_factor + center
            self.obj_orgs[frame_idx].vertices = o3d.utility.Vector3dVector(vertices_final)

        n_frames_sampled = min(self.total_frames, len(self.sampled_orgs_base_vertices),
                               len(self.sampled_orgs) if self.sampled_orgs is not None else 0)
        for frame_idx in range(n_frames_sampled):
            base_vertices = self.sampled_orgs_base_vertices[frame_idx]
            center = np.mean(base_vertices, axis=0)
            vertices_final = (base_vertices - center) * scale_factor + center
            self.sampled_orgs[frame_idx].vertices = o3d.utility.Vector3dVector(vertices_final)

        self.refresh_current_frame()

        messagebox.showinfo("Completed", f"Object scale {scale_factor}x applied!")
        if hasattr(self, 'point_info'):
            self.point_info.insert(tk.END, f"Applied {scale_factor}x scale to all frames (from backup)\n")
            self.point_info.see(tk.END)

    def track_2D_points_with_cotracker_online(self, obj_indices, start_points, start_frame=0):
        if not COTRACKER_AVAILABLE or self.cotracker_model is None or self.video_frames is None:
            print("CoTracker not available, falling back to interpolation")
            return False

        try:
            device = next(self.cotracker_model.parameters()).device

            # self.cotracker_model.reset()

            queries = []
            for i, (obj_idx, point) in enumerate(zip(obj_indices, start_points)):
                queries.append([0.0, point[0], point[1]])
            print(f'queries: {queries}')

            queries_tensor = torch.tensor(queries, dtype=torch.float32).to(device)
            print(f"Tracking {len(queries)} points from frame {start_frame}")

            video_sequence = self.video_frames[start_frame:]
            window_frames = [video_sequence[0]]

            def _process_step(window_frames, is_first_step, queries=None):
                frames_to_use = window_frames[-self.cotracker_model.step * 2:] if len(
                    window_frames) >= self.cotracker_model.step * 2 else window_frames
                if len(frames_to_use) == 0:
                    return None, None

                video_chunk = (
                    torch.tensor(
                        np.stack(frames_to_use), device=device
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )  # (1, T, 3, H, W)
                return self.cotracker_model(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=queries,
                    grid_size=0,
                    grid_query_frame=0,
                )

            is_first_step = True
            pred_tracks_list = []
            pred_visibility_list = []

            for i in range(1, len(video_sequence)):
                window_frames.append(video_sequence[i])

                if (i % self.cotracker_model.step == 0) or (i == len(video_sequence) - 1):
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        queries=queries_tensor[None] if is_first_step else None,
                    )

                    if pred_tracks is not None:
                        pred_tracks_list.append(pred_tracks)
                        pred_visibility_list.append(pred_visibility)

                    is_first_step = False

            if not pred_tracks_list:
                print("No valid tracking results")
                return False

            final_tracks = pred_tracks_list[-1][0].permute(1, 0, 2).cpu().numpy()  # [num_points, num_frames, 2]
            final_visibility = pred_visibility_list[-1][0].permute(1, 0).cpu().numpy()

            print(f"Final tracks shape: {final_tracks.shape}")

            for i, obj_idx in enumerate(obj_indices):
                tracks = final_tracks[i]  # [num_frames, 2]
                visibility = final_visibility[i] if i < len(final_visibility) else None

                self.tracked_points[obj_idx] = []
                for frame_idx in range(len(tracks)):
                    self.tracked_points[obj_idx].append(tracks[frame_idx].tolist())

            print(f"CoTracker online tracking completed for {len(obj_indices)} points")
            return True


        except Exception as e:
            print(f"CoTracker online tracking failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def apply_tracking_results_to_all_frames(self):
        if not self.tracked_points:
            return

        print("Applying tracking results to all frames...")

        for frame_idx in range(self.total_frames):
            frame_file = f"{self.video_dir}/kp_record/{str(frame_idx).zfill(5)}.json"

            if os.path.exists(frame_file):
                with open(frame_file, "r") as file:
                    current_kp = json.load(file)
            else:
                current_kp = {"2D_keypoint": []}

            current_kp["2D_keypoint"] = []
            for obj_idx, tracks in self.tracked_points.items():
                if frame_idx < len(tracks) and tracks[frame_idx] is not None:
                    point = tracks[frame_idx]
                    x, y = point[0], point[1]

                    if 0 <= x <= self.original_width and 0 <= y <= self.original_height:
                        current_kp["2D_keypoint"].append([obj_idx, point])
                    else:
                        print(
                            f"Frame {frame_idx}: Point ({x:.1f}, {y:.1f}) for obj {obj_idx} is out of bounds, skipping")

            with open(frame_file, "w") as file:
                json.dump(current_kp, file, indent=4)

            self.frame_keypoints_cache[frame_idx] = current_kp.copy()

        print("Tracking results applied to all frames")

        self.refresh_current_frame()

    def apply_tracking_results_from_current_frame(self):
        if not self.tracked_points:
            return

        print(f"Applying tracking results from frame {self.current_frame} to end...")

        for frame_idx in range(self.current_frame, self.total_frames):
            frame_file = f"{self.video_dir}/kp_record/{str(frame_idx).zfill(5)}.json"

            if os.path.exists(frame_file):
                with open(frame_file, "r") as file:
                    current_kp = json.load(file)
            else:
                current_kp = {"2D_keypoint": []}

            current_kp["2D_keypoint"] = []
            for obj_idx, tracks in self.tracked_points.items():
                track_idx = frame_idx - self.current_frame
                if track_idx < len(tracks) and tracks[track_idx] is not None:
                    point = tracks[track_idx]
                    x, y = point[0], point[1]

                    if 0 <= x <= self.original_width and 0 <= y <= self.original_height:
                        current_kp["2D_keypoint"].append([obj_idx, point])
                    else:
                        print(
                            f"Frame {frame_idx}: Point ({x:.1f}, {y:.1f}) for obj {obj_idx} is out of bounds, skipping")

            with open(frame_file, "w") as file:
                json.dump(current_kp, file, indent=4)

            self.frame_keypoints_cache[frame_idx] = current_kp.copy()

        print(f"Tracking results applied from frame {self.current_frame} to end")

        self.refresh_current_frame()

    def refresh_current_frame(self):
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((480, 480), Image.Resampling.LANCZOS)

            frame = self.draw_tracking_points_on_frame(frame)

            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label.configure(image=self.obj_img)

    def start_2d_tracking(self):
        if not self.kp_pair["2D_keypoint"]:
            messagebox.showwarning("Warning", "Please annotate 2D keypoints first before tracking")
            return
        if self.tracking_active:
            messagebox.showinfo("Info", "Tracking is already in progress")
            return
        result = messagebox.askyesno(
            "Confirm Tracking",
            f"Will start tracking {len(self.kp_pair['2D_keypoint'])} 2D points from frame {self.current_frame}.\n"
            f"This will overwrite 2D keypoint data for all frames.\n\nAre you sure you want to start tracking?"
        )
        if not result:
            return

        self.tracked_points = {}
        self.tracking_active = False

        obj_indices = [pair[0] for pair in self.kp_pair["2D_keypoint"]]
        start_points = [pair[1] for pair in self.kp_pair["2D_keypoint"]]

        def run_tracking():
            success = self.track_2D_points_with_cotracker_online(obj_indices, start_points, self.current_frame)
            if success:
                self.tracking_active = True
                self.apply_tracking_results_to_all_frames()
                if not self.is_app_closing:
                    try:
                        self.root.after(0, lambda: messagebox.showinfo("Success",
                                                                       "2D point tracking completed!") if not self.is_app_closing else None)
                    except tk.TclError:
                        pass
            else:
                if not self.is_app_closing:
                    try:
                        self.root.after(0, lambda: messagebox.showerror("Error",
                                                                        "Tracking failed, please check if CoTracker is installed correctly") if not self.is_app_closing else None)
                    except tk.TclError:
                        pass

        threading.Thread(target=run_tracking, daemon=True).start()

    def restart_tracking_from_current(self):
        self.tracked_points = {}
        self.tracking_active = False

        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if not os.path.exists(current_file):
            messagebox.showwarning("Warning", "No keypoint data saved for current frame")
            return
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        if "2D_keypoint" not in current_kp or not current_kp["2D_keypoint"]:
            messagebox.showwarning("Warning", "No 2D keypoint annotation for current frame")
            return
        result = messagebox.askyesno(
            "Confirm Retracking",
            f"Will restart tracking {len(current_kp['2D_keypoint'])} 2D points from frame {self.current_frame}.\n"
            f"This will overwrite all 2D keypoint data from current frame to the last frame.\n\nAre you sure you want to retrack?"
        )
        if not result:
            return

        obj_indices = [pair[0] for pair in current_kp["2D_keypoint"]]
        start_points = [pair[1] for pair in current_kp["2D_keypoint"]]

        print(f"Restarting tracking from frame {self.current_frame} with points: {start_points}")

        def run_retracking():
            success = self.track_2D_points_with_cotracker_online(obj_indices, start_points, self.current_frame)
            if success:
                self.tracking_active = True
                self.apply_tracking_results_from_current_frame()
                if not self.is_app_closing:
                    try:
                        self.root.after(0, lambda: messagebox.showinfo("Success",
                                                                       f"Retracking from frame {self.current_frame} completed!") if not self.is_app_closing else None)
                    except tk.TclError:
                        pass
            else:
                if not self.is_app_closing:
                    try:
                        self.root.after(0, lambda: messagebox.showerror("Error",
                                                                        "Retracking failed, please check if CoTracker is installed correctly") if not self.is_app_closing else None)
                    except tk.TclError:
                        pass

        threading.Thread(target=run_retracking, daemon=True).start()

    def delete_2d_keypoint(self, obj_idx, parent_window):
        def _to_int_safe(x):
            try:
                return int(x)
            except Exception:
                return x

        target = _to_int_safe(obj_idx)
        deleted_any = False

        for frame in range(self.current_frame, self.total_frames):
            frame_path = f"{self.video_dir}/kp_record/{str(frame).zfill(5)}.json"
            if not os.path.exists(frame_path):
                continue
            try:
                with open(frame_path, "r", encoding="utf-8") as f:
                    current_kp = json.load(f)
            except Exception:
                continue

            kp_list = current_kp.get("2D_keypoint", [])
            if not kp_list:
                continue

            
            new_list = []
            removed_here = False
            for pair in kp_list:
                try:
                    k_obj = _to_int_safe(pair[0])
                except Exception:
                    k_obj = pair[0]
                if k_obj == target:
                    removed_here = True
                    continue
                new_list.append(pair)

            if removed_here:
                current_kp["2D_keypoint"] = new_list
                with open(frame_path, "w", encoding="utf-8") as f:
                    json.dump(current_kp, f, indent=4, ensure_ascii=False)
                self.frame_keypoints_cache[frame] = current_kp.copy()
                deleted_any = True

        self.refresh_current_frame()
        parent_window.destroy()
        if deleted_any:
            messagebox.showinfo("Success", "2D keypoint deleted")
        else:
            messagebox.showinfo("Info", "No matching 2D keypoint found to delete")
        self.manage_existing_keypoints()

    def delete_3d_keypoint(self, joint_name, parent_window):
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        with open(current_file, "r") as file:
            current_kp = json.load(file)

        if joint_name in current_kp:
            del current_kp[joint_name]
            self.kp_pair = current_kp
            self.no_annot = False

            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                frame_file = f"{self.video_dir}/kp_record/{save_name}.json"

                if os.path.exists(frame_file):
                    with open(frame_file, "r") as file:
                        existing_data = json.load(file)
                    if "2D_keypoint" in existing_data:
                        self.kp_pair["2D_keypoint"] = existing_data["2D_keypoint"]

                with open(frame_file, "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            if os.path.exists(current_file):
                with open(current_file, "r") as file:
                    current_kp = json.load(file)
                self.kp_pair = current_kp

            parent_window.destroy()
            messagebox.showinfo("Success", "3D keypoint deleted")
            self.manage_existing_keypoints()

    def modify_2d_keypoint(self, index, parent_window):
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        with open(current_file, "r") as file:
            current_kp = json.load(file)

        if "2D_keypoint" not in current_kp or index >= len(current_kp["2D_keypoint"]):
            return

        obj_idx, old_img_point = current_kp["2D_keypoint"][index]
        parent_window.destroy()
        self._select_new_2d(obj_idx, old_img_point, index)
        self.manage_existing_keypoints()

    def modify_3d_keypoint(self, joint_name, parent_window):
        parent_window.destroy()
        joint_display_name = self.button_name.get(joint_name, joint_name)
        self.open_o3d_viewer()

        if self.obj_point is not None:
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            with open(current_file, "r") as file:
                current_kp = json.load(file)

            current_kp[joint_name] = int(self.obj_point)
            self.kp_pair = current_kp
            self.no_annot = False

            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                frame_file = f"{self.video_dir}/kp_record/{save_name}.json"

                if os.path.exists(frame_file):
                    with open(frame_file, "r") as file:
                        existing_data = json.load(file)
                    if "2D_keypoint" in existing_data:
                        self.kp_pair["2D_keypoint"] = existing_data["2D_keypoint"]

                with open(frame_file, "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            if os.path.exists(current_file):
                with open(current_file, "r") as file:
                    current_kp = json.load(file)
                self.kp_pair = current_kp

            messagebox.showinfo("Success", "3D keypoint modified")
        self.manage_existing_keypoints()

    def _select_new_2d(self, obj_idx, old_img_point, kp_index):
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        display_img = frame.copy()
        height, width = display_img.shape[:2]
        max_size = 800
        max_dim = max(height, width)

        if max_dim > max_size:
            scale = max_size / max_dim
            new_h, new_w = int(height * scale), int(width * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        else:
            scale = 1

        cv2.namedWindow("2D keypoint modify")

        clicked_point = [None]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point[0] = (x, y)
                cv2.circle(display_img, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(display_img, f"({x}, {y})", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("2D keypoint modify", display_img)

        cv2.setMouseCallback("2D keypoint modify", mouse_callback)
        cv2.imshow("2D keypoint modify", display_img)

        while clicked_point[0] is None:
            cv2.waitKey(30)
            try:
                if cv2.getWindowProperty("2D keypoint modify", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        try:
            cv2.destroyWindow("2D keypoint modify")
        except cv2.error:
            pass
        if clicked_point[0] is not None:
            x, y = clicked_point[0]
            x /= scale
            y /= scale
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            with open(current_file, "r") as file:
                current_kp = json.load(file)

            current_kp["2D_keypoint"][kp_index] = [obj_idx, [x, y]]
            self.kp_pair = current_kp
            self.no_annot = False
            self.no_annote_2D = False
            self.annotated_frames.add(self.current_frame)
            self.annotated_frames_2D.add(self.current_frame)
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            if os.path.exists(current_file):
                with open(current_file, "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
                # self.kp_pair = current_kp

            
            self.refresh_current_frame()

            messagebox.showinfo("Success", "2D keypoint modified")

    def open_scale_check_viewer(self):
        try:
            human_points = self.get_body_points()
            human_pcd = o3d.geometry.PointCloud()
            human_pcd.points = o3d.utility.Vector3dVector(human_points)
            human_pcd.paint_uniform_color([1.0, 0.0, 0.0])

            obj_vertices = self.get_object_points()
            obj_mesh = deepcopy(self.obj_orgs[self.current_frame])
            obj_mesh.vertices = o3d.utility.Vector3dVector(obj_vertices)
            obj_mesh.compute_vertex_normals()
            # obj_mesh.paint_uniform_color([0.0, 1.0, 0.0])
            obj_mesh.simplify_quadric_decimation(target_number_of_triangles=3000)

            print("Object mesh vertices shape:", np.asarray(obj_mesh.vertices).shape)
            print("Human point cloud vertices shape:", np.asarray(human_pcd.points).shape)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Scale Check - Green: Object, Red: Human")
            vis.add_geometry(obj_mesh)
            vis.add_geometry(human_pcd)
            opt = vis.get_render_option()
            opt.point_size = 8.0
            opt.background_color = np.asarray([1, 1, 1])

            print("Scale Check Window:")
            print(f"  Green mesh: Object ({len(self.obj_orgs[self.current_frame].vertices)} points)")
            print(f"  Red points: Human ({len(human_pcd.points)} points)")
            print(f"  Current object scale: {self.scale_var.get()}")
            print("   Close the window when done checking the scale.")

            vis.run()
            vis.destroy_window()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open scale check viewer: {e}")

    def open_o3d_viewer(self):

        vertices = self.get_object_points()

        mesh = deepcopy(self.obj_orgs[self.current_frame])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = mesh.vertex_colors

        # Only show object points for selection
        # combined_pcd = pcd

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="3D Point Selection (Object Only)")
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 15.0
        vis.run()

        picked_points = vis.get_picked_points()
        if picked_points:
            picked_idx = picked_points[-1]
            # Always set object point (no multiview modify mode)
            self.obj_point = str(min(picked_idx, len(pcd.points) - 1))
            self.point_label.config(text=f"Selected point: {self.obj_point}")
        else:
            self.obj_point = None
            self.point_label.config(text="No point selected")
        vis.destroy_window()

    def manage_existing_keypoints(self):
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if not os.path.exists(current_file):
            messagebox.showwarning("Warning", "No keypoint data saved for current frame")
            return

        with open(current_file, "r") as file:
            current_kp = json.load(file)

        manage_window = tk.Toplevel(self.root)
        manage_window.title(f"Manage Keypoints - Frame {self.current_frame}")
        manage_window.geometry("900x600")

        style = ttk_boot.Style(theme="superhero")
        main_frame = ttk_boot.Frame(manage_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        title_label = ttk_boot.Label(main_frame, text=f"Frame {self.current_frame} keypoints", font=TITLE_FONT,
                                     bootstyle=PRIMARY)
        title_label.pack(pady=(0, 20))
        canvas = tk.Canvas(main_frame, bg="#2b3e50", highlightthickness=0)
        scrollbar = ttk_boot.Scrollbar(main_frame, orient="vertical", command=canvas.yview, bootstyle=PRIMARY)
        scrollable_frame = ttk_boot.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        has_any_keypoint = False

        # 2D keypoints section
        if "2D_keypoint" in current_kp and current_kp["2D_keypoint"]:
            has_any_keypoint = True
            section_frame = ttk_boot.LabelFrame(scrollable_frame, text="2D keypoint", padding=15, bootstyle=INFO)
            section_frame.pack(fill=tk.X, pady=(0, 10))
            for i, (obj_idx, img_point) in enumerate(current_kp["2D_keypoint"]):
                item_frame = ttk_boot.Frame(section_frame)
                item_frame.pack(fill=tk.X, pady=5)
                info_label = ttk_boot.Label(item_frame,
                                            text=f"Object point {obj_idx} -> Image point ({img_point[0]:.1f}, {img_point[1]:.1f})",
                                            font=DEFAULT_FONT)
                info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                btn_frame = ttk_boot.Frame(item_frame)
                btn_frame.pack(side=tk.RIGHT)
                modify_btn = ttk_boot.Button(btn_frame, text="Modify",
                                             command=lambda idx=i: self.modify_2d_keypoint(idx, manage_window),
                                             bootstyle=WARNING, width=10)
                modify_btn.pack(side=tk.LEFT, padx=2)
                # pass object index instead of positional index
                delete_btn = ttk_boot.Button(btn_frame, text="Delete",
                                             command=lambda obj=obj_idx: self.delete_2d_keypoint(obj, manage_window),
                                             bootstyle=DANGER, width=10)
                delete_btn.pack(side=tk.LEFT, padx=2)

        # 3D keypoints section (exclude only '2D_keypoint')
        has_3d_keypoints = False
        for key, value in current_kp.items():
            if key == "2D_keypoint":
                continue
            if not has_3d_keypoints:
                has_any_keypoint = True
                section_frame = ttk_boot.LabelFrame(scrollable_frame, text="3D keypoint", padding=15, bootstyle=SUCCESS)
                section_frame.pack(fill=tk.X, pady=(0, 10))
                has_3d_keypoints = True

            item_frame = ttk_boot.Frame(section_frame)
            item_frame.pack(fill=tk.X, pady=5)
            joint_name = self.button_name.get(key, key)
            info_label = ttk_boot.Label(item_frame, text=f"{joint_name} -> Object point {value}", font=DEFAULT_FONT)
            info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            btn_frame = ttk_boot.Frame(item_frame)
            btn_frame.pack(side=tk.RIGHT)
            modify_btn = ttk_boot.Button(btn_frame, text="Modify",
                                         command=lambda joint=key: self.modify_3d_keypoint(joint, manage_window),
                                         bootstyle=WARNING, width=10)
            modify_btn.pack(side=tk.LEFT, padx=2)
            delete_btn = ttk_boot.Button(btn_frame, text="Delete",
                                         command=lambda joint=key: self.delete_3d_keypoint(joint, manage_window),
                                         bootstyle=DANGER, width=10)
            delete_btn.pack(side=tk.LEFT, padx=2)

        if not has_any_keypoint:
            empty_label = ttk_boot.Label(scrollable_frame, text="No keypoint annotation", font=SUBTITLE_FONT,
                                         bootstyle=WARNING)
            empty_label.pack(pady=50)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        close_btn = ttk_boot.Button(main_frame, text="Close", command=manage_window.destroy, bootstyle=SUCCESS,
                                    width=15)
        close_btn.pack(pady=(20, 0))

    def frame_change(self, new_frame):
        self.current_frame = new_frame

    def save_kp_record_merged(self):
        kp_dir = os.path.join(self.video_dir, "kp_record")
        if not os.path.isdir(kp_dir):
            messagebox.showwarning("Warning", "kp_record folder does not exist, cannot save")
            return False

        merged = {}
        invalid_frames = []
        first_annotated_frame = None
        try:
            for frame_idx in range(0, self.current_frame + 1):
                fname = f"{frame_idx:05d}.json"
                fpath = os.path.join(kp_dir, fname)
                if not os.path.exists(fpath):
                    if first_annotated_frame is not None:
                        invalid_frames.append((frame_idx, 0, 0, 0))
                    continue
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                num_2d = len(data.get("2D_keypoint", []) or [])
                num_3d = len([k for k in data.keys() if k != "2D_keypoint"])
                has_annotation = (num_2d > 0) or (num_3d > 0)
                if has_annotation and first_annotated_frame is None:
                    first_annotated_frame = frame_idx
                dof = 3 * num_3d + 2 * num_2d
                if first_annotated_frame is not None and frame_idx >= first_annotated_frame and dof < 6:
                    invalid_frames.append((frame_idx, dof, num_3d, num_2d))
                merged[f"{frame_idx:05d}"] = data
            if first_annotated_frame is None:
                messagebox.showwarning("Warning", "No 2D or 3D annotations found, cannot save")
                return False
            if invalid_frames:
                msg_lines = ["Insufficient degrees of freedom for the following frames (must be >= 6):"]
                for frame_idx, dof, n3, n2 in invalid_frames[:10]:
                    msg_lines.append(f"Frame {frame_idx}: DoF={dof} (3D={n3}x3, 2D={n2}x2)")
                if len(invalid_frames) > 10:
                    msg_lines.append(f"... and {len(invalid_frames) - 10} other frames")
                messagebox.showwarning("Annotation Invalid", "\n".join(msg_lines))
                return False
            try:
                object_scale = float(self.scale_var.get())
            except Exception:
                object_scale = 1.0
            merged["object_scale"] = object_scale
            merged["is_static_object"] = self.is_static_object.get()
            merged["start_frame_index"] = first_annotated_frame
            out_path = os.path.join(self.video_dir, "kp_record_merged.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Saved frames 0..{self.current_frame} to {out_path}")
            if hasattr(self, "point_info"):
                self.point_info.insert(tk.END, f"Saved merged kp_record (0..{self.current_frame}) to: {out_path}\n")
                self.point_info.insert(tk.END, f"Object scale saved: {object_scale}\n")
                self.point_info.see(tk.END)
            
            self.on_close()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")
            return False

    def on_close(self):
        self.is_app_closing = True

        try:
            self.root.report_callback_exception = lambda *args: None
        except Exception:
            pass

        try:
            if hasattr(self, 'update_timer_id') and self.update_timer_id is not None:
                self.root.after_cancel(self.update_timer_id)
                self.update_timer_id = None

            try:
                after_ids = self.root.tk.call('after', 'info')
                for after_id in after_ids:
                    try:
                        self.root.after_cancel(after_id)
                    except (tk.TclError, Exception):
                        pass
            except (tk.TclError, Exception):
                pass
        except Exception:
            pass

        try:
            self.is_playing = False
        except Exception:
            pass

        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if hasattr(self, 'cotracker_model') and self.cotracker_model is not None:
                del self.cotracker_model
                self.cotracker_model = None
        except Exception:
            pass

        try:
            self.video_frames = None
            self.tracked_points = {}
            self.frame_keypoints_cache = {}
            self.obj_orgs = None
            self.sampled_orgs = None
            self.obj_org = None
            self.sampled_obj = None
            self.obj_orgs_base_vertices = None
            self.sampled_orgs_base_vertices = None
        except Exception:
            pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            self.root.protocol("WM_DELETE_WINDOW", "")

            for widget in self.root.winfo_children():
                try:
                    widget.unbind_all("<Button>")
                    widget.unbind_all("<Key>")
                    widget.unbind_all("<Motion>")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def mark_delete_and_next(self):
        """Mark current data as deleted and skip to next: only write all_parameters_delete.json = {"status": "delete"}"""
        try:
            confirm = messagebox.askyesno("Confirm Delete", "This video will be marked for deletion (status: delete). Are you sure?")
            if not confirm:
                return
            # final_dir = os.path.join(self.video_dir, "final_optimized_parameters")
            # os.makedirs(final_dir, exist_ok=True)
            out_path = os.path.join(self.video_dir, "kp_record_merged.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"status": "delete"}, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Completed", f"Deletion mark written: {out_path}")
            
            self.on_close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write deletion mark: {e}")


if __name__ == "__main__":
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        if "tkinter" in str(exc_value).lower() or "tcl" in str(exc_value).lower():
            return
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


    sys.excepthook = handle_exception

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the video directory')
    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        print(f"Error: {args.video_dir} does not exist")
        sys.exit(1)

    try:
        root = ttk_boot.Window(themename="superhero")
        root.report_callback_exception = lambda *args: None
        app = KeyPointApp(root, args)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")