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

import torch
import numpy as np
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
def process_frame2square(frame):
    (h, w) = frame.shape[:2]
    square_size = max(w, h)
    square = np.ones((square_size,square_size, 3), dtype=np.uint8) * 255
    start_y = (square_size - h) // 2
    start_x = (square_size - w) // 2
    square[start_y:start_y + h, start_x:start_x + w] = frame

    return square
def process_frame2square_mask(frame):
    (h, w) = frame.shape[:2]
    square_size = max(w, h)
    square = np.zeros((square_size,square_size, 3), dtype=np.uint8) * 255
    start_y = (square_size - h) // 2
    start_x = (square_size - w) // 2
    square[start_y:start_y + h, start_x:start_x + w] = frame

    return square