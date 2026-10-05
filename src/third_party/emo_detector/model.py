import os
import sys
sys.path.append(os.path.dirname(__file__))

from pathlib import Path
import argparse

import torch
from torch import nn
from skimage import io

from emonet.models import EmoNet
from src.third_party.face_evolve.applications.align.detector import detect_faces # type: ignore

import cv2
import numpy as np

torch.backends.cudnn.benchmark = True

def get_scale_center(bb):
    
    center = np.array([bb[2] - (bb[2]-bb[0])/2, bb[3] - (bb[3]-bb[1])/2])
    scale = (bb[2]-bb[0] + bb[3]-bb[1])/220.0

    return scale, center

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 200
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))

    return t

class EmoNetEncoder(torch.nn.Module):
    def __init__(self, device = 'cuda', n_expression=8):
        super().__init__()
        self.device = device
        self.image_size = 256
        emotion_classes = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger", 7:"Contempt"}
        state_dict_path = os.path.join(os.path.dirname(__file__), "../../../checkpoints", f'emonet_{n_expression}.pth')
        # Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

        state_dict = torch.load(str(state_dict_path), map_location='cpu', weights_only=True)
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        net = EmoNet(n_expression=n_expression).to(device)
        net.load_state_dict(state_dict, strict=False)
        net.eval()

        self.net = net
    def preprocess_image(self, img):
        _, landmarks = detect_faces(img)
        facial5points = np.array([[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)])
        bb = [facial5points.min(axis=0)[0], facial5points.min(axis=0)[1], facial5points.max(axis=0)[0], facial5points.max(axis=0)[1]]

        scale, center = get_scale_center(bb)
        mat = get_transform(center, scale, (self.image_size, self.image_size))[:2]
        image = cv2.warpAffine(np.array(img), mat, (self.image_size, self.image_size))
        image_tensor = torch.from_numpy(image).permute(2,0,1).to(self.device).to(torch.float32) / 255.0
        return image_tensor
    def forward(self, img):
        # [0, 1]
        # image_rgb = io.imread(image_path)[:,:,:3]
        # # Resize image to (256,256)
        # image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))

        # # Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
        # image_tensor = torch.Tensor(image_rgb).permute(2,0,1).to(self.device)/255.0

        # image_tensor = torch.nn.functional.interpolate(image_tensor, (emonet.image_size, emonet.image_size), mode='bilinear', align_corners=False, antialias=True)
        with torch.no_grad():
            output = self.net(self.preprocess_image(img)[None])
        return output