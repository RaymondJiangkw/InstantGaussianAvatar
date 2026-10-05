import torch
from torch import nn
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.third_party.smirk.src.smirk_encoder import SmirkEncoder
from src.third_party.smirk.src.FLAME.FLAME import FLAME
from src.third_party.smirk.src.renderer.renderer import Renderer
import argparse
import os
import src.third_party.smirk.src.utils.masking as masking_utils
from src.third_party.smirk.utils.mediapipe_utils import run_mediapipe
from src.third_party.smirk.datasets.base_dataset import create_mask
import torch.nn.functional as F


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

from PIL import Image

class SMIRKEncoder(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.image_size = 224
        self.smirk_encoder = SmirkEncoder().eval().requires_grad_(False).to(device)

        checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '../../../checkpoints/SMIRK_em1.pt'), weights_only=True)
        checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
        self.smirk_encoder.load_state_dict(checkpoint_encoder)
    
    def forward(self, image: Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # orig_image_height, orig_image_width, _ = image.shape
        kpt_mediapipe = run_mediapipe(image)
        kpt_mediapipe = kpt_mediapipe[..., :2]
        tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=self.image_size)
        cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
        cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
        cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(self.device)

        outputs = self.smirk_encoder(cropped_image)
        return outputs

class SMIRKDecoder(nn.Module):
    def __init__(self, device = 'cpu', suppress_pose=False):
        super().__init__()
        self.device = device
        self.suppress_pose = suppress_pose
        self.flame = FLAME().to(device)
        self.renderer = Renderer().to(device)
    def forward(self, codedict):
        _pose_params = codedict['pose_params']
        _shape_params = codedict['shape_params']
        _cam = codedict['cam']
        if self.suppress_pose:
            codedict['shape_params'] = torch.zeros_like(_shape_params)
            codedict['pose_params'] = torch.tensor([[0., 0., 0.]], device=self.device)
            codedict['cam'] = torch.tensor([[8., 0., 0.]], device=self.device)
        flame_output = self.flame.forward(codedict)
        renderer_output = self.renderer.forward(flame_output['vertices'], codedict['cam'],
                                        landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        
        codedict['pose_params'] = _pose_params
        codedict['shape_params'] = _shape_params
        codedict['cam'] = _cam
        return flame_output, renderer_output