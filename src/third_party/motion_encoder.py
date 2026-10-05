# *************************************************************************
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under PD-FGC, with the full license text
# available at https://github.com/Dorniwang/PD-FGC-inference/blob/main/LICENSE.
#
# This modified file is released under the same license.
# *************************************************************************
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FAN_feature_extractor import FAN_SA
from einops import rearrange
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid
from diffusers.models.modeling_utils import ModelMixin

from PIL import Image
import mediapipe as mp
from torchvision.utils import make_grid

@torch.no_grad()
def render_tensor(img: torch.Tensor, normalize: bool = True, nrow: int = 8) -> Image.Image:
    if type(img) == list:
        img = torch.cat(img, dim=0).expand(-1, 3, -1, -1)
    elif len(img.shape) == 3:
        img = img.expand(3, -1, -1)
    elif len(img.shape) == 4:
        img = img.expand(-1, 3, -1, -1)
    
    img = img.squeeze()
    
    if normalize:
        img = img / 2 + .5
    
    if len(img.shape) == 3:
        return Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 2:
        return Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 4:
        return Image.fromarray((make_grid(img, nrow=nrow).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

class MotEncoder_withExtra(ModelMixin):
    def __init__(self, extra_feat_dim=3, out_ch=16):
        super(MotEncoder_withExtra, self).__init__()
        self.model = FAN_SA()
        self.out_drop = None #nn.Dropout(p=0.4)
        self.out_ch = out_ch
        expr_dim = 512
        extra_pos_embed = get_1d_sincos_pos_embed_from_grid(out_ch, np.arange(expr_dim//out_ch))
        self.register_buffer("pe", torch.from_numpy(extra_pos_embed).float().unsqueeze(0))
        self.bbox_proj = nn.Sequential(
            nn.Linear(extra_feat_dim, 64),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(64, 64),
        )
        self.final_proj = nn.Linear(expr_dim + 64, expr_dim)
        self.out_bn = None

    def change_out_dim(self, out_ch):
        self.out_proj = nn.Linear(self.out_ch, out_ch)

    def forward(self, x, emb):
        if x.ndim == 5:
            latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
            emb = rearrange(emb, "b f c -> (b f) c")
            vid_len = x.shape[2]
        else:
            vid_len = 1
            latent = self.model(x)  # [B, 512]
        if self.out_bn is not None:
            latent = self.out_bn(latent.unsqueeze(-1)).squeeze(-1)        #####
        if self.out_drop is not None:
            latent = self.out_drop(latent)
        face_mot_feat = latent.clone()
        latent = torch.cat([latent, self.bbox_proj(emb)], dim=1)
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        
        if x.ndim == 5:
            latent = rearrange(latent, "(b f) l c -> b f l c", f=x.shape[2])
            face_mot_feat = rearrange(face_mot_feat, "(b f) c -> b f c", f=x.shape[2])

        return latent

    def encode_facemot(self, x):
        if x.ndim == 5:
            latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
        else:
            latent = self.model(x)  # [B, 512]
        if self.out_bn is not None:
            latent = self.out_bn(latent.unsqueeze(-1)).squeeze(-1)  #####
        if x.ndim == 5:
            latent = rearrange(latent, "(b f) c -> b f c", f=x.shape[2])
        return latent

    def decode_facemot(self, mot_tok, emb):
        pred_motion = self.vqvae.forward_decoder(mot_tok)
        b, t = pred_motion.shape[:2]
        c = 512
        split_exp_num = c // self.vqvae.vqvae.input_emb_width
        latent = pred_motion.reshape(b, t, split_exp_num, c // split_exp_num).reshape(b * t, c)

        emb = rearrange(emb.repeat(1, t, 1), "b f c -> (b f) c")
        latent = torch.cat([latent, self.bbox_proj(emb)], dim=1)
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        if self.out_proj is not None:
            latent = self.out_proj(latent)

        latent = rearrange(latent, "(b f) l c -> b f l c", f=t)
        return latent

    def fuse_emb(self, x, emb):
        if x.ndim == 3:
            latent = rearrange(x, "b f c -> (b f) c")
            emb = rearrange(emb, "b f c -> (b f) c")
        else:
            latent = x
        latent = torch.cat([latent, self.bbox_proj(emb)], dim=1)
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        if self.out_proj is not None:
            latent = self.out_proj(latent)

        if x.ndim == 3:
            latent = rearrange(latent, "(b f) l c -> b f l c", f=x.shape[1])

        return latent

def extract_bbox_mp(frame, refbbox, detector, timestamp_ms=None):
    if refbbox is None:
        refbbox = (0, 0, frame.shape[1], frame.shape[0])
    
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame, dtype=np.uint8).copy())

    if timestamp_ms is None:
        detection_result = detector.detect(image)
    else:
        detection_result = detector.detect_for_video(image, timestamp_ms)
    if len(detection_result.detections) > 0:  # 有多个人脸
        bbox = [
                (
                    detection.categories[0].score,
                    (
                        detection.bounding_box.origin_x,
                        detection.bounding_box.origin_y,
                        detection.bounding_box.origin_x + detection.bounding_box.width,
                        detection.bounding_box.origin_y + detection.bounding_box.height,
                    ),  # LEFT,TOP,RIGHT,BOT
                )
                for detection in detection_result.detections
            ]

        if len(bbox) > 1 and min(bbox)[0] > 0.8:
            return "Detect Multiple faces!"
        bbox = np.array(max(bbox)[1])   # 根据给定的画面范围，找出最显著的人脸
    else:
        return "Detect No face!"
    return np.maximum(bbox, 0)  # bbox 范围不能小于零

def scale_bb(bbox, scale, size):
    left, top, right, bot = bbox[:4].tolist()
    width = right - left
    height = bot - top
    length = max(width, height) * scale
    center_X = (left + right) * 0.5
    center_Y = (top + bot) * 0.5
    left, top, right, bot = [
        center_X - length / 2,
        center_Y - length / 2,
        center_X + length / 2,
        center_Y + length / 2,
    ]
    if left < 0 or top < 0 or right > size[1] - 1 or bot > size[0] - 1:
        return bbox
    else:
        return np.array([left, top, right, bot])

def check_oob(bbox, size):
    left, top, right, bot = bbox
    return left < 0 or top < 0 or right > size[1] - 1 or bot > size[0] - 1

def get_bbox_from_center(center, length, size):
    center_X, center_Y = center
    w, h = length
    left, top, right, bot = [
        center_X - w / 2,
        center_Y - h / 2,
        center_X + w / 2,
        center_Y + h / 2,
    ]
    if check_oob((left, top, right, bot), size):
        x_offset = max(-left, 0) + min((size[1] - 1 - right), 0)
        y_offset = max(-top, 0) + min((size[0] - 1 - bot), 0)

        return np.array([left, top, right, bot]) + np.array(
            [x_offset, y_offset, x_offset, y_offset]
        )
    else:
        return np.array([left, top, right, bot])

class MotEncoder(nn.Module):
    def __init__(self, device, wide_crop=False):
        super().__init__()
        self.model = MotEncoder_withExtra().to(device).eval()

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self.bbox_param = torch.tensor([0., 0., 1.]).to(device)[None]
        self.device = device
        self.wide_crop = wide_crop
    def detect_bbox(self, frame: torch.Tensor):
        assert len(frame.shape) == 3 or len(frame) == 1
        img = np.array(render_tensor(frame))
        bbox = extract_bbox_mp(img, None, self.detector)
        bbox = scale_bb(bbox, scale=1.1, size=img.shape[:2])
        fix_length = (
            round(bbox[2] - bbox[0]) // 2 * 2,
            round(bbox[3] - bbox[1]) // 2 * 2,
        )
        left, top, right, bot = bbox
        center_X = (left + right) * 0.5
        center_Y = (top + bot) * 0.5

        bbox = get_bbox_from_center(
            np.array([center_X, center_Y]), fix_length, img.shape[:2]
        )

        return bbox
    def crop_frame(self, img: torch.Tensor, bbox):
        left, top, right, bot = bbox
        img = img[..., int(top) : int(bot), int(left) : int(right)]
        img = torch.nn.functional.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
        return img
    def pure_encode(self, img: torch.Tensor):
        return self.model(img, self.bbox_param).reshape(1, -1)
    def robust_crop(self, frame: torch.Tensor):
        assert len(frame.shape) == 3 or len(frame) == 1
        original_frame = frame
        try:
            img = np.array(render_tensor(frame))
            bbox = extract_bbox_mp(img, None, self.detector)
            bbox = scale_bb(bbox, scale=1.1, size=img.shape[:2])
            fix_length = (
                round(bbox[2] - bbox[0]) // 2 * 2,
                round(bbox[3] - bbox[1]) // 2 * 2,
            )
            left, top, right, bot = bbox
            center_X = (left + right) * 0.5
            center_Y = (top + bot) * 0.5

            bbox = get_bbox_from_center(
                np.array([center_X, center_Y]), fix_length, img.shape[:2]
            )

            left, top, right, bot = bbox
            if int(bot) - int(top) <= 0 or int(right) - int(left) <= 0:
                # Degenerate bbox; force fallback path.
                raise ValueError("Degenerate face bbox")
            if len(frame.shape) == 3:
                frame = frame[None]
            cropped = frame[:, :, int(top) : int(bot), int(left) : int(right)]
            img = torch.nn.functional.interpolate(cropped, (224, 224), mode='bilinear', align_corners=False, antialias=True)
        except Exception:
            frame = original_frame
            if len(frame.shape) == 3:
                frame = frame[None]
            if self.wide_crop:
                frame = torch.nn.functional.interpolate(
                    frame[:, :, 90:, 45:-45], (512, 512), mode='bilinear', align_corners=False, antialias=True
                )
            frame = torch.nn.functional.interpolate(frame, (256, 256), mode='bilinear', align_corners=False, antialias=True)
            frame = frame[:, :, 35:223, 32:220]
            img = torch.nn.functional.interpolate(frame, (224, 224), mode='bilinear', align_corners=False, antialias=True)
        return img
    def forward(self, x: torch.Tensor):
        latents = []
        for b in range(len(x)):
            frame = x[b]
            try:
                img = np.array(render_tensor(frame))
                bbox = extract_bbox_mp(img, None, self.detector)
                bbox = scale_bb(bbox, scale=1.1, size=img.shape[:2])
                fix_length = (
                    round(bbox[2] - bbox[0]) // 2 * 2,
                    round(bbox[3] - bbox[1]) // 2 * 2,
                )
                left, top, right, bot = bbox
                center_X = (left + right) * 0.5
                center_Y = (top + bot) * 0.5

                bbox = get_bbox_from_center(
                    np.array([center_X, center_Y]), fix_length, img.shape[:2]
                )

                left, top, right, bot = bbox
                img = img[int(top) : int(bot), int(left) : int(right)]
                img = (torch.from_numpy(img).to(self.device).permute(2, 0, 1).to(torch.float32) / 127.5 - 1.)[None]
                img = torch.nn.functional.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
            except:
                # Fall back
                if len(frame.shape) == 3:
                    frame = frame[None]
                if self.wide_crop:
                    frame = torch.nn.functional.interpolate(
                        frame[:, :, 90:, 45:-45], (512, 512), mode='bilinear', align_corners=False, antialias=True
                    )
                frame = torch.nn.functional.interpolate(frame, (256, 256), mode='bilinear', align_corners=False, antialias=True)
                frame = frame[:, :, 35:223, 32:220]
                img = torch.nn.functional.interpolate(frame, (224, 224), mode='bilinear', align_corners=False, antialias=True)
            
            latents.append(self.model(img, self.bbox_param))
        return torch.cat(latents).reshape(-1, 16 * 32)