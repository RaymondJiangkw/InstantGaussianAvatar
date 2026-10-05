# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from transformers import AutoModelForImageSegmentation

import os
import cv2
import dlib
import yaml
import numpy as np
from PIL import Image
from gsplat import rasterization

from .third_party.FaceBoxes import FaceBoxes
from .third_party.TDDFA.TDDFA import TDDFA

from typing import Union

# __all__ = ["matting", "crop", "render_tensor", "to_tensor", "animate_gaussians"]

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

def to_tensor(img: Union[Image.Image, np.ndarray], normalize=True) -> torch.Tensor:
    if isinstance(img, Image.Image):
        img = np.array(img)
        if len(img.shape) > 2:
            img = img.transpose(2, 0, 1)
        else:
            img = img[None, ...]
    else:
        if img.shape[0] == img.shape[1]:
            img = img.transpose(2, 0, 1)
    if normalize:
        img = torch.from_numpy(img).to(torch.float32) / 127.5 - 1
    else:
        img = torch.from_numpy(img).to(torch.float32) / 255.
    return img[None, ...].to(device)

device = "cuda"
BIREFNET = None
SIZE = 512
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), "../checkpoints/shape_predictor_68_face_landmarks.dat"))
cfg = yaml.load(open(os.path.join(os.path.dirname(__file__), '../checkpoints/mb1_120x120.yml')), Loader=yaml.SafeLoader)
tddfa = TDDFA(gpu_mode=False, **cfg)
face_boxes = FaceBoxes()

def get_birefnet():
    global BIREFNET
    if BIREFNET is None:
        print("One time load birefnet...", end='')
        BIREFNET = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_HR-matting', trust_remote_code=True).to(device).eval().half()
        print("Done.")
    return BIREFNET

@torch.no_grad()
def matting(image, normalize=True, return_pred=False):
    org_shape = image.shape[-2:]
    if normalize:
        image = image / 2 + .5
    org_image = image
    image = (image - torch.tensor([0.485, 0.456, 0.406], device=image.device)[None, :, None, None]
            ) / torch.tensor([0.229, 0.224, 0.225], device=image.device)[None, :, None, None]
    # image = torch.nn.functional.interpolate(image, (2048, 2048), mode='bilinear', align_corners=False, antialias=True)
    image = image.half()
    pred = get_birefnet()(image)[-1].sigmoid()
    pred = torch.nn.functional.interpolate(pred, org_shape, mode='bicubic', align_corners=False)
    if return_pred:
        return pred
    image = org_image * pred
    if normalize:
        image = image * 2 - 1
    return image


# Image Crop Utility

def find_center_bbox(roi_box_lst, w, h):
    bboxes = np.array(roi_box_lst)
    dx = 0.5*(bboxes[:,0] + bboxes[:,2]) - 0.5*(w-1)
    dy = 0.5*(bboxes[:,1] + bboxes[:,3]) - 0.5*(h-1)
    dist = np.stack([dx,dy],1)
    return np.argmin(np.linalg.norm(dist, axis=1))

def crop_final(
    img, 
    size=512, 
    quad=None,
    top_expand=0., 
    left_expand=0., 
    bottom_expand=0., 
    right_expand=0., 
    blur_kernel=None,
    borderMode=cv2.BORDER_REFLECT,
    upsample=2,
    min_size=512,
):  
    orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    if min_size is not None and orig_size < min_size:
        pass

    crop_w = int(size * (1 + left_expand + right_expand))
    crop_h = int(size * (1 + top_expand + bottom_expand))
    crop_size = (crop_w, crop_h)
    
    top = int(size * top_expand)
    left = int(size * left_expand)
    size -= 1
    bound = np.array([[left, top], [left, top + size], [left + size, top + size], [left + size, top]],
                        dtype=np.float32)

    mat = cv2.getAffineTransform(quad[:3], bound[:3])
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 

    empty = np.ones_like(img) * 255
    crop_mask = cv2.warpAffine(empty, mat, crop_size)

    if True:
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1 if blur_kernel is None else blur_kernel
        downsample_size = (crop_w//8, crop_h//8)
        
        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2),(mask_kernel,mask_kernel)) / 255.0
            blur_mask = blur_mask[...,np.newaxis]#.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)
    
    return crop_img

def get_crop_bound(lm, method="ffhq"):
    if len(lm) == 106:
        left_e = lm[104]
        right_e = lm[105]
        nose = lm[49]
        left_m = lm[84]
        right_m = lm[90]
        center = (lm[1] + lm[31]) * 0.5
    elif len(lm) == 68:
        left_e = np.mean(lm[36:42], axis=0)
        right_e = np.mean(lm[42:48], axis=0)
        nose = lm[33]
        left_m = lm[48]
        right_m = lm[54]
        center = (lm[0] + lm[16]) * 0.5
    else:
        raise ValueError(f"Unknown type of keypoints with a length of {len(lm)}")

    if method == "ffhq":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        mouth_avg = (left_m + right_m) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
    elif method == "default":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        eye_to_nose = nose - eye_avg
        x = eye_to_eye.copy()
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.4, np.hypot(*eye_to_nose) * 2.75)
        y = np.flipud(x) * [-1, 1]
        c = center
    else:
        raise ValueError('%s crop method not supported yet.' % method)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    return quad.astype(np.float32), c, x, y

def crop_image(img, mat, crop_w, crop_h, upsample=1, borderMode=cv2.BORDER_CONSTANT):
    crop_size = (crop_w, crop_h)
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 
    return crop_img

def eg3dcamparams(R_in):
    camera_dist = 2.7
    intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
    # assume inputs are rotation matrices for world2cam projection
    R = np.array(R_in).astype(np.float32).reshape(4,4)
    # add camera translation
    t = np.eye(4, dtype=np.float32)
    t[2, 3] = - camera_dist

    # convert to OpenCV camera
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)

    # world2cam -> cam2world
    P = convert @ t @ R
    cam2world = np.linalg.inv(P)

    # add intrinsics
    label_new = np.concatenate([cam2world.reshape(16), intrinsics.reshape(9)], -1).tolist()
    return label_new

def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + np.arctan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + np.arctan2(R[0, 1], R[0, 2])
    else:
        x = np.arcsin(R[2, 0])
        y = np.arctan2(R[2, 1] / np.cos(x), R[2, 2] / np.cos(x))
        z = np.arctan2(R[1, 0] / np.cos(x), R[0, 0] / np.cos(x))

    return x, y, z

def crop(image, rect_idx=None, load_device="cuda", return_quad=False):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if rect_idx is None:
        rect = max(rects, key=lambda x: abs((x.right() - x.left()) * (x.top() - x.bottom())))
    else:
        rect = rects[rect_idx]
    shape = predictor(gray, rect)
    landmark = [np.array([p.x, p.y]) for p in shape.parts()]
    quad, quad_c, quad_x, quad_y = get_crop_bound(landmark)
    bound = np.array([[0, 0], [0, SIZE-1], [SIZE-1, SIZE-1], [SIZE-1, 0]], dtype=np.float32)
    mat = cv2.getAffineTransform(quad[:3], bound[:3])
    img = crop_image(image, mat, SIZE, SIZE)

    h, w = img.shape[:2]
    boxes = face_boxes(img)
    param_lst, roi_box_lst = tddfa(img, boxes)
    box_idx = find_center_bbox(roi_box_lst, w, h)

    param = param_lst[box_idx]
    P = param[:12].reshape(3, -1)  # camera matrix
    s_relative, R, t3d = P2sRt(P)
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]

    # Adjust z-translation in object space
    R_ = param[:12].reshape(3, -1)[:, :3]
    u = tddfa.bfm.u.reshape(3, -1, order='F')
    trans_z = np.array([ 0, 0, 0.5*u[2].mean() ]) # Adjust the object center
    trans = np.matmul(R_, trans_z.reshape(3,1))
    t3d += trans.reshape(3)

    ''' Camera extrinsic estimation for GAN training '''
    # Normalize P to fit in the original image (before 3DDFA cropping)
    sx, sy, ex, ey = roi_box_lst[0]
    scale_x = (ex - sx) / tddfa.size
    scale_y = (ey - sy) / tddfa.size
    t3d[0] = (t3d[0]-1) * scale_x + sx
    t3d[1] = (tddfa.size-t3d[1]) * scale_y + sy
    t3d[0] = (t3d[0] - 0.5*(w-1)) / (0.5*(w-1)) # Normalize to [-1,1]
    t3d[1] = (t3d[1] - 0.5*(h-1)) / (0.5*(h-1)) # Normalize to [-1,1], y is flipped for image space
    t3d[1] *= -1
    t3d[2] = 0 # orthogonal camera is agnostic to Z (the model always outputs 66.67)

    s_relative = s_relative * 2000
    scale_x = (ex - sx) / (w-1)
    scale_y = (ey - sy) / (h-1)
    s = (scale_x + scale_y) / 2 * s_relative
    # print(f"[{iteration}] s={s} t3d={t3d}")

    quad_c = quad_c + quad_x * t3d[0]
    quad_c = quad_c - quad_y * t3d[1]
    quad_x = quad_x * s
    quad_y = quad_y * s
    c, x, y = quad_c, quad_x, quad_y
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)
    orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))

    s = 1
    t3d = 0 * t3d
    R[:,:3] = R[:,:3] * s
    P = np.concatenate([R,t3d[:,None]],1)
    P = np.concatenate([P, np.array([[0,0,0,1.]])], 0)

    pose = eg3dcamparams(P.flatten())
    cropped_img = crop_final(image, size=SIZE, quad=quad)
    if not return_quad:
        return matting((torch.from_numpy(cropped_img).permute(2, 0, 1)[None].to(torch.float32) / 127.5 - 1.).to(load_device)), torch.as_tensor(pose).float().reshape(1, -1).to(load_device)
    else:
        return matting((torch.from_numpy(cropped_img).permute(2, 0, 1)[None].to(torch.float32) / 127.5 - 1.).to(load_device)), torch.as_tensor(pose).float().reshape(1, -1).to(load_device), quad

# Wide Crop

from src.third_party.hrn_models.hrn import Reconstructor

params = [
    '--checkpoints_dir', '../../checkpoints/',
    '--name', 'hrn_v1.1',
    '--epoch', '10',
]

reconstructor = Reconstructor(params)

def get_HRN_bbox(img, _crop_factor=256, _rotatepoint='0,-.2,-1.2'):
    if isinstance(img, Image.Image):
        image = np.array(image)
    img = img[:, :, ::-1] # convert rgb to bgr

    landmarks = None # auto detect lm
    output = reconstructor.predict(img, landmarks, visualize=False)

    lm_68p_vertex = reconstructor.model.facemodel_front.get_landmarks(output['face_vertices']) # [B, 68, 3]
    lm_68p_vertex = lm_68p_vertex.squeeze(0)

    eyes = lm_68p_vertex[36:42].mean(dim=0), lm_68p_vertex[42:48].mean(dim=0)
    eye_midpoint = (eyes[0] + eyes[1])/2 # find the centerpoint of eyes to get center point
    chin_midpoint = lm_68p_vertex[6:11].mean(dim=0)
    x = eyes[1] - eyes[0]
    x = x / x.norm()
    y = eye_midpoint - chin_midpoint
    y = y / y.norm()
    z = -torch.cross(x, y)

    brow_mid = lm_68p_vertex[19:25].mean(dim=0)
    nose_mid = lm_68p_vertex[31:36].mean(dim=0)
    unit_dist = (brow_mid - nose_mid).norm()
    rotatepoint_world = [float(x) for x in _rotatepoint.split(',')] # convert '0,1,-.2' -> [0, 1, -.2]
    rotatepoint = eye_midpoint + z * unit_dist * rotatepoint_world[2] + y * unit_dist * rotatepoint_world[1] + x * unit_dist * rotatepoint_world[0]

    rotatepoint = reconstructor.model.facemodel_front.to_image(rotatepoint.reshape(1, 1, 3)).squeeze()
    rotatepoint = rotatepoint.cpu().numpy()

    w, h, s, tx, ty = output['trans_params'].tolist()
    # print(output['trans_params'])

    rotatepoint_original = rotatepoint + np.array([(int(w*s)/2 - 224/2), (int(h*s)/2 - 224/2)]) # see HRN/util/preprocess.py#137,139,151
    rotatepoint_original = rotatepoint_original / float(s)
    rotatepoint_original[0] += float(tx) - w/2
    rotatepoint_original[1] += float(ty) - h/2
    rotatepoint_original[1] = h - 1 - rotatepoint_original[1]

    bbox_halfsize = int(unit_dist / s * _crop_factor)
    l = rotatepoint_original[0] - bbox_halfsize
    r = rotatepoint_original[0] + bbox_halfsize
    t = rotatepoint_original[1] - bbox_halfsize
    b = rotatepoint_original[1] + bbox_halfsize
    crop_bbox = l.item(), t.item(), r.item(), b.item()

    return crop_bbox

def wide_crop(image: Image.Image):
    crop_bbox = get_HRN_bbox(np.array(image), 256, '0,-.2,-1.2') # type: ignore
    img = to_tensor(image.crop(crop_bbox).resize(size=(512, 512), resample=Image.LANCZOS), normalize=False)
    img = matting(img, normalize=False)
    return img * 2 - 1, None

# Inference-only Fast Animation
@torch.compile()
@torch.no_grad()
def animate_gaussians(
    E: nn.Module, 
    gaussian_dict: dict, 
    c: torch.Tensor, 
    source_motion_latents: torch.Tensor = None, 
    target_motion_latents: torch.Tensor = None, 
    render_depth: bool = False, 
    render_feature: bool = False, 
):
    # Prepare Rasterization Parameters
    cam2world_matrix = c[:, :16].view(1, 4, 4)
    intrinsics = c[:, 16:25].view(1, 3, 3)
    H = W = 512
    
    viewmatrix = torch.linalg.inv(cam2world_matrix[:1])
    intrinsic = intrinsics * H
    intrinsic[:, -1, -1] = 1

    # Transform Gaussians
    xyzs = gaussian_dict['xyz'][0]
    colors = gaussian_dict['color'][0][:, :3]
    opacities = 1 - torch.exp(-gaussian_dict['density'][0]).squeeze(1)
    scalings = gaussian_dict['scaling'][0]
    rotations = gaussian_dict['rotation'][0]
    
    motions = gaussian_dict['motion'][0]
    features = gaussian_dict['feature'][0]
    
    if source_motion_latents is not None and target_motion_latents is not None:
        source_to_target = E.motion_decoder(
            motions[None], source_motion_latents, target_motion_latents
        )
        features = features + source_to_target[0]
        out = E.renderer.instantiater.instantiate_from_features(E.decoder, [ features ])
        xyzs = out['xyz'][0]
        colors = out['color'][0][:, :3]
        opacities = 1 - torch.exp(-out['density'][0]).squeeze(1)
        scalings = out['scaling'][0]
        rotations = out['rotation'][0]
    
    _rgb, alpha, _ = rasterization(
        means=xyzs, 
        quats=rotations, 
        scales=scalings, 
        opacities=opacities, 
        colors=colors if not render_feature else motions, 
        viewmats=viewmatrix, 
        Ks=intrinsic, 
        width=W, 
        height=H, 
        packed=False, 
        absgrad=False, 
        sparse_grad=False, 
        rasterize_mode=E.rendering_kwargs["rasterize_mode"], 
        render_mode="RGB+ED", 
        sh_degree=None, 
    )
    if render_feature:
        return _rgb[..., :-1]
    rgb = _rgb[0, ..., :3].permute(2, 0, 1) * 2 - 1
    depth = _rgb[0, ..., 3:].permute(2, 0, 1)
    if render_depth:
        return rgb, depth, alpha
    return rgb