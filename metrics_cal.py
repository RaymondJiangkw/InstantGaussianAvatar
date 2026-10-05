# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from src.utils import *

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from met3r import MEt3R
from lpips import LPIPS
from torchvision import transforms

from src.third_party.MagFace import MagFace
from src.third_party.smirk import SMIRKEncoder
from src.third_party.face_parsing.model import BiSeNet
from src.third_party.emo_detector.model import EmoNetEncoder
from src.third_party.face_evolve.applications.align.detector import detect_faces
from src.third_party.face_evolve.applications.align.align_trans import get_reference_facial_points, warp_and_crop_face

mag_face = MagFace("cuda")
emonet = EmoNetEncoder()
met3r = MEt3R(
    img_size=256, # Default to 256, set to `None` to use the input resolution on the fly!
    use_norm=True, # Default to True 
    backbone="mast3r", # Default to MASt3R, select from ["mast3r", "dust3r", "raft"]
    feature_backbone="dino16", # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
    feature_backbone_weights="mhamilton723/FeatUp", # Default
    upsampler="featup", # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
    distance="rmse", # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
    freeze=True, # Default to True
).to(device).eval().requires_grad_(False)
lpips_fn = LPIPS(net='vgg').to(device)

face2seg_ = BiSeNet(n_classes=19).to(device).eval()
face2seg_.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "checkpoints", "face2seg.pth"), map_location=device, weights_only=True))
face2seg = lambda x: face2seg_(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(transforms.ToTensor()(x)[None].to(device)))[0]

smirk = SMIRKEncoder(device)

reference = get_reference_facial_points(default_square = True)
def get_face_feat(img):
    _, landmarks = detect_faces(img)
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(112, 112))
    img_warped = Image.fromarray(warped_face)
    feat = mag_face.model(to_tensor(img_warped))
    return feat / (torch.norm(feat, p=2, dim=-1, keepdim=True) + 1e-8)

def eval_crop(image, return_quad=False, quad=None):
    image = np.array(image)

    if quad is not None:
        cropped_img = crop_final(image, size=SIZE, quad=quad)
        if not return_quad:
            return Image.fromarray(cropped_img), None
        else:
            return Image.fromarray(cropped_img), None, quad

    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    rect = max(rects, key=lambda x: abs((x.right() - x.left()) * (x.top() - x.bottom())))
    shape = predictor(gray, rect)
    landmark = [np.array([p.x, p.y]) for p in shape.parts()]
    quad, quad_c, quad_x, quad_y = get_crop_bound(landmark)
    bound = np.array([[0, 0], [0, SIZE-1], [SIZE-1, SIZE-1], [SIZE-1, 0]], dtype=np.float32)
    mat = cv2.getAffineTransform(quad[:3], bound[:3])
    # if disable_rotation:
    #     A, t = mat[:, :2], mat[:, 2]
    #     U, s, Vt = np.linalg.svd(A)
    #     S = Vt.T @ np.diag(s) @ Vt
    #     A_no_rot = S
    #     mat = np.hstack([A_no_rot, t.reshape(2,1)])
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
        return Image.fromarray(cropped_img), pose
    else:
        return Image.fromarray(cropped_img), pose, quad

C1 = 0.01 ** 2
C2 = 0.03 ** 2

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def face2mask(image: Image.Image):
    seg_idx = face2seg(image.resize((512, 512), resample=Image.Resampling.LANCZOS)).argmax(dim=1, keepdims=True)
    mask = None
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        m = seg_idx == i
        if mask is None:
            mask = m
        else:
            mask = torch.logical_or(mask, m)
    return torch.nn.functional.interpolate(mask.float(), (image.size[1], image.size[0]), mode='bilinear', align_corners=False, antialias=True)

def face2mask_full(image: Image.Image):
    seg_idx = face2seg(image.resize((512, 512), resample=Image.Resampling.LANCZOS)).argmax(dim=1, keepdims=True)
    mask = None
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]:
        m = seg_idx == i
        if mask is None:
            mask = m
        else:
            mask = torch.logical_or(mask, m)
    return torch.nn.functional.interpolate(mask.float(), (image.size[1], image.size[0]), mode='bilinear', align_corners=False, antialias=True)

def binary_iou(pred: torch.Tensor, target: torch.Tensor, threshold=0.5, eps=1e-8):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    
    return intersection / (union + eps)

def preprocess_image(image: Image.Image, size=None, recrop=True, quad=None, apply_matting='matting'):
    if recrop:
        try:
            image, _, quad = eval_crop(np.array(image), quad=quad, return_quad=True)
        except Exception as e:
            # raise e
            return None, None
    if size is not None:
        image = image.resize((size, size), resample=Image.Resampling.LANCZOS)
    if apply_matting == 'matting':
        image = matting(to_tensor(np.array(image), normalize=False), normalize=False)
    elif apply_matting == 'face':
        image = to_tensor(np.array(image), normalize=False) * face2mask(image)
    elif apply_matting == 'full_head':
        image = to_tensor(np.array(image), normalize=False) * face2mask_full(image)
    elif apply_matting == 'mask':
        image = matting(to_tensor(np.array(image), normalize=False) * face2mask_full(image), normalize=False, return_pred=True)
    else:
        image = to_tensor(np.array(image), normalize=False)
    return render_tensor(image, normalize=False), quad

@torch.no_grad()
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

@torch.no_grad()
def calc_metrics(source_image: Image.Image, driving_image: Image.Image, first_frame: Image.Image, pred_image: Image.Image, emo_driving_valence=[], emo_pred_valence=[], emo_driving_arousal=[], emo_pred_arousal=[], disable_metrics = []):
    source_tensor = to_tensor(source_image, normalize=False)
    driving_tensor = to_tensor(driving_image, normalize=False)
    pred_tensor = to_tensor(pred_image, normalize=False)
    first_tensor = to_tensor(first_frame, normalize=False)

    smirk_fail = True
    if not 'smirk' in disable_metrics:
        try:
            # source_codedict = smirk(source_image)
            driving_codedict = smirk(driving_image)
            pred_codedict = smirk(pred_image)

            smirk_fail = False
        except:
            pass
    
    m_iou = 0.0
    if not 'iou' in disable_metrics:
        driving_mask = to_tensor(preprocess_image(driving_image, recrop=False, apply_matting='mask')[0], normalize=False)
        pred_mask = to_tensor(preprocess_image(pred_image, recrop=False, apply_matting='mask')[0], normalize=False)
        m_iou = float(binary_iou(driving_mask, pred_mask, threshold=0.01))
    
    m_psnr = float(psnr(driving_tensor, pred_tensor).mean()) if not 'psnr' in disable_metrics else 0.0
    m_lpips = float(lpips_fn(driving_tensor, pred_tensor, normalize=True).reshape(-1)) if not 'lpips' in disable_metrics else 0.0
    m_ssim = float(ssim(driving_tensor, pred_tensor).reshape(-1)) if not 'ssim' in disable_metrics else 0.0
    m_consistency = float(met3r(
        images=torch.cat((first_tensor, pred_tensor), dim=0)[None], 
        return_overlap_mask=False,
        return_score_map=False,
        return_projections=False
    )[0].mean()) if not 'consistency' in disable_metrics else 0.0
    
    if not 'emo' in disable_metrics:
        try:
            driving_emo = emonet(driving_image)
            pred_emo = emonet(pred_image)

            emo_driving_valence.append(driving_emo['valence'].item())
            emo_driving_arousal.append(driving_emo['arousal'].item())

            emo_pred_valence.append(pred_emo['valence'].item())
            emo_pred_arousal.append(pred_emo['arousal'].item())
        except:
            pass
    
    id_fail = True
    if not 'id' in disable_metrics:
        try:
            # m_id = float(IDLoss()(source_tensor * 2 - 1, pred_tensor * 2 - 1).reshape(-1))
            m_id = float(torch.nn.CosineSimilarity(dim=-1)(
                get_face_feat(source_image).detach().reshape(-1), 
                get_face_feat(pred_image).detach().reshape(-1)
            ).reshape(-1))
            id_fail = False
        except:
            pass
    
    ret = {}
    if not 'psnr' in disable_metrics:
        ret.update({"PSNR": m_psnr})
    if not 'lpips' in disable_metrics:
        ret.update({"LPIPS": m_lpips})
    if not 'ssim' in disable_metrics:
        ret.update({"SSIM": m_ssim})
    if not 'consistency' in disable_metrics:
        ret.update({"CONSISTENCY": m_consistency})
    if not 'iou' in disable_metrics:
        ret.update({"IOU": m_iou})
    if not 'smirk' in disable_metrics:
        ret.update({
            "ED": (driving_codedict['expression_params'].reshape(-1) - pred_codedict['expression_params'].reshape(-1)).square().mean().sqrt().item() if not smirk_fail else 0.0, 
            "PD": (driving_codedict['pose_params'].reshape(-1)[:3] - pred_codedict['pose_params'].reshape(-1)[:3]).square().mean().sqrt().item() if not smirk_fail else 0.0, 
        })
    if not 'id' in disable_metrics:
        ret.update({"ID": m_id})

    return ret

def PCC(ground_truth, predictions):
    return np.corrcoef(ground_truth, predictions)[0,1]

def CCC(ground_truth, predictions):
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)

def batch_calc_metrics(source_image_path: str, driving_image_path_s: list, pred_image_path_s: list, disable_metrics = [], recrop=True, sample_interval=5, force_recrop=False, **kwargs):
    assert len(driving_image_path_s) == len(pred_image_path_s)

    metrics = []

    source_image, _ = preprocess_image(Image.open(source_image_path), recrop=recrop, **kwargs)

    if source_image is None:
        return {}

    emo_driving_valence = []
    emo_pred_valence    = []
    emo_driving_arousal = []
    emo_pred_arousal    = []

    drive_quad = None
    pred_quad = None
    first_frame = None

    for driving_image_path, pred_image_path in (list(zip(driving_image_path_s, pred_image_path_s))[::sample_interval]):
        driving_image, drive_quad = preprocess_image(Image.open(driving_image_path), recrop=recrop, quad=drive_quad if not force_recrop else None, **kwargs)
        pred_image, pred_quad = preprocess_image(Image.open(pred_image_path), recrop=recrop, quad=pred_quad if not force_recrop else None, **kwargs)
        
        if first_frame is None:
            first_frame = pred_image
        if driving_image is None or pred_image is None:
            return {}
        metrics.append(calc_metrics(source_image, driving_image, first_frame, pred_image, emo_driving_valence=emo_driving_valence, emo_pred_valence=emo_pred_valence, emo_driving_arousal=emo_driving_arousal, emo_pred_arousal=emo_pred_arousal, disable_metrics=disable_metrics))
    
    final_metrics = {}
    for metric in metrics:
        for metric_name in metric:
            if not metric_name in final_metrics:
                final_metrics[metric_name] = []
            final_metrics[metric_name].append(metric[metric_name])
    
    emo_driving_valence = np.array(emo_driving_valence)
    emo_pred_valence    = np.clip(np.array(emo_pred_valence), -1.0, 1.0)
    emo_driving_arousal = np.array(emo_driving_arousal)
    emo_pred_arousal    = np.clip(np.array(emo_pred_arousal), -1.0, 1.0)

    if not "emo" in disable_metrics:
        valence_PCC = PCC(emo_driving_valence, emo_pred_valence)
        valence_CCC = CCC(emo_driving_valence, emo_pred_valence)
        arousal_PCC = PCC(emo_driving_arousal, emo_pred_arousal)
        arousal_CCC = CCC(emo_driving_arousal, emo_pred_arousal)
        EMO = 8 / ((1 / valence_PCC) + (1 / valence_CCC) + (1 / arousal_PCC) + (1 / arousal_CCC))
        final_metrics.update({"EMO": EMO})
    for name in final_metrics:
        final_metrics[name] = np.array(final_metrics[name])
        final_metrics[name] = np.average(final_metrics[name][final_metrics[name] > 0.0])
    return final_metrics