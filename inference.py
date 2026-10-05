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
import argparse
from PIL import Image

from time import time
from tqdm import tqdm
from src.utils import *
from src.model import *
from src.camera_utils import *
from src.third_party.motion_encoder import *

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--source_image", type=str, default='./demo/00015.png')
parser.add_argument("--driving_images_folder", type=str, default='./demo/motion_sequences')
parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints")
parser.add_argument("--output_folder", type=str, default="outputs")
parser.add_argument("--measure_speed", action="store_true")
parser.add_argument("--assume_fixed_crop_box_for_driving_images", action="store_true", help="Use the same cropping window for driving images. Useful while doing the streaming.")
parser.add_argument("--legacy", action="store_true", help="whether use the legacy model.")

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    print("Load checkpoints ...")
    state_dict = torch.load(os.path.join(args.checkpoint_folder, "ckpt.pth" if not args.legacy else "ckpt-legacy.pth"), map_location=device)

    M = MotEncoder(device)
    E = InstantGaussianAvatar(legacy=args.legacy).to(device).eval().requires_grad_(False)

    M.load_state_dict(state_dict["M_state_dict"])
    E.load_state_dict(state_dict["E_state_dict"])

    print("Load images and warm up ...")
    intrinsics = FOV_to_intrinsics(18.837 if args.legacy else 23.28, device=device)
    cam_pivot = torch.tensor([0., 0.0, 0.], device=device)
    cam_radius = 2.7
    cond_pose = torch.cat([LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    pose = torch.cat([LookAtPoseSampler.sample(12*np.pi/24, 12*np.pi/24, cam_pivot, radius=cam_radius, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    src_image, _ = (crop if args.legacy else wide_crop)(Image.open(args.source_image))
    tar_fns = sorted(os.listdir(args.driving_images_folder))
    tar_images = [ to_tensor(Image.open(os.path.join(args.driving_images_folder, fn))) for fn in tar_fns ]

    gaussian_dict = E(src_image, cond_pose)["gaussian_dict"]
    src_motion = M(src_image)

    if args.measure_speed:
        start = time()
        # Warm-up
        E(src_image, cond_pose)
        for _ in tqdm(range(100), desc="Measure encoding time"):
            E(src_image, cond_pose)
        torch.cuda.synchronize()
        encode_time_ms = ((time() - start) / 100) * 1000.
        print(f"Encoding speed: {encode_time_ms:.2f} ms per image.")
    
    # Warm-up
    tar_motion = M(tar_images[0])
    tar_bbox = M.detect_bbox(tar_images[0])
    animate_gaussians(E, gaussian_dict, cond_pose, src_motion, tar_motion)

    print("Render ...")

    start = time()
    image_s = []
    for tar_image in tqdm(tar_images, desc="Render frames ..."):
        # You can replace `pose` with whatever pose you want for viewpoint control by changing the pitch or yaw.
        # Or, you can also use the pose from the driving image. Caution: this is more accurate but slow and not what we use during the end-to-end streaming setting. We also don't have pose for wide cropped image (newer version).
        # >>> _, tar_pose = crop(render_tensor(tar_image))
        # 
        # For speed measurement under the end-to-end streaming setting, we use a GPU-accelerated pose prediction module for estimating the pose from the real-time captured driving image.
        image_s.append(animate_gaussians(E, gaussian_dict, pose, src_motion, M(tar_image) if not args.assume_fixed_crop_box_for_driving_images else M.pure_encode(M.crop_frame(tar_image, tar_bbox))))
    torch.cuda.synchronize()
    render_time_fps = len(tar_images) / (time() - start)
    print(f"Animate and render at {render_time_fps:.2f} FPS.")

    print("Saving ...")
    for image, fn in zip(image_s, tar_fns):
        render_tensor(image.clamp(-1., 1.)).save(os.path.join(args.output_folder, fn))
    
    print("Done.")