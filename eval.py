# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
from metrics_cal import batch_calc_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--source_image_path", type=str, required=True)
parser.add_argument("--driving_image_folder", type=str, required=True)
parser.add_argument("--pred_image_folder", type=str, required=True)
parser.add_argument("--is_self", action="store_true", help="By default, assume evaluating the cross-reenactment. Flag this for evaluating the self-reenactment.")

if __name__ == "__main__":
    args = parser.parse_args()
    ret = batch_calc_metrics(
        args.source_image_path, 
        [os.path.join(args.driving_image_folder, fn) for fn in sorted(os.listdir(args.driving_image_folder))], 
        [os.path.join(args.pred_image_folder, fn) for fn in sorted(os.listdir(args.pred_image_folder))], 
        disable_metrics={True: ["emo", "smirk"], False: ['iou', 'psnr', 'lpips', 'ssim']}[args.is_self], 
        apply_matting={True: "face", False: "matting"}[args.is_self]
    )
    
    for name in ret:
        print(f"{name}: {ret[name]:.4f}")