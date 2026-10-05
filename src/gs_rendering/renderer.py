# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import torch
from torch import nn
from src.gs_rendering.instantiater import *
from gsplat.rendering import rasterization

class GaussianRenderer(torch.nn.Module):
    def __init__(self, instantiation_mode):
        super().__init__()
        self.instantiater = { "ray": GaussianRayImportanceInstantiater, "plane": GaussianPlaneInstantiater }[instantiation_mode](low_pass_opacity=1./255.)
    
    def forward(self, planes, motion_planes, decoder, motion_decoder, ray_origins, ray_directions, cam2world_matrix, intrinsics, rendering_options, source_motion_latents=None, target_motion_latents=None, override_gaussian_dict=None):
        gaussian_dict = self.instantiater(planes, motion_planes, decoder, ray_origins, ray_directions, rendering_options) if override_gaussian_dict is None else override_gaussian_dict
        
        xyzs = gaussian_dict['xyz']
        colors = gaussian_dict['color']
        densities = gaussian_dict['density']
        scalings = gaussian_dict['scaling']
        rotations = gaussian_dict['rotation']
        motions = gaussian_dict['motion']
        features = gaussian_dict['feature']

        batch_size = len(xyzs)
        if motion_decoder is not None and source_motion_latents is not None and target_motion_latents is not None:
            for idx in range(batch_size):
                source_to_target = motion_decoder(motions[idx][None], source_motion_latents[idx][None], target_motion_latents[idx][None])
                features[idx] = features[idx] + source_to_target[0]
        
        out = self.instantiater.instantiate_from_features(decoder, features)
        xyzs = out['xyz']
        colors = out['color']
        densities = out['density']
        scalings = out['scaling']
        rotations = out['rotation']

        gaussian_dict['xyz'] = xyzs
        gaussian_dict['color'] = colors
        gaussian_dict['density'] = densities
        gaussian_dict['scaling'] = scalings
        gaussian_dict['rotation'] = rotations
        
        rgb_final = []
        depth_final = []
        alpha_final = []
        for idx in range(batch_size):
            viewmatrix = torch.linalg.inv(cam2world_matrix[idx])
            H = W = rendering_options['3dgs_resolution']
            intrinsic = intrinsics[idx][None] * H
            intrinsic[:, -1, -1] = 1

            rgb, alpha, _ = rasterization(
                means=xyzs[idx], 
                quats=rotations[idx], 
                scales=scalings[idx], 
                opacities=1 - torch.exp(-densities[idx]).squeeze(1), 
                colors=colors[idx], 
                viewmats=viewmatrix[None], 
                Ks=intrinsic, 
                width=W, 
                height=H, 
                packed=False, 
                absgrad=False, 
                sparse_grad=False, 
                rasterize_mode=rendering_options["rasterize_mode"], 
                render_mode="RGB+ED", 
                sh_degree=None, 
            )
            
            rgb = rgb[0].permute(2, 0, 1)
            alpha = alpha[0].permute(2, 0, 1)

            rgb_final.append(rgb[:-1])
            depth_final.append(rgb[-1:])
            alpha_final.append(alpha)
            
        rgb_final = torch.stack(rgb_final) * 2 - 1
        depth_final = torch.stack(depth_final)
        
        return rgb_final.reshape(batch_size, rgb_final.shape[1], -1).permute(0, 2, 1), depth_final.reshape(batch_size, -1, 1), gaussian_dict
