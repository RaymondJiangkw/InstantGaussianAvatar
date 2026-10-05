# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
from torch_utils.ops import bias_act

import torch
from torch import nn
from torch.nn import functional as F

from src.backbone import *
from src.gs_rendering.renderer import *
from src.gs_rendering.ray_sampler import *

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class AdaLN(nn.Module):
    def __init__(self, c_dim, out_dim, lr_multiplier=0.5):
        super().__init__()
        
        self.c_dim = c_dim
        self.out_dim = out_dim
        self.lr_multiplier = lr_multiplier
        
        self.mapping = MyLinear(c_dim, out_dim * 2, bias=False, lr_multiplier=lr_multiplier)
        self.proj = MyLinear(out_dim, out_dim, lr_multiplier=lr_multiplier)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MyLinear):
            trunc_normal_(m.weight, std=.02 / self.lr_multiplier)
            if isinstance(m, MyLinear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, c: torch.Tensor, features: torch.Tensor):
        c = self.mapping(c)
        scale = c[:, :c.shape[-1] // 2][:, None, :]
        shift = c[:, c.shape[-1] // 2:][:, None, :]
        x = nn.LayerNorm(self.out_dim, elementwise_affine=False)(features) * scale + shift
        x = self.proj(x)
        return x

class GaussianDecoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = n_features * 2
        self.offset = 8 # Legacy, not used eventually
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + 3 + 3 + 3 + 4 + self.offset)
        )
        with torch.no_grad():
            self.net[-1].bias.zero_()
        
    def forward(self, sampled_features, *args, **kwargs):
        if len(sampled_features.shape) > 3:
            # Need to reduce
            sampled_features = sampled_features.mean(1)
        x = sampled_features
        N, M, C = x.shape
        x = x.reshape(N*M, C)
        x = self.net(x)
        x = x.reshape(N, M, -1)

        raw_sigma, raw_rgb, _, raw_delta, raw_scaling, raw_rotation, _ = torch.split(x, (1, 3, 7, 3, 3, 4, 1), dim=-1)

        sigma = torch.nn.functional.softplus(raw_sigma - 1)
        rgb = torch.sigmoid(raw_rgb)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        delta = torch.tanh(raw_delta)
        scaling = torch.sigmoid(raw_scaling)*0.05
        rotation = torch.nn.functional.normalize(raw_rotation, dim=-1)
        return {'rgb': rgb, 'sigma': sigma, 'delta': delta, 'scaling': scaling, 'rotation': rotation, 'feature': sampled_features}

class MotionDecoder(torch.nn.Module):
    def __init__(self, feat_dim, motion_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.motion_dim = motion_dim
        self.transform = AdaLN(motion_dim * 2, feat_dim, lr_multiplier=0.5)
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(self.feat_dim, self.feat_dim * 2),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.feat_dim * 2, self.feat_dim)
        )
    
    def forward(self, sampled_features, motion_latents_u, motion_latents_v):
        x = self.transform(torch.cat((motion_latents_u, motion_latents_v), dim=-1), sampled_features)

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)

        return 0.01*x

class InstantGaussianAvatar(torch.nn.Module):
    def __init__(self, 
        feat_dim: int        = 96, 
        motion_dim: int      = 512, 
        img_resolution: int  = 512, 
        legacy: bool         = False
    ) -> None:
        super().__init__()

        self.feat_dim = feat_dim
        self.motion_dim = motion_dim
        self.img_resolution = img_resolution
        self.rendering_kwargs = {
            "box_warp": 1., 
            "ray_start": 2.25, 
            "ray_end": 3.5, 
            "3dgs_resolution": 512, 
            "neural_rendering_resolution": 64, 
            "depth_resolution": 24, 
            "depth_resolution_importance": 24, 
            "disparity_space_sampling": False, 
            "rasterize_mode": "antialiased" if legacy else "classic", 
            "instantiation_mode": "ray" if legacy else "plane"
        }
        
        self.backbone = LP3DBackbone(img_resolution=img_resolution, feat_dim=self.feat_dim)
        
        self.ray_sampler = RaySampler()
        self.renderer = GaussianRenderer(self.rendering_kwargs["instantiation_mode"])
        self.decoder = GaussianDecoder(self.feat_dim // 2)
        self.motion_decoder = MotionDecoder(self.feat_dim // 2, self.motion_dim)
    
    def synthesis(self, image: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function('E_forward_synthesis'):
            planes = self.backbone(image)
        geoapp_planes = planes[:, :, :self.feat_dim // 2, :, :]
        motion_planes = planes[:, :, self.feat_dim // 2:, :, :]
        return geoapp_planes, motion_planes
    
    def render(self, geoapp_planes: torch.Tensor, motion_planes: torch.Tensor, cf: torch.Tensor, ct: torch.Tensor = None, source_motion_latents: torch.Tensor = None, target_motion_latents: torch.Tensor = None) -> torch.Tensor:
        if ct is None: ct = cf

        inst_cam2world_matrix = cf[:, :16].view(-1, 4, 4)
        inst_intrinsics = cf[:, 16:25].view(-1, 3, 3)

        view_cam2world_matrix = ct[:, :16].view(-1, 4, 4)
        view_intrinsics = ct[:, 16:25].view(-1, 3, 3)

        with torch.autograd.profiler.record_function('E_forward_render'):
            N = len(cf)

            feature_samples, depth_samples, gaussian_dict = self.renderer(geoapp_planes, motion_planes, self.decoder, self.motion_decoder, *self.ray_sampler(inst_cam2world_matrix, inst_intrinsics, self.rendering_kwargs['neural_rendering_resolution']), view_cam2world_matrix, view_intrinsics, self.rendering_kwargs, source_motion_latents=source_motion_latents, target_motion_latents=target_motion_latents) # channels last

            H = W = int(feature_samples.shape[1] ** 0.5)
            rgb_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, depth_samples.shape[-1], H, W) if depth_samples is not None else None

        return {'image': rgb_image, 'image_depth': depth_image, 'gaussian_dict': gaussian_dict}
    def forward(self, image: torch.Tensor, cf: torch.Tensor = None, ct: torch.Tensor = None, **kwargs) -> torch.Tensor:
        '''
        Args:
            cf: The camera pose where we instantiate the Gaussians from.
            ct: The camera pose where we render the Gaussians from.
        '''
        if ct is None: ct = cf
        geoapp_planes, motion_planes = self.synthesis(image)
        return self.render(geoapp_planes, motion_planes, cf, ct, **kwargs)