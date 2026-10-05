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

from . import math_utils
from .ray_sampler import RaySampler

from torch_utils import misc

def generate_planes_fixed():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32) 

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    assert W > 1
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

class GaussianRayImportanceInstantiater(nn.Module):
    '''This instantiater promises to generate same number of Gaussian kernels for a batch of inputs. However, ofc, filtering could be applied to make the number vary.'''
    def __init__(self, low_pass_opacity: float = 1 / 255) -> None:
        super().__init__()
        self.plane_axes_fixed = generate_planes_fixed()
        self.ray_sampler = RaySampler()
        self.low_pass_opacity = low_pass_opacity
    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples
    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse
    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals
    def sample_planes_fixed(self, planes, sample_coordinates, options):
        return sample_from_planes(self.plane_axes_fixed, planes, sample_coordinates, padding_mode='zeros', box_warp=options['box_warp'])
    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        sampled_features = self.sample_planes_fixed(planes, sample_coordinates, options)
        out = decoder(sampled_features, sample_directions)
        return out
    def instantiate_from_features(self, decoder, features: torch.Tensor):
        xyz_s       = []
        color_s     = []
        density_s   = []
        scaling_s   = []
        rotation_s  = []
        
        for _, feat in enumerate(features):
            out = decoder(feat[None])

            deltas = out['delta'].squeeze(0)
            colors = out['rgb'].squeeze(0)
            densities = out['sigma'].squeeze(0)
            scalings = out['scaling'].squeeze(0)
            rotations = out['rotation'].squeeze(0)

            xyz_s.append(deltas)
            color_s.append(colors)
            density_s.append(densities)
            scaling_s.append(scalings)
            rotation_s.append(rotations)

        return {
            'xyz':          xyz_s, 
            'color':        color_s, # in [0, 1]
            'density':      density_s, 
            'scaling':      scaling_s, 
            'rotation':     rotation_s
        }
    def forward(self, planes, motion_planes, decoder, cam2world_matrixORray_origins, intrinsicsORray_directions, rendering_options):
        misc.assert_shape(planes, [None, 3, None, None, None])
        if self.plane_axes_fixed.device != planes.device:
            self.plane_axes_fixed = self.plane_axes_fixed.to(planes.device)

        if len(cam2world_matrixORray_origins.shape) == 2 and len(intrinsicsORray_directions.shape) == 2 and cam2world_matrixORray_origins.shape[1] == 16 and intrinsicsORray_directions.shape[1] == 9:
            # Create a batch of rays for volume rendering
            ray_origins, ray_directions = self.ray_sampler(cam2world_matrixORray_origins.reshape(-1, 4, 4), intrinsicsORray_directions.reshape(-1, 3, 3), rendering_options['neural_rendering_resolution'])
        elif len(cam2world_matrixORray_origins.shape) == 3 and len(intrinsicsORray_directions.shape) == 3 and cam2world_matrixORray_origins.shape[-1] == 3 and intrinsicsORray_directions.shape[-1] == 3 and cam2world_matrixORray_origins.shape[-2] == intrinsicsORray_directions.shape[-2]:
            ray_origins = cam2world_matrixORray_origins
            ray_directions = intrinsicsORray_directions
        else:
            raise ValueError(f'{cam2world_matrixORray_origins.shape}, {intrinsicsORray_directions.shape}')

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        
        reduction_fn = lambda x: torch.mean(x, dim=1)
        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3).detach()
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)

        xyzs_coarse = sample_coordinates
        deltas_coarse = out['delta']
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        scalings_coarse = out['scaling']
        rotations_coarse = out['rotation']
        features_coarse = out['feature']
        motions_coarse = reduction_fn(self.sample_planes_fixed(motion_planes, sample_coordinates, rendering_options))

        xyzs_coarse = sample_coordinates.reshape(batch_size, num_rays, samples_per_ray, xyzs_coarse.shape[-1])
        deltas_coarse = deltas_coarse.reshape(batch_size, num_rays, samples_per_ray, deltas_coarse.shape[-1])
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        scalings_coarse = scalings_coarse.reshape(batch_size, num_rays, samples_per_ray, scalings_coarse.shape[-1])
        rotations_coarse = rotations_coarse.reshape(batch_size, num_rays, samples_per_ray, rotations_coarse.shape[-1])
        features_coarse = features_coarse.reshape(batch_size, num_rays, samples_per_ray, features_coarse.shape[-1])
        motions_coarse = motions_coarse.reshape(batch_size, num_rays, samples_per_ray, motions_coarse.shape[-1])

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            # Notice that in the coarse pass, sampled coordinates are sorted.
            # Therefore, we could apply alpha-blending to estimate the weights.
            # Approximate the importance sampling here, since the center of 
            # gaussians do not necessarily lie on the ray.
            alpha = 1 - torch.exp(-densities_coarse)
            alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
            weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]
            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3).detach()

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)

            xyzs_fine = sample_coordinates
            deltas_fine = out['delta']
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            scalings_fine = out['scaling']
            rotations_fine = out['rotation']
            features_fine = out['feature']
            motions_fine = reduction_fn(self.sample_planes_fixed(motion_planes, sample_coordinates, rendering_options))

            xyzs_fine = xyzs_fine.reshape(batch_size, num_rays, N_importance, xyzs_fine.shape[-1])
            deltas_fine = deltas_fine.reshape(batch_size, num_rays, N_importance, deltas_fine.shape[-1])
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            scalings_fine = scalings_fine.reshape(batch_size, num_rays, N_importance, scalings_fine.shape[-1])
            rotations_fine = rotations_fine.reshape(batch_size, num_rays, N_importance, rotations_fine.shape[-1])
            features_fine = features_fine.reshape(batch_size, num_rays, N_importance, features_fine.shape[-1])
            motions_fine = motions_fine.reshape(batch_size, num_rays, N_importance, motions_fine.shape[-1])

            all_xyzs = torch.cat([xyzs_coarse, xyzs_fine], dim=-2)
            all_deltas = torch.cat([deltas_coarse, deltas_fine], dim=-2)
            all_colors = torch.cat([colors_coarse, colors_fine], dim=-2)
            all_densities = torch.cat([densities_coarse, densities_fine], dim=-2)
            all_scalings = torch.cat([scalings_coarse, scalings_fine], dim=-2)
            all_rotations = torch.cat([rotations_coarse, rotations_fine], dim=-2)
            all_features = torch.cat([features_coarse, features_fine], dim=-2)
            all_motions = torch.cat([motions_coarse, motions_fine], dim=-2)
        else:
            all_xyzs = xyzs_coarse
            all_deltas = deltas_coarse
            all_colors = colors_coarse
            all_densities = densities_coarse
            all_scalings = scalings_coarse
            all_rotations = rotations_coarse
            all_motions = motions_coarse
            all_features = features_coarse
        
        xyzs = all_xyzs.reshape(batch_size, -1, 3) + all_deltas.reshape(batch_size, -1, 3)
        colors = all_colors.reshape(batch_size, -1, all_colors.shape[-1])
        densities = all_densities.reshape(batch_size, -1, 1)
        scalings = all_scalings.reshape(batch_size, -1, 3)
        rotations = all_rotations.reshape(batch_size, -1, 4)
        features = all_features.reshape(batch_size, -1, all_features.shape[-1])
        motions = all_motions.reshape(batch_size, -1, all_motions.shape[-1])

        # Implement filtering out here
        xyz_s       = []
        color_s     = []
        density_s   = []
        scaling_s   = []
        rotation_s  = []
        feature_s = []
        motion_s = []
        for xyz, color, density, scaling, rotation, feature, motion in zip(xyzs.unbind(dim=0), colors.unbind(dim=0), densities.unbind(dim=0), scalings.unbind(dim=0), rotations.unbind(dim=0), features.unbind(dim=0), motions.unbind(dim=0)):
            mask = (density >= self.low_pass_opacity).squeeze()
            xyz_s.append(xyz[mask])
            color_s.append(color[mask])
            density_s.append(density[mask])
            scaling_s.append(scaling[mask])
            rotation_s.append(rotation[mask])
            feature_s.append(feature[mask])
            motion_s.append(motion[mask])
        
        return {
            'xyz':          xyz_s, 
            'color':        color_s, # in [0, 1]
            'density':      density_s, 
            'scaling':      scaling_s, 
            'rotation':     rotation_s, 
            'motion':       motion_s, 
            'feature':      feature_s
        }

class GaussianPlaneInstantiater(nn.Module):
    def __init__(self, low_pass_opacity: float = 1 / 255) -> None:
        super().__init__()
        self.low_pass_opacity = low_pass_opacity

    def instantiate_from_features(self, decoder, features: torch.Tensor):
        xyz_s       = []
        color_s     = []
        density_s   = []
        scaling_s   = []
        rotation_s  = []
        
        for i, feat in enumerate(features):
            out = decoder(feat[None])

            deltas = out['delta'].squeeze(0)
            colors = out['rgb'].squeeze(0)
            densities = out['sigma'].squeeze(0)
            scalings = out['scaling'].squeeze(0)
            rotations = out['rotation'].squeeze(0)

            xyz_s.append(deltas)
            color_s.append(colors)
            density_s.append(densities)
            scaling_s.append(scalings)
            rotation_s.append(rotations)

        return {
            'xyz':          xyz_s, 
            'color':        color_s, # in [0, 1]
            'density':      density_s, 
            'scaling':      scaling_s, 
            'rotation':     rotation_s
        }
    def forward(self, planes, motion_planes, decoder, *args, **kwargs):
        B, _, C, _, _ = planes.shape

        planes = planes.mean(1)
        motion_planes = motion_planes.mean(1)

        planes = torch.nn.functional.interpolate(planes, (512, 512), mode='bilinear', align_corners=False, antialias=True)
        motion_planes = torch.nn.functional.interpolate(motion_planes, (512, 512), mode='bilinear', align_corners=False, antialias=True)
        
        planes = planes.permute(0, 2, 3, 1).reshape(B, -1, C)
        out = decoder(planes)
        deltas_coarse = out['delta']
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        scalings_coarse = out['scaling']
        rotations_coarse = out['rotation']
        features_coarse = out['feature']
        motions_coarse = motion_planes.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        all_deltas = deltas_coarse
        all_colors = colors_coarse
        all_densities = densities_coarse
        all_scalings = scalings_coarse
        all_rotations = rotations_coarse
        all_motions = motions_coarse
        all_features = features_coarse
        
        xyzs = all_deltas.reshape(B, -1, 3)
        colors = all_colors.reshape(B, -1, all_colors.shape[-1])
        densities = all_densities.reshape(B, -1, 1)
        scalings = all_scalings.reshape(B, -1, 3)
        rotations = all_rotations.reshape(B, -1, 4)
        features = all_features.reshape(B, -1, all_features.shape[-1])
        motions = all_motions.reshape(B, -1, all_motions.shape[-1])

        # Implement filtering out here
        xyz_s       = []
        color_s     = []
        density_s   = []
        scaling_s   = []
        rotation_s  = []
        feature_s = []
        motion_s = []
        for xyz, color, density, scaling, rotation, feature, motion in zip(xyzs.unbind(dim=0), colors.unbind(dim=0), densities.unbind(dim=0), scalings.unbind(dim=0), rotations.unbind(dim=0), features.unbind(dim=0), motions.unbind(dim=0)):
            mask = (density >= self.low_pass_opacity).squeeze()
            xyz = xyz[mask]
            color = color[mask]
            density = density[mask]
            scaling = scaling[mask]
            rotation = rotation[mask]
            features = feature[mask]
            motions = motion[mask]

            xyz_s.append(xyz)
            color_s.append(color)
            density_s.append(density)
            scaling_s.append(scaling)
            rotation_s.append(rotation)
            feature_s.append(features)
            motion_s.append(motions)
        
        return {
            'xyz':          xyz_s, 
            'color':        color_s, # in [0, 1]
            'density':      density_s, 
            'scaling':      scaling_s, 
            'rotation':     rotation_s, 
            'motion':       motion_s, 
            'feature':      feature_s
        }