# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""

import math

import torch
import torch.nn as nn

from src.gs_rendering import math_utils

class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, up_vector=[0, 1, 0], origin_vector=[0, 0, 0], device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)
        camera_origins = camera_origins + torch.tensor(origin_vector, device=device)[None]

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins, up_vector)
    
    @staticmethod
    def place(v, h, lookat_position, radius=1, batch_size=1, order=[0, 2, 1], device='cpu'):
        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, order[0]] = radius*math.sin(v) * math.cos(h)
        camera_origins[:, order[1]] = radius*math.sin(v) * math.sin(h)
        camera_origins[:, order[2]] = radius*math.cos(v)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins, up_vector=[0, 0, 1])

class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 - 1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)    

def create_cam2world_matrix(forward_vector, origin, up_vector=[0, 1, 0]):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = torch.tensor(up_vector, dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

import numpy as np
from typing import List
from torch.nn import functional as F

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def get_camera_positions(
    yaw: torch.Tensor, 
    pitch: torch.Tensor, 
    radius: torch.Tensor, 
) -> torch.Tensor:
    """
    Get Camera Positions by transforming given vertical position `v`
    and horizontal position `h`.
    Args:
        yaw: Tensor of the shape (..., 1) denoting the
            Horizontal positions.
        pitch: Tensor of the shape (..., 1) denoting the 
            Vertical positions.
        radius: Tensor of the shape (..., 1) denoting the
            radius.
    Returns:
        points: tensor with shape (..., 3) denoting the position of 
            the point on the sphere with radius 1 given `phi` and `theta`.
    """

    theta = pitch
    yaw = yaw / torch.pi
    phi = torch.arccos(1 - 2 * yaw)

    x = torch.sin(phi) * torch.cos(np.pi-theta) * radius
    y = torch.cos(phi) * radius
    z = torch.sin(phi) * torch.sin(np.pi-theta) * radius

    points = torch.cat((x, y, z), dim=-1)
    return points

def create_camera2world_matrix(
    direction: torch.Tensor, 
    roll: torch.Tensor, 
) -> torch.Tensor:
    """
    Compute a transformation matrix mapping from Camera Space to World Space.
    Args:
        origin: Tensor with shape (..., 3) denoting the origin of the camera in World
            Space
        direction: Tensor with shape (..., 3) denoting the normalized direction vector
            of the camera
        roll: The tensor of shape (batch_size, 1) denoting the roll angle in radian.
    Returns:
        rotation: Tensor with shape (..., 3, 3) denoting the rotation matrix.
            To transform `vectors` (..., 3) from Camera Space to World Space, you only
            need to do torch.bmm(`rotation`, `vectors`)
    """
    forward = F.normalize(direction, dim=-1) # (..., 3)
    # print(forward, forward_ref)
    up_ref = torch.tensor([0., 1., 0.], dtype=torch.float, device=forward.device).view(*[1 for _ in range(len(forward.shape) - 1)], -1).expand_as(forward) # (..., 3)
    roll_mat = axis_angle_to_matrix(forward * roll) # (..., 3, 3)
    axis = F.normalize(torch.cross(up_ref, forward, dim=-1), dim=-1)
    up = -F.normalize(torch.cross(axis, forward, dim=-1), dim=-1).unsqueeze(-1) # (..., 3, 1)
    up = torch.matmul(roll_mat, up).squeeze(-1) # (..., 3)
    left = F.normalize(torch.cross(up, forward, dim=-1), dim=-1)
    up = -F.normalize(torch.cross(forward, left, dim=-1), dim=-1) # Yes, necessary. The result is changed from World Up direction to Camera Up direction.

    rotation = torch.stack((-left, up, forward), axis=-1) # (..., 3, 3)

    return rotation

def transform_pos_to_cond(
    pos: torch.Tensor, 
) -> torch.Tensor:
    pitch, yaw, roll, radius, pivot_x, pivot_y, pivot_z = torch.split(pos, 1, -1)
    pivot = torch.cat((pivot_x, pivot_y, pivot_z), dim=-1)
    # Sample Camera Positions
    # cameras_origin: (batch, 3)
    cameras_origin = get_camera_positions(yaw, pitch, radius)
    cameras_forward = F.normalize(pivot-cameras_origin, dim=-1)

    # Transform `origins` and `directions`
    # translation: (batch, 3)
    translation = cameras_origin
    # rotation: (batch, 3, 3)
    rotation = create_camera2world_matrix(cameras_forward, roll)
    # matrix: (batch, 4, 4)
    matrix = torch.eye(4, device=translation.device, dtype=translation.dtype).unsqueeze(0).repeat(translation.size(0), 1, 1)
    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = translation
    return matrix.view(-1, 16)

def sample_intrinsics(batch_size: int, fov_degrees_mean: float, pp_mean: float, fov_degrees_std: float = 0, pp_std: float = 0, device='cpu'):
    fov_degrees = torch.randn(batch_size, device=device) * fov_degrees_std + fov_degrees_mean
    focal_lengths = 1 / (torch.tan(fov_degrees * torch.pi / 360) * 1.414)
    zeros = torch.zeros_like(fov_degrees)
    ones = torch.ones_like(fov_degrees)

    pp_x = torch.randn(batch_size, device=device) * pp_std + pp_mean
    pp_y = torch.randn(batch_size, device=device) * pp_std + pp_mean

    intrinsics = torch.stack([
        focal_lengths, zeros, pp_x, 
        zeros, focal_lengths, pp_y, 
        zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3)
    return intrinsics

def sample_extrinsics(batch_size: int, pitch_mean: float, yaw_mean: float, roll_mean: float, radius_mean: float, pitch_std: float = 0, yaw_std: float = 0, roll_std: float = 0, radius_std: float = 0, pivot: List[float] = [0., 0., 0.], device='cpu'):
    pitch = (torch.rand(batch_size, device=device) * 2 - 1) * pitch_std + pitch_mean
    yaw = (torch.rand(batch_size, device=device) * 2 - 1) * yaw_std + yaw_mean
    roll = (torch.rand(batch_size, device=device) * 2 - 1) * roll_std + roll_mean
    radius = (torch.rand(batch_size, device=device) * 2 - 1) * radius_std + radius_mean
    pivot = torch.tensor(pivot)[None].to(device).expand(batch_size, -1)
    pose = torch.cat((torch.stack((pitch, yaw, roll, radius), dim=-1), pivot), dim=-1)
    return transform_pos_to_cond(pose)

def sample_wide_c(N : int, device):
    c_ref_intrinsics = sample_intrinsics(N, 23.28, 0.5, 0, 0 / 512, device=device)
    c_ref_extrinsics = sample_extrinsics(N, torch.pi / 2, torch.pi / 2, 0., 2.7, 
                                            70 / 180 * torch.pi, 49 / 180 * torch.pi, 2 / 180 * torch.pi, 0., [0., 0., 0.], device=device)
    return torch.cat((c_ref_extrinsics.reshape(-1, 16), c_ref_intrinsics.reshape(-1, 9)), dim=-1)

def sample_c(N : int, device):
    c_ref_intrinsics = sample_intrinsics(N, 18.83, 0.5, 0, 0 / 512, device=device)
    c_ref_extrinsics = sample_extrinsics(N, torch.pi / 2, torch.pi / 2, 0., 2.7, 
                                            70 / 180 * torch.pi, 49 / 180 * torch.pi, 2 / 180 * torch.pi, 0., [0., 0., 0.], device=device)
    return torch.cat((c_ref_extrinsics.reshape(-1, 16), c_ref_intrinsics.reshape(-1, 9)), dim=-1)