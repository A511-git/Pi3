import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from typing import *
import itertools
import json
import warnings

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# 3D Gaussian Splatting PLY Exporter (Binary Format)
# Compatible with SuperSplat, PlayCanvas, Luma AI, WebGL 3DGS Viewers
# ---------------------------------------------------------------------------

def save_gaussian_splat_ply(
    filepath: Union[str, Path],
    points: Union[np.ndarray, torch.Tensor],
    colors: Union[np.ndarray, torch.Tensor],
    scales: Union[np.ndarray, torch.Tensor],
    quats: Union[np.ndarray, torch.Tensor],
    opacities: Optional[Union[np.ndarray, torch.Tensor]] = None
):
    """
    Exports points as standard 3D Gaussian Splatting binary PLY format.
    Compatible with SuperSplat, PlayCanvas, Luma AI, and WebGL 3DGS Viewers.
    Supports both NumPy arrays and PyTorch CUDA tensors.
    """
    filepath = Path(filepath)
    if filepath.parent:
        filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    if isinstance(scales, torch.Tensor):
        scales = scales.detach().cpu().numpy()
    if isinstance(quats, torch.Tensor):
        quats = quats.detach().cpu().numpy()
    if opacities is not None and isinstance(opacities, torch.Tensor):
        opacities = opacities.detach().cpu().numpy()

    N = len(points)
    if N == 0:
        print(f"⚠️ No points to save for {filepath}")
        return

    if opacities is None:
        opacities = np.full((N, 1), 4.5, dtype=np.float32)  # High opacity logit (~0.989)
    elif opacities.ndim == 1:
        opacities = opacities[:, None]

    # Handle colors in [0, 1] vs [0, 255]
    colors_f = colors.astype(np.float32)
    if colors_f.max() <= 1.05 and colors_f.min() >= 0.0:
        colors_f = colors_f * 255.0

    # Spherical Harmonics DC (Degree 0) from RGB: f_dc = (rgb/255 - 0.5) / 0.28209479177387814
    sh_dc = ((colors_f / 255.0) - 0.5) / 0.28209479177387814
    normals = np.zeros((N, 3), dtype=np.float32)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    elements = np.empty(N, dtype=dtype)
    elements['x'] = points[:, 0].astype(np.float32)
    elements['y'] = points[:, 1].astype(np.float32)
    elements['z'] = points[:, 2].astype(np.float32)
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = sh_dc[:, 0].astype(np.float32)
    elements['f_dc_1'] = sh_dc[:, 1].astype(np.float32)
    elements['f_dc_2'] = sh_dc[:, 2].astype(np.float32)
    elements['opacity'] = opacities[:, 0].astype(np.float32)
    elements['scale_0'] = scales[:, 0].astype(np.float32)
    elements['scale_1'] = scales[:, 1].astype(np.float32)
    elements['scale_2'] = scales[:, 2].astype(np.float32)
    elements['rot_0'] = quats[:, 0].astype(np.float32)
    elements['rot_1'] = quats[:, 1].astype(np.float32)
    elements['rot_2'] = quats[:, 2].astype(np.float32)
    elements['rot_3'] = quats[:, 3].astype(np.float32)

    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(elements.tobytes())
    print(f"✅ Saved 3D Gaussian Splat ({N:,} splats) -> {filepath}")


# ---------------------------------------------------------------------------
# Multi-View Perspective Point Cloud to 3D Gaussian Splats (Pi3 / Pi3X)
# ---------------------------------------------------------------------------

def pi3_points_to_gaussians_torch(
    points: torch.Tensor,
    imgs: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    stride: int = 1,
    is_indoor: bool = True,
    global_scale: float = 1.2,
    disc_thickness: float = 0.2,
    min_depth: float = 0.1,
    max_depth: Optional[float] = None,
    conf_threshold: float = 0.1,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts Pi3 / Pi3X multi-view 3D point predictions and RGB images into 3D Gaussian Splats on GPU.
    
    Args:
        points: [N, H, W, 3] or [B, N, H, W, 3] Global 3D point cloud coordinates.
        imgs: [N, 3, H, W] or [N, H, W, 3] RGB images in [0, 1] or [0, 255].
        conf: Optional [N, H, W, 1] or [N, H, W] raw confidence logits or probabilities.
        masks: Optional [N, H, W] boolean validity mask.
        stride: Pixel subsampling stride (1=full resolution, 2=half resolution).
        is_indoor: True for indoor preset (default cutoff 15m), False for outdoor (default cutoff 80m).
        global_scale: Splat radius scale multiplier.
        disc_thickness: Disc thickness ratio relative to min(s1, s2).
        min_depth: Minimum distance threshold in meters.
        max_depth: Maximum distance cutoff in meters (defaults: 15.0m indoor, 80.0m outdoor).
        conf_threshold: Minimum sigmoid(confidence) threshold.
        device: Computation device.

    Returns:
        (flat_points, flat_rgb, flat_scales, flat_quats, flat_opacities)
    """
    if device is None:
        device = points.device
    elif isinstance(device, str):
        device = torch.device(device)

    if points.ndim == 5:
        points = points[0]  # [N, H, W, 3]
    points_t = points.to(device=device, dtype=torch.float32)
    N, H, W, _ = points_t.shape

    # Normalize image tensor to [N, H, W, 3]
    if imgs.ndim == 5:
        imgs = imgs[0]
    imgs_t = imgs.to(device=device)
    if imgs_t.shape[1] == 3 and imgs_t.ndim == 4:
        imgs_t = imgs_t.permute(0, 2, 3, 1)  # [N, H, W, 3]
    
    if imgs_t.max() <= 1.05:
        imgs_t = imgs_t * 255.0

    if max_depth is None:
        max_depth = 15.0 if is_indoor else 80.0

    # Subsampling
    if stride > 1:
        points_t = points_t[:, ::stride, ::stride]
        imgs_t = imgs_t[:, ::stride, ::stride]
        if conf is not None:
            if conf.ndim == 5:
                conf = conf[0]
            conf = conf[:, ::stride, ::stride]
        if masks is not None:
            if masks.ndim == 4 and masks.shape[0] == 1:
                masks = masks[0]
            masks = masks[:, ::stride, ::stride]
        N, H, W, _ = points_t.shape

    # Compute tangent vectors across grid
    # Horizontal tangent: t1 along u
    pad_u = torch.cat([points_t, points_t[:, :, -1:]], dim=2)
    t1 = pad_u[:, :, 1:] - pad_u[:, :, :-1]  # [N, H, W, 3]

    # Vertical tangent: t2 along v
    pad_v = torch.cat([points_t, points_t[:, -1:, :]], dim=1)
    t2 = pad_v[:, 1:, :] - pad_v[:, :-1, :]  # [N, H, W, 3]

    # Normal vector: n = t1 x t2
    normals = torch.cross(t1, t2, dim=-1)
    n_len = torch.norm(normals, dim=-1, keepdim=True)
    n_valid = n_len > 1e-6
    n_safe = torch.where(n_valid, n_len, torch.ones_like(n_len))
    n_unit = normals / n_safe

    # In-plane tangential scales
    s1 = torch.norm(t1, dim=-1, keepdim=True) * global_scale
    s2 = torch.norm(t2, dim=-1, keepdim=True) * global_scale
    s3 = disc_thickness * torch.minimum(s1, s2)
    scales = torch.cat([s1, s2, s3], dim=-1)
    log_scales = torch.log(torch.clamp(scales, min=1e-5, max=1e2))

    # Orthonormal frame: [t1_unit, t2_unit, n_unit]
    t1_len = torch.norm(t1, dim=-1, keepdim=True)
    t1_safe = torch.where(t1_len > 1e-6, t1_len, torch.ones_like(t1_len))
    t1_unit = t1 / t1_safe
    t2_unit = torch.cross(n_unit, t1_unit, dim=-1)

    # Rotation matrix to quaternion conversion
    R00, R01, R02 = t1_unit[..., 0], t2_unit[..., 0], n_unit[..., 0]
    R10, R11, R12 = t1_unit[..., 1], t2_unit[..., 1], n_unit[..., 1]
    R20, R21, R22 = t1_unit[..., 2], t2_unit[..., 2], n_unit[..., 2]

    tr = R00 + R11 + R22
    qw = torch.sqrt(torch.clamp(1.0 + tr, min=0.0)) / 2.0
    denom = 4.0 * torch.clamp(qw, min=1e-6)
    qx = (R21 - R12) / denom
    qy = (R02 - R20) / denom
    qz = (R10 - R01) / denom

    quats = torch.stack([qw, qx, qy, qz], dim=-1)
    q_norm = torch.clamp(torch.norm(quats, dim=-1, keepdim=True), min=1e-6)
    quats = quats / q_norm

    # Validity masking
    pt_dist = torch.norm(points_t, dim=-1)
    valid = torch.isfinite(points_t).all(dim=-1) & (pt_dist >= min_depth) & (pt_dist <= max_depth) & n_valid.squeeze(-1)

    if conf is not None:
        conf_t = conf.to(device=device, dtype=torch.float32)
        if conf_t.shape[-1] == 1:
            conf_t = conf_t.squeeze(-1)
        # If conf is raw logit, apply sigmoid
        if conf_t.min() < 0.0 or conf_t.max() > 1.0:
            conf_prob = torch.sigmoid(conf_t)
        else:
            conf_prob = conf_t
        valid = valid & (conf_prob > conf_threshold)
        
        # Continuous opacity logit
        opacities = torch.logit(torch.clamp(conf_prob * 0.99, min=1e-3, max=0.99)).unsqueeze(-1)
    else:
        opacities = torch.full((N, H, W, 1), 4.5, dtype=torch.float32, device=device)

    if masks is not None:
        masks_t = masks.to(device=device, dtype=torch.bool)
        valid = valid & masks_t

    flat_pts = points_t[valid]
    flat_rgb = imgs_t[valid]
    flat_scales = log_scales[valid]
    flat_quats = quats[valid]
    flat_opacities = opacities[valid]

    return flat_pts, flat_rgb, flat_scales, flat_quats, flat_opacities


def pi3_points_to_gaussians(
    points: Union[np.ndarray, torch.Tensor],
    imgs: Union[np.ndarray, torch.Tensor],
    conf: Optional[Union[np.ndarray, torch.Tensor]] = None,
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    stride: int = 1,
    is_indoor: bool = True,
    global_scale: float = 1.2,
    disc_thickness: float = 0.2,
    min_depth: float = 0.1,
    max_depth: Optional[float] = None,
    conf_threshold: float = 0.1,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy wrapper for pi3_points_to_gaussians_torch."""
    pts_t = torch.as_tensor(points) if not isinstance(points, torch.Tensor) else points
    imgs_t = torch.as_tensor(imgs) if not isinstance(imgs, torch.Tensor) else imgs
    conf_t = torch.as_tensor(conf) if (conf is not None and not isinstance(conf, torch.Tensor)) else conf
    masks_t = torch.as_tensor(masks) if (masks is not None and not isinstance(masks, torch.Tensor)) else masks

    pts, rgb, scs, qts, op = pi3_points_to_gaussians_torch(
        points=pts_t,
        imgs=imgs_t,
        conf=conf_t,
        masks=masks_t,
        stride=stride,
        is_indoor=is_indoor,
        global_scale=global_scale,
        disc_thickness=disc_thickness,
        min_depth=min_depth,
        max_depth=max_depth,
        conf_threshold=conf_threshold,
        device=device
    )
    return (
        pts.detach().cpu().numpy(),
        rgb.detach().cpu().numpy(),
        scs.detach().cpu().numpy(),
        qts.detach().cpu().numpy(),
        op.detach().cpu().numpy()
    )


# ---------------------------------------------------------------------------
# Equirectangular / Panorama Support
# ---------------------------------------------------------------------------

def spherical_uv_to_directions_torch(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Generates (H, W, 3) spherical ray direction vectors directly as a PyTorch CUDA tensor."""
    u = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    v = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
    theta = (1.0 - u_grid) * (2.0 * np.pi)
    phi = v_grid * np.pi
    sin_phi = torch.sin(phi)
    dirs = torch.stack([sin_phi * torch.cos(theta), sin_phi * torch.sin(theta), torch.cos(phi)], dim=-1)
    return dirs


def depth_to_spherical_gaussians_torch(
    depth: Union[torch.Tensor, np.ndarray],
    rgb: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    stride: int = 1,
    is_indoor: bool = True,
    global_scale: float = 1.2,
    disc_thickness: float = 0.2,
    min_depth: float = 0.1,
    max_depth: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """100% GPU-Accelerated conversion of spherical depth and RGB panorama into 3D Gaussian Splats."""
    if device is None:
        if isinstance(depth, torch.Tensor):
            device = depth.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    if not isinstance(depth, torch.Tensor):
        depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    else:
        depth_t = depth.to(device=device, dtype=torch.float32)

    if not isinstance(rgb, torch.Tensor):
        rgb_t = torch.as_tensor(rgb, device=device)
    else:
        rgb_t = rgb.to(device=device)

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
        else:
            mask_t = mask.to(device=device, dtype=torch.bool)
    else:
        mask_t = None

    if max_depth is None:
        max_depth = 15.0 if is_indoor else 80.0

    H, W = depth_t.shape[:2]
    if rgb_t.shape[:2] != (H, W):
        orig_dtype = rgb_t.dtype
        rgb_f = rgb_t.permute(2, 0, 1).unsqueeze(0).float()
        rgb_resized = F.interpolate(rgb_f, size=(H, W), mode='area')
        rgb_t = rgb_resized.squeeze(0).permute(1, 2, 0).to(orig_dtype)

    if stride > 1:
        depth_t = depth_t[::stride, ::stride]
        rgb_t = rgb_t[::stride, ::stride]
        if mask_t is not None:
            mask_t = mask_t[::stride, ::stride]
        H, W = depth_t.shape[:2]

    u = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) / W
    v = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) / H
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')

    theta = (1.0 - u_grid) * (2.0 * torch.pi)
    phi = v_grid * torch.pi

    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    dx = sin_phi * cos_theta
    dy = sin_phi * sin_theta
    dz = cos_phi

    dirs = torch.stack([dx, dy, dz], dim=-1)
    pts = dirs * depth_t.unsqueeze(-1)

    t1 = torch.stack([-sin_theta, cos_theta, torch.zeros_like(theta)], dim=-1)
    t2 = torch.stack([cos_phi * cos_theta, cos_phi * sin_theta, -sin_phi], dim=-1)

    d_theta = (2.0 * torch.pi) / W
    d_phi = torch.pi / H

    s1 = depth_t * (d_theta * torch.clamp(sin_phi, min=1e-3)) * global_scale
    s2 = depth_t * (d_phi * global_scale)
    s3 = disc_thickness * torch.minimum(s1, s2)

    scales = torch.stack([s1, s2, s3], dim=-1)
    log_scales = torch.log(torch.clamp(scales, min=1e-5, max=1e2))

    R00, R01, R02 = t1[..., 0], t2[..., 0], dx
    R10, R11, R12 = t1[..., 1], t2[..., 1], dy
    R20, R21, R22 = t1[..., 2], t2[..., 2], dz

    tr = R00 + R11 + R22
    qw = torch.sqrt(torch.clamp(1.0 + tr, min=0.0)) / 2.0
    denom = 4.0 * torch.clamp(qw, min=1e-6)
    qx = (R21 - R12) / denom
    qy = (R02 - R20) / denom
    qz = (R10 - R01) / denom

    quats = torch.stack([qw, qx, qy, qz], dim=-1)
    q_norm = torch.clamp(torch.norm(quats, dim=-1, keepdim=True), min=1e-6)
    quats = quats / q_norm

    valid = torch.isfinite(depth_t) & (depth_t >= min_depth) & (depth_t <= max_depth)
    if mask_t is not None:
        valid = valid & mask_t

    flat_pts = pts[valid]
    flat_rgb = rgb_t[valid]
    flat_scales = log_scales[valid]
    flat_quats = quats[valid]

    return flat_pts, flat_rgb, flat_scales, flat_quats


# ---------------------------------------------------------------------------
# Comprehensive Output Exporter (EXR, NPY, PLY, JSON, PNG)
# ---------------------------------------------------------------------------

def export_all_model_outputs(
    out_dir: Union[str, Path],
    res: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    save_splat_ply: bool = True,
    save_standard_ply: bool = True,
    save_depth_exr: bool = True,
    save_depth_npy: bool = True,
    save_points_exr: bool = True,
    save_points_npy: bool = True,
    save_poses: bool = True,
    save_conf: bool = True,
    is_indoor: bool = True,
    stride: int = 1,
    global_scale: float = 1.2,
    disc_thickness: float = 0.2,
    min_depth: float = 0.1,
    max_depth: Optional[float] = None
):
    """
    Saves complete predictions and intermediate diagnostic outputs for Pi3/Pi3X:
      - splat.ply: 3D Gaussian Splatting binary file
      - pointcloud.ply: Standard colored vertex point cloud
      - depth_{i}.exr / depth_{i}.npy / depth_all.npy: Per-frame local metric depth
      - points_{i}.exr / points_world.npy / local_points.npy: 3D points
      - camera_poses.npy / camera_poses.json: Camera extrinsics (4x4)
      - conf_{i}.png / conf.npy: Confidence maps
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Exporting model outputs to: {out_dir}")

    # Extract tensors
    points = res['points'] if res['points'].ndim == 4 else res['points'][0]  # [N, H, W, 3]
    local_points = res.get('local_points', None)
    if local_points is not None and local_points.ndim == 5:
        local_points = local_points[0]  # [N, H, W, 3]
    
    poses = res.get('camera_poses', None)
    if poses is not None and poses.ndim == 4:
        poses = poses[0]  # [N, 4, 4]
        
    conf = res.get('conf', None)
    if conf is not None and conf.ndim == 5:
        conf = conf[0]  # [N, H, W, 1]

    if imgs.ndim == 5:
        imgs = imgs[0]
    if imgs.shape[1] == 3 and imgs.ndim == 4:
        imgs_rgb = imgs.permute(0, 2, 3, 1)  # [N, H, W, 3]
    else:
        imgs_rgb = imgs

    N, H, W, _ = points.shape

    # 1. 3D Gaussian Splat PLY
    if save_splat_ply:
        splat_path = out_dir / "splat.ply"
        flat_pts, flat_rgb, flat_scs, flat_qts, flat_op = pi3_points_to_gaussians_torch(
            points=points,
            imgs=imgs_rgb,
            conf=conf,
            masks=masks,
            stride=stride,
            is_indoor=is_indoor,
            global_scale=global_scale,
            disc_thickness=disc_thickness,
            min_depth=min_depth,
            max_depth=max_depth
        )
        save_gaussian_splat_ply(splat_path, flat_pts, flat_rgb, flat_scs, flat_qts, flat_op)

    # 2. Standard Vertex Colored Point Cloud PLY
    if save_standard_ply:
        std_ply_path = out_dir / "pointcloud.ply"
        if masks is not None:
            m = masks.detach().cpu().numpy()
            pts_np = points.detach().cpu().numpy()[m]
            rgb_np = imgs_rgb.detach().cpu().numpy()[m]
        else:
            pts_np = points.detach().cpu().numpy().reshape(-1, 3)
            rgb_np = imgs_rgb.detach().cpu().numpy().reshape(-1, 3)
        
        if rgb_np.max() <= 1.05:
            rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            rgb_np = rgb_np.clip(0, 255).astype(np.uint8)

        # Write vertex ply
        from pi3.utils.basic import write_ply
        write_ply(torch.as_tensor(pts_np), torch.as_tensor(rgb_np), str(std_ply_path))
        print(f"✅ Saved standard point cloud -> {std_ply_path}")

    # 3. Depth Maps (EXR & NPY)
    if local_points is not None:
        depths = local_points[..., 2].detach().cpu().numpy()  # [N, H, W] metric Z
    else:
        depths = np.linalg.norm(points.detach().cpu().numpy(), axis=-1)

    if save_depth_npy:
        np.save(out_dir / "depth_all.npy", depths.astype(np.float32))
        print(f"✅ Saved depth_all.npy -> {out_dir / 'depth_all.npy'}")

    if save_depth_exr and cv2 is not None:
        depth_dir = out_dir / "depth_exr"
        depth_dir.mkdir(exist_ok=True)
        for i in range(N):
            exr_path = depth_dir / f"depth_{i:04d}.exr"
            cv2.imwrite(str(exr_path), depths[i].astype(np.float32))
        print(f"✅ Saved per-frame depth EXR ({N} frames) -> {depth_dir}")

    # 4. Points Maps (EXR & NPY)
    if save_points_npy:
        np.save(out_dir / "points_world.npy", points.detach().cpu().numpy().astype(np.float32))
        if local_points is not None:
            np.save(out_dir / "local_points.npy", local_points.detach().cpu().numpy().astype(np.float32))
        print(f"✅ Saved points_world.npy & local_points.npy -> {out_dir}")

    if save_points_exr and cv2 is not None:
        pts_dir = out_dir / "points_exr"
        pts_dir.mkdir(exist_ok=True)
        pts_np = points.detach().cpu().numpy()
        for i in range(N):
            exr_path = pts_dir / f"points_{i:04d}.exr"
            # OpenCV writes BGR/multichannel EXR
            cv2.imwrite(str(exr_path), pts_np[i].astype(np.float32))
        print(f"✅ Saved per-frame points EXR ({N} frames) -> {pts_dir}")

    # 5. Camera Poses (NPY & JSON)
    if save_poses and poses is not None:
        poses_np = poses.detach().cpu().numpy().astype(np.float32)
        np.save(out_dir / "camera_poses.npy", poses_np)
        poses_list = poses_np.tolist()
        with open(out_dir / "camera_poses.json", "w") as f:
            json.dump(poses_list, f, indent=2)
        print(f"✅ Saved camera_poses.npy & camera_poses.json -> {out_dir}")

    # 6. Confidence Maps (NPY & PNG)
    if save_conf and conf is not None:
        conf_np = conf.squeeze(-1).detach().cpu().numpy().astype(np.float32)
        conf_prob = 1.0 / (1.0 + np.exp(-conf_np))
        np.save(out_dir / "conf.npy", conf_prob)
        conf_dir = out_dir / "conf_png"
        conf_dir.mkdir(exist_ok=True)
        for i in range(N):
            conf_u8 = (np.clip(conf_prob[i], 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(conf_u8).save(conf_dir / f"conf_{i:04d}.png")
        print(f"✅ Saved conf.npy & per-frame confidence PNGs -> {conf_dir}")
