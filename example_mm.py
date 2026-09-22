import torch
import argparse
import numpy as np
import os
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d
from pi3.models.pi3x import Pi3X

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    # parser.add_argument("--conditions_path", type=str, default='examples/room/condition.npz',
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")

    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--splat_path", type=str, default=None,
                        help="Path to save 3D Gaussian Splatting splat.ply file (default: save_path with .splat.ply or splat.ply in output folder).")
    parser.add_argument("--ply", action="store_true", default=True,
                        help="Save 3D Gaussian Splatting splat.ply file.")
    parser.add_argument("--ply_is_indoor", action="store_true", default=True,
                        help="Indoor scene preset (default cutoff 15m).")
    parser.add_argument("--ply_is_outdoor", dest="ply_is_indoor", action="store_false",
                        help="Outdoor scene preset (default cutoff 80m).")
    parser.add_argument("--ply_stride", type=int, default=1,
                        help="Pixel sampling stride for splat generation (1=full res, 2=half res).")
    parser.add_argument("--ply_scale", type=float, default=1.2,
                        help="Global splat radius scale multiplier.")
    parser.add_argument("--ply_thickness", type=float, default=0.2,
                        help="Splat disc thickness ratio.")
    parser.add_argument("--ply_min_depth", type=float, default=0.1,
                        help="Minimum distance threshold in meters.")
    parser.add_argument("--ply_max_depth", type=float, default=None,
                        help="Maximum distance cutoff in meters.")
    parser.add_argument("--maps", dest="save_maps", action="store_true", default=False,
                        help="Generate ALL additional diagnostic maps (depth.exr/npy, points.exr/npy, camera_poses.npy/json, conf.npy/png).")
    parser.add_argument("--save_depth_exr", action="store_true", default=False,
                        help="Save depth maps as 32-bit floating point EXR files.")
    parser.add_argument("--save_depth_npy", action="store_true", default=False,
                        help="Save depth maps as numpy .npy array.")
    parser.add_argument("--save_points_exr", action="store_true", default=False,
                        help="Save 3D coordinate point maps as EXR files.")
    parser.add_argument("--save_points_npy", action="store_true", default=False,
                        help="Save points as numpy .npy array.")
    parser.add_argument("--save_poses", action="store_true", default=True,
                        help="Save camera poses as .npy and .json.")
    parser.add_argument("--save_conf", action="store_true", default=False,
                        help="Save confidence maps as .npy and .png.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare input data
    device = torch.device(args.device)

    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 
    use_multimodal = any(v is not None for v in conditions.values())
    if not use_multimodal:
        print("No multimodal conditions found. Disable multimodal branch to reduce memory usage.")

    # 2. Prepare model
    print(f"Loading model...")
    if args.ckpt is not None:
        model = Pi3X(use_multimodal=use_multimodal).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`
        if not use_multimodal:
            model.disable_multimodal()
    model = model.to(device)

    """
    Args:
        imgs (torch.Tensor): Input RGB images valued in [0, 1].
            Shape: (B, N, 3, H, W).
        intrinsics (torch.Tensor, optional): Camera intrinsic matrices.
            Shape: (B, N, 3, 3).
            Values are in pixel coordinates (not normalized).
        rays (torch.Tensor, optional): Pre-computed ray directions (unit vectors).
            Shape: (B, N, H, W, 3).
            Can replace `intrinsics` as a geometric condition.
        poses (torch.Tensor, optional): Camera-to-World matrices.
            Shape: (B, N, 4, 4).
            Coordinate system: OpenCV convention (Right-Down-Forward).
        depths (torch.Tensor, optional): Ground truth or prior depth maps.
            Shape: (B, N, H, W).
            Invalid values (e.g., sky or missing data) should be set to 0.
        mask_add_depth (torch.Tensor, optional): Mask for depth condition.
            Shape: (B, N, N).
        mask_add_ray (torch.Tensor, optional): Mask for ray/intrinsic condition.
            Shape: (B, N, N).
        mask_add_pose (torch.Tensor, optional): Mask for pose condition.
            Shape: (B, N, N).
            Note: Requires at least two frames to be True to establish a meaningful
            coordinate system (absolute pose for a single frame provides no relative constraint).
    """

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(
                imgs=imgs, 
                **conditions
            )

    # 3.5 Recover intrinsic from rays_d
    rays_d = torch.nn.functional.normalize(res['local_points'], dim=-1)
    K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    print(f"Recovered frist frame intrinsic: \n{K[0, 0].cpu().numpy()}")
    if conditions['intrinsics'] is not None:
        print(f"Original frist frame intrinsic: \n{conditions['intrinsics'][0, 0].cpu().numpy()}")

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_normal_edge(res['local_points'], rtol=0.03, mask=masks)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points & Splat PLY
    out_dir = os.path.dirname(args.save_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving standard point cloud to: {args.save_path}")
    write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], args.save_path)

    # 6. Save 3D Gaussian Splat PLY & Exports
    from pi3_splat_utils import export_all_model_outputs, save_gaussian_splat_ply, pi3_points_to_gaussians_torch

    save_maps = args.save_maps
    if save_maps or args.save_depth_exr or args.save_depth_npy or args.save_points_exr or args.save_points_npy or args.save_conf:
        export_all_model_outputs(
            out_dir=out_dir,
            res=res,
            imgs=imgs,
            masks=masks,
            save_splat_ply=args.ply,
            save_standard_ply=False, # already saved above
            save_depth_exr=args.save_depth_exr or save_maps,
            save_depth_npy=args.save_depth_npy or save_maps,
            save_points_exr=args.save_points_exr or save_maps,
            save_points_npy=args.save_points_npy or save_maps,
            save_poses=args.save_poses or save_maps,
            save_conf=args.save_conf or save_maps,
            is_indoor=args.ply_is_indoor,
            stride=args.ply_stride,
            global_scale=args.ply_scale,
            disc_thickness=args.ply_thickness,
            min_depth=args.ply_min_depth,
            max_depth=args.ply_max_depth
        )
    elif args.ply:
        splat_file = args.splat_path if args.splat_path else os.path.join(out_dir, "splat.ply")
        flat_pts, flat_rgb, flat_scs, flat_qts, flat_op = pi3_points_to_gaussians_torch(
            points=res['points'][0],
            imgs=imgs[0],
            conf=res['conf'][0],
            masks=masks,
            stride=args.ply_stride,
            is_indoor=args.ply_is_indoor,
            global_scale=args.ply_scale,
            disc_thickness=args.ply_thickness,
            min_depth=args.ply_min_depth,
            max_depth=args.ply_max_depth
        )
        save_gaussian_splat_ply(splat_file, flat_pts, flat_rgb, flat_scs, flat_qts, flat_op)

    print("Done.")
