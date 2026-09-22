import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
import json
import argparse
import numpy as np
import torch
from typing import Optional, List, Dict, Any

try:
    import click
except ImportError:
    click = None

from pi3.models.pi3x import Pi3X
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, load_multimodal_data, write_ply
from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d
from pi3_splat_utils import (
    save_gaussian_splat_ply,
    pi3_points_to_gaussians_torch,
    export_all_model_outputs,
    depth_to_spherical_gaussians_torch,
    spherical_uv_to_directions_torch
)


# ===========================================================================
# Core Inference Function for Pi3 / Pi3X
# ===========================================================================

def run_pi3x_inference(
    data_path: str,
    output_dir: str = "output",
    model_type: str = "pi3x",
    conditions_path: Optional[str] = None,
    ckpt: Optional[str] = None,
    device_str: str = "cuda",
    interval: int = -1,
    # Splat & Export Options
    save_ply: bool = True,
    save_standard_ply: bool = True,
    ply_is_indoor: bool = True,
    ply_stride: int = 1,
    ply_scale: float = 1.2,
    ply_thickness: float = 0.2,
    ply_min_depth: float = 0.1,
    ply_max_depth: Optional[float] = None,
    conf_threshold: float = 0.1,
    rtol: float = 0.03,
    # Additional Output Exports
    save_depth_exr: bool = False,
    save_depth_npy: bool = False,
    save_points_exr: bool = False,
    save_points_npy: bool = False,
    save_poses: bool = True,
    save_conf: bool = False,
    save_all: bool = False,
):
    """
    Unified high-level pipeline for Pi3 / Pi3X with 3D Gaussian Splatting and detailed metric exports.
    """
    if save_all:
        save_ply = True
        save_standard_ply = True
        save_depth_exr = True
        save_depth_npy = True
        save_points_exr = True
        save_points_npy = True
        save_poses = True
        save_conf = True

    if interval < 0:
        interval = 10 if data_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else 1
    print(f"🔹 Sampling interval: {interval}")

    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Multimodal Conditions (if any)
    poses = None
    depths = None
    intrinsics = None
    conditions = dict(intrinsics=None, poses=None, depths=None)

    if conditions_path is not None and os.path.exists(conditions_path):
        print(f"📂 Loading multimodal conditions from: {conditions_path}")
        data_npz = np.load(conditions_path, allow_pickle=True)
        poses = data_npz.get('poses', None)
        depths = data_npz.get('depths', None)
        intrinsics = data_npz.get('intrinsics', None)
        conditions = dict(intrinsics=intrinsics, poses=poses, depths=depths)

    # 2. Load Input Images
    print(f"🖼️ Loading images from: {data_path}")
    if model_type == 'pi3x':
        imgs, conditions = load_multimodal_data(data_path, conditions, interval=interval, device=device)
        use_multimodal = any(v is not None for v in conditions.values())
    else:
        imgs = load_images_as_tensor(data_path, interval=interval).to(device)
        use_multimodal = False

    # 3. Initialize Model
    print(f"🚀 Initializing {model_type.upper()} model...")
    if model_type == 'pi3x':
        if ckpt is not None:
            model = Pi3X(use_multimodal=use_multimodal).eval()
            if ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(ckpt)
            else:
                weight = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(weight, strict=False)
        else:
            model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
            if not use_multimodal:
                model.disable_multimodal()
    else:
        if ckpt is not None:
            model = Pi3().eval()
            if ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(ckpt)
            else:
                weight = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(weight)
        else:
            model = Pi3.from_pretrained("yyfz233/Pi3").eval()

    model = model.to(device)

    # 4. Model Forward Pass
    print("⚡ Running neural network forward pass...")
    dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=dtype):
            if model_type == 'pi3x':
                res = model(imgs=imgs, **conditions)
            else:
                res = model(imgs[None] if imgs.ndim == 4 else imgs)

    # 5. Mask & Geometry Postprocessing
    print("🔍 Processing validity and normal edge masks...")
    conf_tensor = res.get('conf', None)
    if conf_tensor is not None:
        raw_conf = conf_tensor[..., 0] if conf_tensor.shape[-1] == 1 else conf_tensor
        conf_prob = torch.sigmoid(raw_conf)
        masks = conf_prob > conf_threshold
    else:
        masks = torch.ones_like(res['points'][..., 0], dtype=torch.bool)

    if 'local_points' in res:
        non_edge = ~depth_normal_edge(res['local_points'], rtol=rtol, mask=masks)
        final_mask = torch.logical_and(masks, non_edge)
    else:
        final_mask = masks

    if final_mask.ndim == 4 and final_mask.shape[0] == 1:
        final_mask = final_mask[0]

    # 6. Save Splat and Diagnostic Exports
    export_all_model_outputs(
        out_dir=out_path,
        res=res,
        imgs=imgs,
        masks=final_mask,
        save_splat_ply=save_ply,
        save_standard_ply=save_standard_ply,
        save_depth_exr=save_depth_exr,
        save_depth_npy=save_depth_npy,
        save_points_exr=save_points_exr,
        save_points_npy=save_points_npy,
        save_poses=save_poses,
        save_conf=save_conf,
        is_indoor=ply_is_indoor,
        stride=ply_stride,
        global_scale=ply_scale,
        disc_thickness=ply_thickness,
        min_depth=ply_min_depth,
        max_depth=ply_max_depth
    )

    print(f"\n🎉 Pi3X pipeline execution finished successfully! Outputs written to: {out_path.resolve()}\n")
    return res


# ===========================================================================
# Click CLI Interface (with Sub-Commands & Environment Variable Support)
# ===========================================================================

if click is not None:
    @click.group()
    def cli():
        """Pi3X 3D Gaussian Splatting (splat.ply) & Geometry Export CLI."""
        pass

    @cli.command('predict')
    @click.option('--data_path', '-i', type=str, required=True, envvar='PI3_DATA_PATH', help='Path to image directory, video (.mp4), or single image.')
    @click.option('--output_dir', '-o', type=str, default='output', show_default=True, envvar='PI3_OUTPUT_DIR', help='Directory to save output files.')
    @click.option('--model_type', type=click.Choice(['pi3x', 'pi3']), default='pi3x', show_default=True, help='Model architecture to use.')
    @click.option('--conditions_path', type=str, default=None, envvar='PI3_CONDITIONS_PATH', help='Optional path to multimodal conditions .npz.')
    @click.option('--ckpt', type=str, default=None, envvar='PI3_CKPT', help='Path to local checkpoint (.pt or .safetensors).')
    @click.option('--device', 'device_str', type=str, default='cuda', show_default=True, help='Device (cuda/cpu).')
    @click.option('--interval', type=int, default=-1, help='Frame sampling interval (default: 1 for images, 10 for video).')
    # 3DGS splat.ply CLI Options
    @click.option('--ply/--no-ply', 'save_ply', default=True, show_default=True, envvar='PI3_PLY', help='Save 3D Gaussian Splatting splat.ply file.')
    @click.option('--pointcloud/--no-pointcloud', 'save_standard_ply', default=True, show_default=True, help='Save standard colored vertex pointcloud.ply.')
    @click.option('--ply_is_indoor/--ply_is_outdoor', default=True, show_default=True, envvar='PI3_PLY_IS_INDOOR', help='Scene environment preset (indoor: 15m cutoff; outdoor: 80m cutoff).')
    @click.option('--ply_stride', type=int, default=1, show_default=True, envvar='PI3_PLY_STRIDE', help='Pixel sampling stride for splat generation (1=full res, 2=half res).')
    @click.option('--ply_scale', type=float, default=1.2, show_default=True, envvar='PI3_PLY_SCALE', help='Global splat radius scale multiplier.')
    @click.option('--ply_thickness', type=float, default=0.2, show_default=True, envvar='PI3_PLY_THICKNESS', help='Splat disc thickness ratio.')
    @click.option('--ply_min_depth', type=float, default=0.1, show_default=True, envvar='PI3_PLY_MIN_DEPTH', help='Minimum distance threshold in meters.')
    @click.option('--ply_max_depth', type=float, default=None, envvar='PI3_PLY_MAX_DEPTH', help='Maximum distance cutoff in meters.')
    @click.option('--conf_thre', 'conf_threshold', type=float, default=0.1, show_default=True, help='Confidence threshold for filtering.')
    @click.option('--rtol', type=float, default=0.03, show_default=True, help='Depth normal edge relative tolerance.')
    # Diagnostic metric export options
    @click.option('--maps', 'save_maps', is_flag=True, default=False, envvar='PI3_MAPS', help='Generate ALL additional diagnostic maps (depth.exr/npy, points.exr/npy, camera_poses.npy/json, conf.npy/png).')
    @click.option('--save_depth_exr', is_flag=True, default=False, help='Save 32-bit floating point depth .exr maps.')
    @click.option('--save_depth_npy', is_flag=True, default=False, help='Save numpy depth_all.npy array.')
    @click.option('--save_points_exr', is_flag=True, default=False, help='Save per-frame 3D coordinate .exr files.')
    @click.option('--save_points_npy', is_flag=True, default=False, help='Save points_world.npy and local_points.npy.')
    @click.option('--save_poses', is_flag=True, default=True, show_default=True, help='Save camera_poses.npy and camera_poses.json.')
    @click.option('--save_conf', is_flag=True, default=False, help='Save conf.npy and per-frame confidence PNGs.')
    def predict_cmd(save_maps, **kwargs):
        """Run Pi3 / Pi3X prediction, generate splat.ply, and export 3D outputs."""
        if save_maps:
            kwargs['save_all'] = True
        run_pi3x_inference(**kwargs)

    @cli.command('splat')
    @click.option('--data_path', '-i', type=str, required=True, help='Path to input images/video.')
    @click.option('--output_dir', '-o', type=str, default='output_splat', show_default=True, help='Output directory.')
    @click.option('--model_type', type=click.Choice(['pi3x', 'pi3']), default='pi3x', show_default=True, help='Model architecture.')
    @click.option('--conditions_path', type=str, default=None, help='Optional path to multimodal conditions .npz.')
    @click.option('--ckpt', type=str, default=None, help='Path to local checkpoint.')
    @click.option('--device', 'device_str', type=str, default='cuda', show_default=True, help='Device (cuda/cpu).')
    @click.option('--interval', type=int, default=-1, help='Frame sampling interval.')
    @click.option('--ply_is_indoor/--ply_is_outdoor', default=True, show_default=True, help='Indoor (15m) vs Outdoor (80m) preset.')
    @click.option('--ply_stride', type=int, default=1, show_default=True, help='Pixel sampling stride (1=full res, 2=half res).')
    @click.option('--ply_scale', type=float, default=1.2, show_default=True, help='Splat radius scale multiplier.')
    @click.option('--ply_thickness', type=float, default=0.2, show_default=True, help='Splat disc thickness ratio.')
    @click.option('--ply_min_depth', type=float, default=0.1, show_default=True, help='Minimum depth (meters).')
    @click.option('--ply_max_depth', type=float, default=None, help='Maximum depth cutoff (meters).')
    @click.option('--conf_thre', 'conf_threshold', type=float, default=0.1, show_default=True, help='Confidence threshold.')
    @click.option('--maps', 'save_maps', is_flag=True, default=False, help='Also generate extra diagnostic maps (depth/points EXR/NPY, camera poses, conf).')
    def splat_cmd(data_path, output_dir, save_maps, **kwargs):
        """Dedicated sub-command to generate 3D Gaussian Splatting splat.ply."""
        run_pi3x_inference(
            data_path=data_path,
            output_dir=output_dir,
            save_ply=True,
            save_standard_ply=True,
            save_all=save_maps,
            **kwargs
        )

    @cli.command('maps')
    @click.option('--data_path', '-i', type=str, required=True, help='Path to input images/video.')
    @click.option('--output_dir', '-o', type=str, default='output_maps', show_default=True, help='Output directory.')
    @click.option('--model_type', type=click.Choice(['pi3x', 'pi3']), default='pi3x', show_default=True, help='Model architecture.')
    @click.option('--conditions_path', type=str, default=None, help='Optional path to multimodal conditions .npz.')
    @click.option('--ckpt', type=str, default=None, help='Path to local checkpoint.')
    @click.option('--device', 'device_str', type=str, default='cuda', show_default=True, help='Device (cuda/cpu).')
    @click.option('--interval', type=int, default=-1, help='Frame sampling interval.')
    def maps_cmd(data_path, output_dir, **kwargs):
        """Dedicated sub-command to export all intermediate diagnostic maps."""
        run_pi3x_inference(
            data_path=data_path,
            output_dir=output_dir,
            save_ply=True,
            save_standard_ply=True,
            save_all=True,
            **kwargs
        )


# ===========================================================================
# Argparse CLI Fallback
# ===========================================================================

def build_argparser():
    parser = argparse.ArgumentParser(description="Pi3 / Pi3X 3D Gaussian Splatting & Geometry Exporter CLI")
    parser.add_argument("--data_path", "-i", type=str, default="examples/skating.mp4", help="Path to input images or video")
    parser.add_argument("--output_dir", "-o", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--model_type", type=str, default="pi3x", choices=["pi3x", "pi3"], help="Model type")
    parser.add_argument("--conditions_path", type=str, default=None, help="Path to conditions .npz")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--device", dest="device_str", type=str, default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--interval", type=int, default=-1, help="Frame sampling interval")
    
    # Splat options
    parser.add_argument("--ply", dest="save_ply", action="store_true", default=True, help="Save splat.ply")
    parser.add_argument("--no_ply", dest="save_ply", action="store_false", help="Do not save splat.ply")
    parser.add_argument("--pointcloud", dest="save_standard_ply", action="store_true", default=True, help="Save pointcloud.ply")
    parser.add_argument("--ply_is_indoor", dest="ply_is_indoor", action="store_true", default=True, help="Indoor preset (15m cutoff)")
    parser.add_argument("--ply_is_outdoor", dest="ply_is_indoor", action="store_false", help="Outdoor preset (80m cutoff)")
    parser.add_argument("--ply_stride", type=int, default=1, help="Pixel sampling stride")
    parser.add_argument("--ply_scale", type=float, default=1.2, help="Splat scale multiplier")
    parser.add_argument("--ply_thickness", type=float, default=0.2, help="Splat thickness ratio")
    parser.add_argument("--ply_min_depth", type=float, default=0.1, help="Min depth in meters")
    parser.add_argument("--ply_max_depth", type=float, default=None, help="Max depth in meters")
    parser.add_argument("--conf_thre", dest="conf_threshold", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--rtol", type=float, default=0.03, help="Edge normal tolerance")

    # Exports / Maps
    parser.add_argument("--maps", dest="save_maps", action="store_true", default=False, help="Save ALL extra diagnostic maps (depth.exr/npy, points.exr/npy, camera_poses, conf)")
    parser.add_argument("--save_depth_exr", action="store_true", default=False, help="Save depth EXRs")
    parser.add_argument("--save_depth_npy", action="store_true", default=False, help="Save depth NPY")
    parser.add_argument("--save_points_exr", action="store_true", default=False, help="Save points EXRs")
    parser.add_argument("--save_points_npy", action="store_true", default=False, help="Save points NPY")
    parser.add_argument("--save_poses", action="store_true", default=True, help="Save camera poses")
    parser.add_argument("--save_conf", action="store_true", default=False, help="Save confidence NPY/PNG")
    return parser


if __name__ == '__main__':
    if click is not None and len(sys.argv) > 1 and sys.argv[1] in ['predict', 'splat', 'maps', '--help']:
        cli()
    else:
        parser = build_argparser()
        args = parser.parse_args()
        save_all = args.save_maps
        run_pi3x_inference(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_type=args.model_type,
            conditions_path=args.conditions_path,
            ckpt=args.ckpt,
            device_str=args.device_str,
            interval=args.interval,
            save_ply=args.save_ply,
            save_standard_ply=args.save_standard_ply,
            ply_is_indoor=args.ply_is_indoor,
            ply_stride=args.ply_stride,
            ply_scale=args.ply_scale,
            ply_thickness=args.ply_thickness,
            ply_min_depth=args.ply_min_depth,
            ply_max_depth=args.ply_max_depth,
            conf_threshold=args.conf_threshold,
            rtol=args.rtol,
            save_depth_exr=args.save_depth_exr,
            save_depth_npy=args.save_depth_npy,
            save_points_exr=args.save_points_exr,
            save_points_npy=args.save_points_npy,
            save_poses=args.save_poses,
            save_conf=args.save_conf,
            save_all=save_all
        )
