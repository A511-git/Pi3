#!/usr/bin/env python3
"""
Pi3X 3D Gaussian Splatting CLI
Generates splat.ply for 3D Gaussian Splatting viewers (SuperSplat, PlayCanvas, WebGL).
Optionally generates extra diagnostic maps with --maps.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path

try:
    import click
except ImportError:
    click = None

import argparse
from splat_cli import run_pi3x_inference


if click is not None:
    @click.command(context_settings=dict(help_option_names=['-h', '--help']))
    @click.option('--data_path', '-i', type=str, default='examples/skating.mp4', show_default=True, envvar='PI3_DATA_PATH', help='Path to image directory, video (.mp4), or single image.')
    @click.option('--output_dir', '-o', type=str, default='output', show_default=True, envvar='PI3_OUTPUT_DIR', help='Directory to save splat.ply and generated outputs.')
    @click.option('--model_type', type=click.Choice(['pi3x', 'pi3']), default='pi3x', show_default=True, help='Model architecture.')
    @click.option('--conditions_path', type=str, default=None, envvar='PI3_CONDITIONS_PATH', help='Optional path to multimodal conditions .npz.')
    @click.option('--ckpt', type=str, default=None, envvar='PI3_CKPT', help='Path to local checkpoint (.pt or .safetensors).')
    @click.option('--device', 'device_str', type=str, default='cuda', show_default=True, help='Computation device (cuda/cpu).')
    @click.option('--interval', type=int, default=-1, help='Frame sampling interval (default: 1 for images, 10 for video).')
    
    # Splat parameters
    @click.option('--ply/--no-ply', 'save_ply', default=True, show_default=True, envvar='PI3_PLY', help='Save 3D Gaussian Splatting splat.ply file.')
    @click.option('--pointcloud/--no-pointcloud', 'save_standard_ply', default=True, show_default=True, help='Save standard colored vertex pointcloud.ply.')
    @click.option('--ply_is_indoor/--ply_is_outdoor', default=True, show_default=True, envvar='PI3_PLY_IS_INDOOR', help='Scene preset (indoor: max 15m depth cutoff; outdoor: max 80m cutoff).')
    @click.option('--ply_stride', type=int, default=1, show_default=True, envvar='PI3_PLY_STRIDE', help='Pixel sampling stride (1=full res, 2=half res).')
    @click.option('--ply_scale', type=float, default=1.2, show_default=True, envvar='PI3_PLY_SCALE', help='Global splat radius scale multiplier.')
    @click.option('--ply_thickness', type=float, default=0.2, show_default=True, envvar='PI3_PLY_THICKNESS', help='Splat disc thickness ratio.')
    @click.option('--ply_min_depth', type=float, default=0.1, show_default=True, envvar='PI3_PLY_MIN_DEPTH', help='Minimum distance threshold in meters.')
    @click.option('--ply_max_depth', type=float, default=None, envvar='PI3_PLY_MAX_DEPTH', help='Maximum distance cutoff in meters.')
    @click.option('--conf_thre', 'conf_threshold', type=float, default=0.1, show_default=True, help='Confidence filtering threshold.')
    @click.option('--rtol', type=float, default=0.03, show_default=True, help='Depth normal edge relative tolerance.')
    
    # Diagnostic maps option (all extra stuff wrapped in --maps)
    @click.option('--maps', 'save_maps', is_flag=True, default=False, envvar='PI3_MAPS', help='Generate ALL additional diagnostic maps (depth.exr/npy, points.exr/npy, camera_poses.npy/json, conf.npy/png).')
    def main(data_path, output_dir, model_type, conditions_path, ckpt, device_str, interval,
             save_ply, save_standard_ply, ply_is_indoor, ply_stride, ply_scale, ply_thickness,
             ply_min_depth, ply_max_depth, conf_threshold, rtol, save_maps):
        """Pi3X 3D Gaussian Splatting CLI."""
        run_pi3x_inference(
            data_path=data_path,
            output_dir=output_dir,
            model_type=model_type,
            conditions_path=conditions_path,
            ckpt=ckpt,
            device_str=device_str,
            interval=interval,
            save_ply=save_ply,
            save_standard_ply=save_standard_ply,
            ply_is_indoor=ply_is_indoor,
            ply_stride=ply_stride,
            ply_scale=ply_scale,
            ply_thickness=ply_thickness,
            ply_min_depth=ply_min_depth,
            ply_max_depth=ply_max_depth,
            conf_threshold=conf_threshold,
            rtol=rtol,
            save_all=save_maps
        )
else:
    def main():
        parser = argparse.ArgumentParser(description="Pi3X 3D Gaussian Splatting CLI")
        parser.add_argument("--data_path", "-i", type=str, default="examples/skating.mp4", help="Path to input images or video")
        parser.add_argument("--output_dir", "-o", type=str, default="output", help="Directory to save output files")
        parser.add_argument("--model_type", type=str, default="pi3x", choices=["pi3x", "pi3"], help="Model type")
        parser.add_argument("--conditions_path", type=str, default=None, help="Path to conditions .npz")
        parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
        parser.add_argument("--device", dest="device_str", type=str, default="cuda", help="Computation device (cuda/cpu)")
        parser.add_argument("--interval", type=int, default=-1, help="Frame sampling interval")
        
        parser.add_argument("--ply", dest="save_ply", action="store_true", default=True, help="Save splat.ply")
        parser.add_argument("--no_ply", dest="save_ply", action="store_false", help="Do not save splat.ply")
        parser.add_argument("--pointcloud", dest="save_standard_ply", action="store_true", default=True, help="Save pointcloud.ply")
        parser.add_argument("--ply_is_indoor", dest="ply_is_indoor", action="store_true", default=True, help="Indoor scene preset")
        parser.add_argument("--ply_is_outdoor", dest="ply_is_indoor", action="store_false", help="Outdoor scene preset")
        parser.add_argument("--ply_stride", type=int, default=1, help="Pixel sampling stride")
        parser.add_argument("--ply_scale", type=float, default=1.2, help="Splat scale multiplier")
        parser.add_argument("--ply_thickness", type=float, default=0.2, help="Splat disc thickness ratio")
        parser.add_argument("--ply_min_depth", type=float, default=0.1, help="Minimum depth in meters")
        parser.add_argument("--ply_max_depth", type=float, default=None, help="Maximum depth in meters")
        parser.add_argument("--conf_thre", dest="conf_threshold", type=float, default=0.1, help="Confidence threshold")
        parser.add_argument("--rtol", type=float, default=0.03, help="Edge normal tolerance")
        parser.add_argument("--maps", dest="save_maps", action="store_true", default=False, help="Save all extra diagnostic maps (depth.exr/npy, points.exr/npy, camera_poses, conf)")
        
        args = parser.parse_args()
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
            save_all=args.save_maps
        )


if __name__ == '__main__':
    main()
