# import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def process_folder(input_folder, 
                   output_folder, 
                   model_name, 
                   image_size, 
                   device):
    batch_size = 1
    schedule = 'cosine' # schedule: "linear" | "cosine"
    lr = 0.01
    niter = 300

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load DUSt3R model
    print("Loading model...")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    model.eval()

    input_files = sorted([f for f in input_folder.iterdir() if f.suffix in ('.png', '.jpg', '.jpeg', '.bmp')])
    input_files_str = [str(f) for f in input_files]

    if not input_files:
        print("No valid image files found in the input folder.")
        return
    
    print(f"Found {len(input_files)} images. Processing...")

    # Load images and create pairs (assume monocular images for this example)
    images = load_images(input_files_str, size=image_size)

    # scene_graph: "swin", "logwin", "complete"
    pairs = make_pairs(images, scene_graph='swin', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device="cpu", mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    # outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
    #                                   clean_depth, transparent_cams, cam_size)
    
    # Subdirectories
    imgs_dir = output_folder / "imgs"
    conf_mask_dir = output_folder / "conf_mask"
    conf_dir = output_folder / "conf"
    depths_dir = output_folder / "depths"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    conf_mask_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)

    # retrieve useful values from scene:
    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = to_numpy(scene.get_masks())
    confs = to_numpy([c for c in scene.im_conf])
    depths = to_numpy(scene.get_depthmaps())  # Get depth maps in raw format

    n = len(imgs)
    assert len(imgs) == len(depths) == len(confidence_masks) == len(focals) == len(poses)

    print(
        f"imgs: {type(imgs)}, shape: {imgs.shape if hasattr(imgs, 'shape') else 'N/A'}",
        f"imgs[0]: {type(imgs[0])}, shape: {imgs[0].shape if hasattr(imgs[0], 'shape') else 'N/A'}",
        f"to_numpy([im for im in imgs]): {type(to_numpy([im for im in imgs]))}, shape: {to_numpy([im for im in imgs]).shape if hasattr(to_numpy([im for im in imgs]), 'shape') else 'N/A'}",
        f"focals: {type(focals)}, shape: {focals.shape if hasattr(focals, 'shape') else 'N/A'}",
        f"poses: {type(poses)}, shape: {poses.shape if hasattr(poses, 'shape') else 'N/A'}",
        f"pts3d: {type(pts3d)}, shape: {pts3d.shape if hasattr(pts3d, 'shape') else 'N/A'}",
        f"confidence_masks: {type(confidence_masks)}, shape: {confidence_masks.shape if hasattr(confidence_masks, 'shape') else 'N/A'}",
        f"confidence_masks[0]: {type(confidence_masks[0])}, shape: {confidence_masks[0].shape if hasattr(confidence_masks[0], 'shape') else 'N/A'}",
        f"confs: {type(confs)}, shape: {confs.shape if hasattr(confs, 'shape') else 'N/A'}",
        f"depths: {type(depths)}, shape: {depths.shape if hasattr(depths, 'shape') else 'N/A'}",
        sep="\n"
    )

    # Save images and related data
    for i in range(n):
        # Save images
        img = imgs[i]
        img_name = str(input_files[i]) + '.png'
        img_path = imgs_dir / img_name
        cv2.imwrite(str(img_path), img)

        # Save confidence masks
        mask = confidence_masks[i]
        mask_name = str(input_files[i]) + '_mask.png'
        mask_path = conf_mask_dir / mask_name
        cv2.imwrite(str(mask_path), mask)

        # Save confidence values
        conf = confs[i]
        conf_name = str(input_files[i]) + '_conf.npy'
        conf_path = conf_dir / conf_name
        np.save(str(conf_path), conf)

        # Save depth maps
        depth = depths[i]
        depth_name = str(input_files[i]) + '_depth.npy'
        depth_path = depths_dir / depth_name
        np.save(str(depth_path), depth)

    # Save focal lengths to YAML
    with open(output_folder / 'focals.yaml', 'w') as f:
        yaml.dump(focals, f)

    # Save poses to YAML
    with open(output_folder / 'poses.yaml', 'w') as f:
        yaml.dump(poses, f)

    # Create and save point cloud (scene.ply)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    o3d.io.write_point_cloud(str(output_folder / 'scene.ply'), pcd)

    print(f"Scene data saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of images to depth images using DUSt3R.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input folder containing images.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output folder for saving depth images.")
    parser.add_argument('--weights', type=str, default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="Name of the pre-trained DUSt3R model.")
    parser.add_argument('--image_size', type=int, default=512, help="Size to resize images before processing.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()

    process_folder(args.input, args.output, args.weights, args.image_size, args.device)

# python run_dust3r.py \
#     --input ~/real2sim/data/cooler_closed/images_2 \
#     --output ~/real2sim/data/cooler_closed/depth_tmp/ \
#     --weights "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
#     --image_size 512 \
#     --device cuda

# python run_dust3r.py --input ../rsrd/eric_data/scans/cooler_2/images_4 --output output/cooler_2/ > output/cooler_2/run.log