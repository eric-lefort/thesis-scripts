#!/usr/bin/env python3
"""
convert_gaussians_to_mesh.py
-----------------------------
Load 3D Gaussians (.pt or .ply) → Rasterize → Marching Cubes → Save Mesh.
"""

import torch
import numpy as np
from skimage import measure
import open3d as o3d
from plyfile import PlyData
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────
# INFILE = Path("clusters/cluster_0/cooler_5/dig/2025-04-03_094939/nerfstudio_models/step-000014999.ckpt")   # or .ply
INFILE = Path("clusters/cluster_2/splat.ply")   # or .ply
GRID_SIZE = 92                           # Grid resolution (try 256, 512)
SIGMA_SCALE = 1.0                         # Adjust Gaussian width (maybe 0.8–1.5)
ISO_LEVEL = 0.01                          # Threshold for Marching Cubes
DEVICE = "cuda"                           # 'cuda' or 'cpu'
# ──────────────────────────────────────────────────────────────────────


def load_splats_pt(path: Path):
    """Load .pt file and extract Gaussians."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    data = data["pipeline"]
    # debug
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}")
    means = data["_model.gauss_params.means"]            # (N,3)
    scales = data["_model.gauss_params.scales"]          # (N,3)
    quats = data["_model.gauss_params.quats"]            # (N,4)
    opacities = data.get("_model.gauss_params.opacities", torch.ones(means.shape[0], 1))  # fallback
    # apply sigmoid to opacities
    opacities = torch.sigmoid(opacities).squeeze(-1)    # (N,)
    print(f"Opacities min={opacities.min().item():.6f}, max={opacities.max().item():.6f}")
    print(f"Means range: min {means.min(0).values}, max {means.max(0).values}")
    print(f"Scales range: min {scales.min(0).values}, max {scales.max(0).values}")

    return means, scales, quats, opacities

def load_splats_ply(path: Path):
    """Load .ply file and extract Gaussians."""
    plydata = PlyData.read(str(path))
    vertex = plydata['vertex']

    means = torch.stack([
        torch.from_numpy(vertex['x'].astype(np.float32)),
        torch.from_numpy(vertex['y'].astype(np.float32)),
        torch.from_numpy(vertex['z'].astype(np.float32))
    ], dim=1)

    scales = torch.stack([
        torch.from_numpy(vertex['scale_0'].astype(np.float32)),
        torch.from_numpy(vertex['scale_1'].astype(np.float32)),
        torch.from_numpy(vertex['scale_2'].astype(np.float32))
    ], dim=1)
    scales = torch.exp(scales)

    quats = torch.stack([
        torch.from_numpy(vertex['rot_0'].astype(np.float32)),
        torch.from_numpy(vertex['rot_1'].astype(np.float32)),
        torch.from_numpy(vertex['rot_2'].astype(np.float32)),
        torch.from_numpy(vertex['rot_3'].astype(np.float32))
    ], dim=1)

    if 'opacity' in vertex.data.dtype.names:
        opacities = torch.from_numpy(vertex['opacity'].astype(np.float32))
        opacities = torch.sigmoid(opacities).squeeze(-1)
    else:
        opacities = torch.ones(means.shape[0])

    print(f"Loaded {means.shape[0]} Gaussians from {path}")
    print(f"Opacities min={opacities.min().item():.6f}, max={opacities.max().item():.6f}")
    print(f"Means range: min {means.min(0).values}, max {means.max(0).values}")
    print(f"Scales range: min {scales.min(0).values}, max {scales.max(0).values}")
    return means, scales, quats, opacities

def build_grid(means, scales, opacities, grid_size, sigma_scale):
    """Rasterize Gaussians into a voxel grid (GPU)."""
    # 1. Normalize coordinates to [0,1]
    mins = means.min(0).values
    maxs = means.max(0).values
    center = (mins + maxs) / 2
    scale = (maxs - mins).max()

    normalized = (means - center) / scale + 0.5

    # 2. Setup grid
    voxel_size = 1.0 / grid_size
    grid = torch.zeros((grid_size, grid_size, grid_size), device=means.device)

    # 3. Accumulate each Gaussian
    for i in range(len(means)):
        mu = normalized[i]
        s = scales[i] * sigma_scale
        a = opacities[i].clamp(0, 1)
        # debug

        # Generate 3D grid indices around the center
        radius = (s.max() * 3 / scale).item()  # 3-sigma range
        low = ((mu - radius) * grid_size).clamp(0, grid_size-1).long()
        high = ((mu + radius) * grid_size).clamp(0, grid_size-1).long() + 1
        
        if (low >= high).any():
            print(f"Skipping degenerate Gaussian {i}: low={low}, high={high}")
            continue  # skip degenerate Gaussians
        xs = torch.arange(low[0], high[0], device=grid.device)
        ys = torch.arange(low[1], high[1], device=grid.device)
        zs = torch.arange(low[2], high[2], device=grid.device)

        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            continue

        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).float() / grid_size

        diff = coords - mu[None,None,None,:]
        # print the first 5 values of diff
        dist2 = (diff**2 / (s[None,None,None,:]**2 + 1e-6)).sum(-1)

        influence = a * torch.exp(-0.5 * dist2)

        grid[xs[:,None,None], ys[None,:,None], zs[None,None,:]] += influence

    return grid


def save_mesh(vertices, faces, out_path="output_mesh.ply"):
    """Save vertices/faces as .ply mesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved mesh: {out_path}")


def main():
    means, scales, quats, opacities = load_splats_ply(INFILE)
    means, scales, opacities = means.to(DEVICE), scales.to(DEVICE), opacities.to(DEVICE)

    print(f"Loaded {means.shape[0]} Gaussians")

    scale = (means.max(0).values - means.min(0).values).max().cpu().numpy()
    center = ((means.max(0).values + means.min(0).values) / 2).cpu().numpy()
    grid = build_grid(means, scales, opacities, GRID_SIZE, SIGMA_SCALE)
    print(f"Built grid: {grid.shape}")

    grid_cpu = grid.cpu().numpy()

    # Clamp iso_level inside [min, max]
    min_density, max_density = grid_cpu.min(), grid_cpu.max()
    print(f"Grid min={min_density:.6f}, max={max_density:.6f}")

    safe_iso = np.clip(ISO_LEVEL, min_density + 1e-6, max_density - 1e-6)
    print(f"Using iso-level: {safe_iso:.6f}")

    # Marching cubes
    verts, faces, normals, values = measure.marching_cubes(grid_cpu, level=safe_iso)
    print(f"Extracted mesh: {verts.shape[0]} verts, {faces.shape[0]} faces")

    # After marching_cubes:
    verts = (verts / GRID_SIZE - 0.5) * scale + center

    save_mesh(verts, faces, out_path="mesh_output.ply")


if __name__ == "__main__":
    main()
