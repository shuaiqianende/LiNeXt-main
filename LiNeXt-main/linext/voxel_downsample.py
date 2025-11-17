import os
import torch
import numpy as np
import click
from natsort import natsorted
from utils.pcd_preprocess import load_poses
from utils.pcd_transforms import (
    rotate_point_cloud,
    rotate_perturbation_point_cloud,
    random_scale_point_cloud,
    random_flip_point_cloud,
)

# -----------------------------------------------------------------------------
# Efficient voxel-grid downsampling for both points and labels
# -----------------------------------------------------------------------------
def efficient_voxel_downsample(points, labels, voxel_size, device="cuda:0"):
    """
    Fast voxel-grid downsampling that returns both downsampled points
    and their corresponding labels.

    Args:
        points (np.ndarray): (N, 3) point cloud coordinates.
        labels (np.ndarray or None): (N,) integer labels. Pass None if not needed.
        voxel_size (float): voxel edge length.

    Returns:
        down_pts (np.ndarray): downsampled points.
        down_lbs (np.ndarray): downsampled labels (or None if labels was None).
    """
    voxel_indices = torch.floor(points / voxel_size).to(torch.int64)
    unique, inverse = torch.unique(voxel_indices, dim=0, return_inverse=True)
    _, repr_ids = torch.sort(inverse)
    repr_ids = repr_ids[torch.unique(inverse, return_counts=True)[1].cumsum(0) - 1]

    down_pts = points[repr_ids]
    down_lbs = labels[repr_ids] if labels is not None else None
    return down_pts, down_lbs

# -----------------------------------------------------------------------------
# Binary search for voxel size that yields ≈ target points
# -----------------------------------------------------------------------------
def auto_voxel_downsample(pts, lbs, target=180_000, init_vs=0.5, tol=1_000,
                          vs_min=0.1, vs_max=4.0, max_iters=20):
    """
    Binary-search voxel size so that the downsampled point count N satisfies
    target <= N < target + tol.

    Args:
        pts (torch.Tensor): (N, 3) coordinates.
        lbs (torch.Tensor or None): (N,) labels.
        ... (other args are self-explanatory).

    Returns:
        down_pts (np.ndarray), down_lbs (np.ndarray), best_vs (float)
    """
    lo, hi = vs_min, vs_max
    vs = init_vs
    for i in range(max_iters):
        down_pts, down_lbs = efficient_voxel_downsample(pts, lbs, vs)
        N = down_pts.shape[0]
        diff = N - target

        if 0 <= diff < tol:
            print(f"[iter {i}] vs={vs:.4f}, N={N} (diff={diff})  ✓")
            return down_pts, down_lbs, vs

        if diff >= tol:
            lo = vs        # Too many points → coarser
        else:
            hi = vs        # Too few points → finer
        vs = 0.5 * (lo + hi)
        print(f"[iter {i}] vs={vs:.4f}, N={N} (diff={diff}), range=({lo:.4f},{hi:.4f})")

    print(f"Not converged within tolerance; returning vs={vs:.4f}, N={N}, diff={diff}")
    return down_pts, down_lbs, vs

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Generic placeholders; replace with your own paths or argparse/click args
    # -------------------------------------------------------------------------
    run_name        = "map_clean_0"
    data_root       = "<PATH_TO_DATA_DIR>"   # e.g., /path/to/KITTI_odometry/09
    in_file         = f"{data_root}/{run_name}.npy"
    out_file        = f"{data_root}/{run_name}_downsample.npy"

    target       = 180_000   # desired point count after downsampling
    init_vs      = 0.5       # initial voxel size (m)
    tol          = 1_000     # tolerance on point count
    vs_min, vs_max = 0.1, 4.0
    max_iters    = 20
    device       = "cuda:0"  # change to "cpu" if no GPU

    data = np.load(in_file)

    # Separate coordinates and labels (if any)
    if data.ndim == 2 and data.shape[1] >= 4:
        pts_np = data[:, :3]
        lbs_np = data[:, 3].astype(np.int32)
        lbs_t  = torch.from_numpy(lbs_np).to(device)
    else:
        pts_np = data
        lbs_t  = None

    pts_t = torch.from_numpy(pts_np).float().to(device)

    # Run auto voxel-size search
    d_pts, d_lbs, best_vs = auto_voxel_downsample(
        pts_t, lbs_t,
        target=target, init_vs=init_vs, tol=tol,
        vs_min=vs_min, vs_max=vs_max, max_iters=max_iters
    )

    # Save results
    np.save(out_file, d_pts.cpu().numpy())
    print(f"points: {pts_np.shape[0]} -> {d_pts.shape[0]}, vs={best_vs:.4f}, saved: {out_file}")
