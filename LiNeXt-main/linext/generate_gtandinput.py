import os
import torch
import numpy as np
from natsort import natsorted
import open3d as o3d
import MinkowskiEngine as ME

from utils.pcd_preprocess import load_poses

# --------------- Configuration ---------------
device = 'cuda:0'
data_dir = '/data-12/M2024-HWZ/KITTI_Odometry/'  # Root directory containing sequence folders
seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']  # Sequences to preprocess
split = 'train'  # 'train' / 'val' / 'test'
max_range = 50.0

# --------------- Data Loading ---------------
# Load poses and maps for each sequence
seq_poses = {}
cache_maps = {}

for seq in seqs:
    seq_path = os.path.join(data_dir, seq)

    # Load calibration and poses (KITTI format)
    poses = load_poses(
        os.path.join(seq_path, 'calib.txt'),
        os.path.join(seq_path, 'poses.txt'),
    )
    seq_poses[seq] = poses

    # Preload clean map point clouds for non-test splits
    if split != 'test':
        cache_maps[seq] = np.load(os.path.join(seq_path, 'map_clean.npy'))

# Transfer cached maps to GPU as double precision tensors
cache_maps_tensor = {
    seq: torch.from_numpy(cache_maps[seq]).to(device).double()
    for seq in cache_maps
}

# Collect all point cloud file paths into a dictionary per sequence
points_datapath = {}
for seq in seqs:
    seq_path = os.path.join(data_dir, seq, 'velodyne')
    bins = natsorted(os.listdir(seq_path))
    points_datapath[seq] = [os.path.join(seq_path, fname) for fname in bins]


# --------------- Helper Functions ---------------
def efficient_voxel_downsample(map_points, labels, voxel_size):
    """Downsample point cloud efficiently using MinkowskiEngine sparse quantization."""
    _, mapping = ME.utils.sparse_quantize(
        coordinates=map_points / voxel_size,
        return_index=True,
        device=device
    )
    map_points = map_points[mapping]

    if labels is not None:
        labels = labels[mapping]
        return map_points, labels
    return map_points, None


def auto_voxel_downsample(pts, lbs, target=180000, init_vs=0.5, tol=1000, vs_min=0.1, vs_max=1.1, max_iters=20):
    """
    Binary search for the ideal voxel_size so the downsampled points count N satisfies:
    target <= N < target + tol
    """
    lo, hi = vs_min, vs_max
    vs = init_vs

    for i in range(max_iters):
        # 1) Downsample using current voxel_size
        down_pts, down_lbs = efficient_voxel_downsample(pts, lbs, voxel_size=vs)
        N = down_pts.shape[0]
        diff = N - target

        # 2) Check if current point count is within tolerance
        if 0 <= diff < tol:
            print(f"[iter {i}] vs={vs:.4f}, N={N} (diff={diff})  ✓")
            return down_pts, down_lbs, vs

        # 3) Adjust search range based on difference
        if diff >= tol:
            lo = vs  # Too many points, increase voxel size (coarser)
        else:
            hi = vs  # Too few points, decrease voxel size (finer)

        vs = 0.5 * (lo + hi)
        print(f"[iter {i}] vs={vs:.4f}, N={N} (diff={diff}), new range=({lo:.4f},{hi:.4f})")

    print(f"Failed to converge within tolerance. Returning last iteration: vs={vs:.4f}, N={N}, diff={diff}")
    return down_pts, down_lbs, vs


def process_one_tensor(path, frame_idx):
    """Load, preprocess, and downsample a single frame's point cloud and map."""
    # 1. Load partial point cloud (current frame)
    p_part_np = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    p_part = torch.from_numpy(p_part_np).to(device)

    # Create Open3D voxel grid for viewpoint checking later
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part_np)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.0)

    # 2. Filter out dynamic objects using labels (if available)
    if split != 'test':
        lbl_file = path.replace('velodyne', 'labels').replace('.bin', '.label')
        l_set = np.fromfile(lbl_file, dtype=np.uint32).reshape(-1) & 0xFFFF
        static_idx = (l_set < 252) & (l_set > 1)
        p_part = p_part[static_idx]

    # 3. Apply range and height filtering to partial point cloud
    d = torch.norm(p_part, dim=1)
    mask = (d < max_range) & (d > 3.5) & (p_part[:, 2] > -4)
    p_part = p_part[mask]

    # 4. Process global map points relative to current pose
    seq = os.path.normpath(path).split(os.sep)[-3]
    pose = torch.from_numpy(seq_poses[seq][frame_idx]).to(device)
    t = pose[:-1, -1]
    p_map = cache_maps_tensor[seq]

    # Extract map points within range of current translation
    df = torch.norm(p_map - t, dim=1)
    p_full = p_map[df < max_range]

    # Transform map points to the local coordinate system
    ones = torch.ones(p_full.shape[0], 1, device=device, dtype=p_full.dtype)
    p_full_homogeneous = torch.cat([p_full, ones], dim=1)
    p_full = (p_full_homogeneous @ torch.inverse(pose).T)[:, :3]

    # Filter map points by height
    p_full = p_full[p_full[:, 2] > -4]

    # Keep only map points that fall within the current viewpoint grid
    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full.cpu().numpy()))
    p_full = p_full[np.asarray(in_viewpoint)]

    # 5. Downsample both point clouds to target sizes
    p_full, _, _ = auto_voxel_downsample(
        p_full, None, target=180000, init_vs=0.2, tol=1000,
        vs_min=0.1, vs_max=0.5, max_iters=20
    )
    p_part, _, _ = auto_voxel_downsample(
        p_part, None, target=18000, init_vs=0.2, tol=100,
        vs_min=0.1, vs_max=0.5, max_iters=20
    )

    # 6. Randomly sub-sample to get exact target sizes
    idx_full = np.random.choice(p_full.shape[0], 180000, replace=False)
    p_full = p_full[idx_full]

    idx_part = np.random.choice(p_part.shape[0], 18000, replace=False)
    p_part = p_part[idx_part]

    return p_full.cpu().numpy(), p_part.cpu().numpy()


# --------------- Main Processing Loop ---------------
for seq, datapath in points_datapath.items():

    # Create input and ground truth directories for the sequence
    save_dir = os.path.join(data_dir, seq)
    input_dir = os.path.join(save_dir, 'input')
    gt_dir = os.path.join(save_dir, 'gt')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for i, path in enumerate(datapath):
        print(f'--- Processing frame {i} / {len(datapath)} in sequence {seq} ---')
        frame_id = os.path.basename(path).replace('.bin', '')

        # Process the point clouds
        p_full, p_part = process_one_tensor(path, i)

        # Convert NumPy arrays to Open3D PointCloud objects
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(p_full)

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(p_part)

        # Save as .pcd files
        o3d.io.write_point_cloud(os.path.join(gt_dir, f'{frame_id}.pcd'), pcd_full)
        o3d.io.write_point_cloud(os.path.join(input_dir, f'{frame_id}.pcd'), pcd_part)
        # -----------------------------------------------------------

print('All sequences processed successfully.')