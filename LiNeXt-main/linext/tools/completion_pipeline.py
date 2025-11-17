import numpy as np
import torch
import open3d as o3d
from pytorch_lightning.core.lightning import LightningModule
import yaml
import os
import tqdm
from natsort import natsorted
import click
import time
import linext.models.linextnet as linextnet
from linext.utils.histogram_metrics import compute_hist_metrics, compute_hist_metrics_torch
from linext.utils.metrics import ChamferDistance, CompletionIoU


completion_iou = CompletionIoU(voxel_sizes=[0.5, 0.2, 0.1]) # 0.5, 0.2, 0.1
chamfer_distance = ChamferDistance()

total_time = 0
time_cnt = 0
total_time_refine = 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def repeat_by_distance_quartiles(pcd_part, repeats=(5, 8, 12, 15)):
    B, P, C = pcd_part.shape
    dist2 = (pcd_part ** 2).sum(dim=-1)  # [B, P]
    _, idx_sort = torch.sort(dist2, dim=-1)  # [B, P]
    idx_sort_exp = idx_sort.unsqueeze(-1).expand(-1, -1, C)
    pcd_sorted = torch.gather(pcd_part, dim=1, index=idx_sort_exp)  # [B, P, C]
    quart = P // 4
    sizes = [quart, quart, quart, P - 3 * quart]
    chunks = torch.split(pcd_sorted, sizes, dim=1)  # [B, size_i, C]
    rep_chunks = [
        chunk.repeat_interleave(rep, dim=1)
        for chunk, rep in zip(chunks, repeats)
    ]
    pcd_repeated = torch.cat(rep_chunks, dim=1)  # [B, P*(sum(repeats)/4), C]
    return pcd_repeated



class LiNextCompletion(LightningModule):
    def __init__(self, n2c_path, refine_path):
        super().__init__()
        n2c_ckpt = torch.load(n2c_path)
        self.save_hyperparameters(n2c_ckpt['hyper_parameters'])
        self.model = linextnet.LiNeXt_N2C()
        self.load_state_dict(n2c_ckpt['state_dict'], strict=False)
        self.model.eval()
        self.cuda()

        if refine_path != None:
            self.model_refine = linextnet.LiNeXt_Refine()
            self.use_refine = True
        else:
            self.use_refine = False
        # for fast sampling
        self.hparams['data']['max_range'] = 50.
        try:
            self.alpha_noise = self.hparams['model']['alpha_noise']
        except (KeyError, TypeError):
            self.alpha_noise = 1
        try:
            self.repeats = self.hparams['model']['repeats']
        except (KeyError, TypeError):
            self.repeats = (5, 8, 12, 15)

        self.cnt = 0

    def preprocess_scan(self, scan):
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < self.hparams['data']['max_range']) & (dist > 3.5)][:,:3]

        # use farthest point sampling
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.farthest_point_down_sample(int(self.hparams['data']['num_points'] / 10))
        scan = torch.tensor(np.array(pcd_scan.points), dtype=torch.float32, device=self.device)
        full_scan = scan.repeat(10,1)
        scan = scan[None,:,:]
        full_scan = full_scan[None,:,:]

        return scan, full_scan

    def postprocess_scan(self, completed_scan, input_scan):
        if isinstance(completed_scan, torch.Tensor):
            completed_scan = completed_scan.cpu().numpy()
        dist = np.sqrt(np.sum((completed_scan)**2, -1))
        post_scan = completed_scan[dist < self.hparams['data']['max_range']]
        max_z = input_scan[...,2].max().item()
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()

        post_scan = post_scan[(post_scan[:,2] < max_z) & (post_scan[:,2] > min_z)]

        return post_scan

    def complete_scan(self, scan):
        scan, full_scan = self.preprocess_scan(scan)
        x_full_no_noise = repeat_by_distance_quartiles(scan, repeats=self.repeats)
        x_full = x_full_no_noise
        x_cond = scan
        if self.use_refine:
            coarse_pcd, refine_pcd = self.forward_refine(x_full, x_cond)
            post_scan = self.postprocess_scan(refine_pcd, scan)
        else:
            coarse_pcd, _ = self.forward(x_full, x_cond)
            post_scan = self.postprocess_scan(coarse_pcd, scan)

        return post_scan, scan

    def forward_refine(self, x_full, x_part):
        with torch.no_grad():
            output = self.model(x_full, x_part)
            x_full = output["p_full"]
            full_hidden = output["hidden_f"]
            x_part = output["p_part"]
            part_feat = output["part_feat"]
            seed_part = output["seed_part"]
            seed_feat = output["seed_feat"]
            x_full = x_full.transpose(1, 2)
            refine_pcd = self.model_refine(x_full, full_hidden, x_part, part_feat, seed_part, seed_feat)
        return x_full, refine_pcd

    def forward(self, x_full, x_part):
        with torch.no_grad():
            start_time = time.time()
            output = self.model(x_full, x_part)
            p_full = output["p_full"]
            end_time = time.time()
            print(f"inference time: {end_time - start_time:.4f} 秒")
        return p_full, None



def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply') or pcd_file.endswith('.pcd'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")


def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses

def get_ground_truth(pose, cur_scan, seq_map, max_range):
    trans = pose[:-1,-1]
    dist_gt = np.sum((seq_map - trans)**2, axis=-1)**.5
    scan_gt = seq_map[dist_gt < max_range]
    scan_gt = np.concatenate((scan_gt, np.ones((len(scan_gt),1))), axis=-1)
    scan_gt = (scan_gt @ np.linalg.inv(pose).T)[:,:3]
    scan_gt = scan_gt[(scan_gt[:,2] > -4.) & (scan_gt[:,2] < 4.4)]
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(scan_gt)

    # filter only over the view point
    cur_pcd = o3d.geometry.PointCloud()
    cur_pcd.points = o3d.utility.Vector3dVector(cur_scan)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cur_pcd, voxel_size=10.)
    in_viewpoint = viewpoint_grid.check_if_included(pcd_gt.points)
    points_gt = np.array(pcd_gt.points)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt[in_viewpoint])

    return pcd_gt


PATH_DATA = './KITTI_Odometry/08'

@click.command()
@click.option('--n2c', '-d', type=str, default=None, help='n2c module checkpoint')
@click.option('--refine', '-r', type=str, default=None, help='refine module checkpoint')
@click.option('--path_scan', '-p', type=str, default=None, help='path to the scan sequence')
@click.option('--use_eval', '-p', type=str, default=False, help='eval the scan sequence')
@click.option('--max_range', '-m', type=float, default=50, help='max range')
def main(n2c, refine, path_scan, use_eval, max_range):
    lidar_completion = LiNextCompletion(n2c, refine)
    if use_eval:
        poses = load_poses(os.path.join(PATH_DATA, 'calib.txt'), os.path.join(PATH_DATA, 'poses.txt'))
        seq_map = np.load(f'{PATH_DATA}/map_clean.npy')

        jsd_3d = []
        jsd_bev = []

        for pose, scan_path in tqdm.tqdm(list(zip(poses, natsorted(os.listdir(f'{PATH_DATA}/velodyne'))))):
            pcd_file = os.path.join(PATH_DATA, 'velodyne', scan_path)
            points = load_pcd(pcd_file)
            complete_point, scan = lidar_completion.complete_scan(points)
            complete_scan = o3d.geometry.PointCloud()
            complete_scan.points = o3d.utility.Vector3dVector(complete_point)
            scan_np = scan.squeeze(0).cpu().numpy().astype(np.float64)  # (N, 3)
            pcd_gt = get_ground_truth(pose, scan_np, seq_map, max_range)
            jsd_3d.append(compute_hist_metrics(pcd_gt, complete_scan, bev=False))
            jsd_bev.append(compute_hist_metrics(pcd_gt, complete_scan, bev=True))
            print(f'JSD 3D: {jsd_3d[-1]}')
            print(f'JSD BEV: {jsd_bev[-1]}')
            completion_iou.update(pcd_gt, complete_scan)
            thr_ious = completion_iou.compute()
            for v_size in thr_ious.keys():
                print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
            chamfer_distance.update(pcd_gt, complete_scan)
            cd_mean, cd_std = chamfer_distance.compute()
            print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print('\n\n=================== FINAL RESULTS ===================\n\n')
        print(f'JSD 3D: {np.array(jsd_3d).mean()}')
        print(f'JSD BEV: {np.array(jsd_bev).mean()}')
        thr_ious = completion_iou.compute()
        for v_size in thr_ious.keys():
            print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
        cd_mean, cd_std = chamfer_distance.compute()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
    else:
        os.makedirs(f'{path_scan}/results', exist_ok=True)
        for pcd_path in tqdm.tqdm(natsorted(os.listdir(path_scan))):
            if not (pcd_path.endswith(".bin") or
                    pcd_path.endswith(".ply") or
                    pcd_path.endswith(".pcd")):
                continue
            pcd_file = os.path.join(path_scan, pcd_path)
            points = load_pcd(pcd_file)
            complete_point, scan = lidar_completion.complete_scan(points)

            pcd_complete = o3d.geometry.PointCloud()
            pcd_complete.points = o3d.utility.Vector3dVector(complete_point)
            pcd_complete.estimate_normals()
            o3d.io.write_point_cloud(f'{path_scan}/results/{pcd_path.split(".")[0]}.ply', pcd_complete)

if __name__ == '__main__':
    main()
