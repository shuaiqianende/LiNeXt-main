import numpy as np
import open3d as o3d
from scipy.spatial.distance import jensenshannon
from linext.utils.metrics import ChamferDistance, PrecisionRecall
import matplotlib.pyplot as plt
import torch

def histogram_point_cloud_torch(pcd, resolution, max_range, bev=False):
    """
    计算点云的三维直方图（使用 torch 实现）。

    参数：
        pcd: torch.Tensor，形状 (N, 3)，点云数据。
        resolution: float，每个体素的边长。
        max_range: float，在 [-max_range, max_range] 范围内构建直方图。
        bev: bool，如果为 True，则返回二值化的直方图（每个体素大于 0 则置为 1）。

    返回：
        hist: torch.Tensor，形状 (bins, bins, bins) 的直方图。
    """
    bins = int(2 * max_range / resolution)
    delta_bin = int(bins - 2 * max_range)
    # 将点云平移到 [0, 2*max_range] 区间，并根据体素大小量化为离散索引
    indices = torch.floor((pcd + max_range) / resolution).long()

    # 筛选出完全落在 [0, bins) 范围内的点
    valid_mask = (indices >= 0+delta_bin//2) & (indices < bins-delta_bin//2)
    valid_mask = valid_mask.all(dim=1)
    indices = indices[valid_mask]

    # 初始化直方图
    hist = torch.zeros((bins, bins, bins), dtype=torch.float32, device=pcd.device)

    if indices.shape[0] > 0:
        # 将三维索引转换为一维索引
        flat_indices = indices[:, 0] * (bins * bins) + indices[:, 1] * bins + indices[:, 2]
        flat_hist = hist.view(-1)
        ones = torch.ones(flat_indices.size(0), dtype=flat_hist.dtype, device=pcd.device)
        flat_hist.index_add_(0, flat_indices, ones)

    if bev:
        hist = (hist > 0).float()

    return hist


def compute_jsd_torch(P, Q, bev):
    """
    计算两个直方图 P 和 Q 的 Jensen-Shannon Divergence（JSD），使用 torch 实现。

    参数：
        P, Q: torch.Tensor，形状相同的直方图。
        bev: bool，目前未做特殊处理，可用于扩展逻辑。

    返回：
        jsd: torch.Tensor，一个标量，表示 JSD 值。
    """
    eps = 1e-8
    P_norm = P / (P.sum() + eps)
    Q_norm = Q / (Q.sum() + eps)
    M = (P_norm + Q_norm) / 2.0
    jsd = 0.5 * (P_norm * (torch.log(P_norm + eps) - torch.log(M + eps))).sum() \
          + 0.5 * (Q_norm * (torch.log(Q_norm + eps) - torch.log(M + eps))).sum()
    return jsd


def compute_hist_metrics_torch(pcd_gt, pcd_pred, bev=False):
    """
    根据 ground truth 点云与预测点云，构造它们的三维直方图，
    并用 Jensen-Shannon Divergence 作为指标进行衡量（均基于 torch 实现）。

    参数：
        pcd_gt: torch.Tensor，ground truth 点云，形状 (N, 3)。
        pcd_pred: torch.Tensor，预测点云，形状 (M, 3)。
        bev: bool，如果为 True，则直方图二值化（只考虑占用情况）。

    返回：
        JSD 值，一个 torch.Tensor 标量（如果需要可以调用 .item() 得到数值）。
    """
    hist_pred = histogram_point_cloud_torch(pcd_pred, resolution=0.5, max_range=50.0, bev=bev)
    hist_gt = histogram_point_cloud_torch(pcd_gt, resolution=0.5, max_range=50.0, bev=bev)

    return compute_jsd_torch(hist_gt, hist_pred, bev)

def histogram_point_cloud(pcd, resolution, max_range, bev=False):
    # get bins size by the number of voxels in the pcd
    bins = int(2 * max_range / resolution)

    hist = np.histogramdd(pcd, bins=bins, range=([-max_range,max_range],[-max_range,max_range],[-max_range,max_range]))

    return np.clip(hist[0], a_min=0., a_max=1.) if bev else hist[0]

def compute_jsd(hist_gt, hist_pred, bev=False, visualize=False):
    bev_gt = hist_gt.sum(-1) if bev else hist_gt
    norm_bev_gt = bev_gt / bev_gt.sum()
    norm_bev_gt = norm_bev_gt.flatten()

    bev_pred = hist_pred.sum(-1) if bev else hist_pred
    norm_bev_pred = bev_pred / bev_pred.sum()
    norm_bev_pred = norm_bev_pred.flatten()
    
    if visualize:
        # for visualization purposes
        grid = np.meshgrid(np.arange(len(hist_gt)), np.arange(len(hist_gt)))
        points = np.concatenate((grid[0].flatten()[:,None], grid[1].flatten()[:,None]), axis=-1)
        points = np.concatenate((points, np.zeros((len(points),1))),axis=-1)

        # build bev histogram gt view
        norm_hist_gt = bev_gt / bev_gt.max()
        colors_gt = plt.get_cmap('viridis')(norm_hist_gt)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt.reshape(-1,4)[:,:3])

        # build bev histogram pred view
        norm_hist_pred = bev_pred / bev_pred.max()
        colors_pred = plt.get_cmap('viridis')(norm_hist_pred)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred.reshape(-1,4)[:,:3])

    return jensenshannon(norm_bev_gt, norm_bev_pred)


def compute_hist_metrics(pcd_gt, pcd_pred, bev=False):
    hist_pred = histogram_point_cloud(np.array(pcd_pred.points), 0.5, 50., bev)
    hist_gt = histogram_point_cloud(np.array(pcd_gt.points), 0.5, 50., bev)
    
    return compute_jsd(hist_gt, hist_pred, bev)

def compute_chamfer(pcd_pred, pcd_gt):
    chamfer_distance = ChamferDistance()
    chamfer_distance.update(pcd_gt, pcd_pred)
    cd_pred_mean, cd_pred_std = chamfer_distance.compute()

    return cd_pred_mean

def compute_precision_recall(pcd_pred, pcd_gt):
    precision_recall = PrecisionRecall(0.05,2*0.05,100)
    precision_recall.update(pcd_gt, pcd_pred)
    pr, re, f1 = precision_recall.compute_auc()

    return pr, re, f1 

def preprocess_pcd(pcd):
    points = np.array(pcd.points)
    dist = np.sqrt(np.sum(points**2, axis=-1))
    pcd.points = o3d.utility.Vector3dVector(points[dist < 30.])

    return pcd

def compute_metrics(pred_path, gt_path):
    pcd_pred = preprocess_pcd(o3d.io.read_point_cloud(pred_path))
    points_pred = np.array(pcd_pred.points)
    pcd_gt = preprocess_pcd(o3d.io.read_point_cloud(gt_path))
    points_gt = np.array(pcd_gt.points)

    jsd_pred = compute_hist_metrics(points_pred, points_gt)

    cd_pred = compute_chamfer(pcd_pred, pcd_gt)

    pr_pred, re_pred, f1_pred = compute_precision_recall(pcd_pred, pcd_gt)

    return cd_pred, pr_pred, re_pred, f1_pred

