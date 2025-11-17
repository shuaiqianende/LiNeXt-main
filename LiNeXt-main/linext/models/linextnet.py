from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional
from linext.models.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation,
    grouping_operation,
)
import time
from torch import nn, einsum
from linext.models.serial.serialization import Point
import random
import torch.nn.functional as F  
import spconv.pytorch as spconv
from typing import Tuple


@torch.no_grad()
def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """
    kNN with PyKeOps.
    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)
    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """

    import pykeops
    pykeops.set_verbose(False)

    xi = pykeops.torch.LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = pykeops.torch.LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).sqnorm2()  # (*, N, M)
    knn_d2, knn_indices = dij.Kmin_argKmin(k, dim=q_points.dim() - 1)  # (*, N, K)
    return knn_d2, knn_indices

def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    # new_xyz = torch.mean(xyz, dim=1)
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx



class MlpRes(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MlpRes, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class MlpConv(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MlpConv, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=(1, 1),
        stride=(1, 1),
        if_bn=True,
        activation_fn: Optional[Callable] = torch.relu,
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.conv(x)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class _sample_and_group_serial(nn.Module):
    def __init__(self, npoint, n_knn, in_ch, out_ch, end_k=8, pos_hidden_dim=64, use_xyz=False):
        """
        Args:
            npoint: int, number of points to sample
            k: int, number of neighbors to group
            n_knn: int, final number of neighbors after top-k selection
            pos_hidden_dim: int, hidden dimension for positional MLP
            use_xyz: bool, whether to concatenate xyz with features
        """
        super().__init__()
        self.npoint = npoint
        self.n_knn = n_knn
        self.use_xyz = use_xyz
        self.up_ksn = 2
        self.s_k = n_knn * self.up_ksn  # extended neighborhood size for initial sampling
        self.end_k = end_k
        self.in_ch = in_ch

        self.conv_up = nn.Conv1d(in_ch, out_ch, 1)

        self.serial_attn = Serial_ATTN(in_channel=in_ch, pos_channel=3, dim=in_ch, cls=0,
                                   n_knn=n_knn,attn_hidden_multiplier=2,end_k=end_k,up_ksn=2)

    def forward(self, xyz, points=None, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)
            idx: Optional[Tensor], (B, npoint, nsample)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, 3+f or f, npoint, n_knn)
            idx: Tensor, (B, npoint, n_knn)
            grouped_xyz: Tensor, (B, 3, npoint, n_knn)
        """
        B, _, N = xyz.shape

        if N != self.npoint:
            sampled_idx = furthest_point_sample(xyz.transpose(1, 2).contiguous(), self.npoint)
            sampled_idx, _ = torch.sort(sampled_idx, dim=1)  # (B, npoint)
            new_xyz = gather_operation(xyz,sampled_idx).contiguous()
            new_feat = gather_operation(points,sampled_idx).contiguous()
        else:
            # sampled_idx = torch.arange(self.npoint, device=xyz.device).int().unsqueeze(0).expand(B, -1).contiguous()
            new_xyz = xyz
            new_feat = points


        new_points = self.serial_attn(
            new_xyz, new_feat, new_feat, xyz, points
        )  # (B, 128, N_prev)
        new_points = self.conv_up(new_points)
        return new_xyz, new_points, idx


class Serialdownsampling(nn.Module):
    def __init__(
        self,
        npoint,
        nsample,
        in_channel,
        mlp,
        end_k=8,
        if_bn=True,
        group_all=False,
        use_xyz=False,
        if_idx=False,
    ):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(Serialdownsampling, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        if self.group_all == False:
            self.serial_down = _sample_and_group_serial(npoint, nsample, in_ch=in_channel, out_ch=mlp[-1],end_k=end_k)



    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)
            idx: Tensor, (B, npoint, nsample)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx = sample_and_group_all(
                xyz, points, self.use_xyz
            )
        else:
            new_xyz, new_points, idx = self.serial_down(
                xyz, points, idx
            )

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points



def serialization(pos, order="random", grid_size=0.02):
    bs, n_p, _ = pos.size()
    if order == "random":
        # options = ["z", "z-trans", "hilbert", "hilbert-trans", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        options = ["z", "z-trans", "hilbert", "hilbert-trans"]
        order = random.choice(options)

    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64)
    batch_idx = batch_idx.to(pos.device)
    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse
    return random.choice([order, inverse_order])


class Serial_ATTN(nn.Module):
    def __init__(
        self,
        in_channel,
        pos_channel,
        dim=64,
        n_knn=16,
        attn_hidden_multiplier=2,
        end_k=4,
        up_ksn=2,
        cls=0,
    ):
        super(Serial_ATTN, self).__init__()
        self.n_knn = n_knn
        self.end_k = end_k
        self.up_ksn = up_ksn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(pos_channel+cls, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, dim, 1),
        )
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)
        self.cls = cls

    def forward(self, pcd, query, value, p_part, part_feat):
        identity = value
        value = self.conv_value(value)
        key = self.conv_key(part_feat)
        query = self.conv_query(query)

        b, dim, n = value.shape

        _, idx = keops_knn(pcd.transpose(1, 2).contiguous(), p_part.transpose(1, 2).contiguous(), self.n_knn)
        idx, _ = torch.sort(idx, dim=-1)
        idx = idx.int()

        pcd_rel = pcd.reshape((b, -1, n, 1)) - grouping_operation(
            p_part, idx
        )  # b, 3, n, n_knn*4
        pos_embedding = self.pos_mlp(pcd_rel)

        key = grouping_operation(key, idx)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key + pos_embedding
        qk_rel = qk_rel.view(b, dim, n * self.end_k, -1)
        qk_rel, _ = torch.max(qk_rel, dim=-1)
        qk_rel = qk_rel.view(b, dim, n, self.end_k)

        value = value.reshape(b, -1, n, 1) - key + pos_embedding  # b, dim, n, n_knn
        value = value.view(b, dim, n * self.end_k, -1)
        value, _ = torch.max(value, dim=-1)
        value = value.view(b, dim, n, self.end_k)

        attention = self.attn_mlp(qk_rel)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)
        agg = einsum("b c i j, b c i j -> b c i", attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, hidden_dim=128, up_factor=2, radius=1.0, num_p0=4096, cls=16, kp_radius=128, grid_size=0.01):
        super(SeedGenerator, self).__init__()
        # self.spd = MulScaleSPD(dim_feat=dim_feat, up_factor=up_factor, hidden_dim=hidden_dim, num=num_p0, radius=radius)
        self.serialpd = SerialPD(dim_feat=dim_feat, up_factor=up_factor, hidden_dim=hidden_dim, num=num_p0, radius=radius, id=0)
        self.num_p0 = num_p0
        self.grid_size = grid_size
        self.dim = hidden_dim

    def forward(self, xyz, feat, global_feat):
        pcd_coarse, k_prev_corse, _ = self.serialpd(
            xyz, global_feat, feat
        )  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)
        pcd_coarse = torch.cat([pcd_coarse, xyz], dim=2)
        k_prev_corse = torch.cat([k_prev_corse, feat], dim=2)
        pcd_coarse, k_prev_corse = point_shift(pcd_coarse, k_prev_corse, self.grid_size)

        return pcd_coarse, k_prev_corse

class MSSC(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=32, out_dim=128, kernel_size=3, grid_size=None):
        super(MSSC, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if grid_size == None:
            self.grid_size = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
        else:
            self.grid_size = grid_size
        self.length = len(self.grid_size)
        subconvs = []
        for i, grid in enumerate(self.grid_size):
            subconvs.append(spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True, indice_key=None))
            subconvs.append(spconv.SubMConv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True, indice_key=None))
        self.subconvs = nn.ModuleList(subconvs)
        self.mlp_1 = nn.Linear(in_dim, hidden_dim)
        self.mlp_2 = nn.Linear(hidden_dim * self.length, out_dim)
        self.mlp_list = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.length)
        ])


    def forward(self, p, x=None):
        if x == None:
            x = p
        b, n, c = x.shape
        xyz = p
        points = x
        points = self.mlp_1(points)
        batch = torch.arange(0, b).repeat_interleave(n).view(-1, 1).to(xyz.device)
        multi_scale_outputs = []
        for i, grid_size in enumerate(self.grid_size):
            # 体素�?
            scaled_coord = xyz / grid_size
            grid_coord = torch.floor(scaled_coord).to(torch.int64)
            min_coord = grid_coord.min(dim=1, keepdim=True)[0]
            grid_coord = grid_coord - min_coord
            indices_batch = grid_coord.view(-1, 3).contiguous()  # [b * n, 3]
            spatial_shape = (grid_coord.max(dim=1)[0].max(dim=0)[0] + 1).tolist()
            points_hidden = self.mlp_list[i](points)
            features_batch = points_hidden.view(-1, self.hidden_dim).contiguous()  # [b * n, hidden_dim]
            sparse_tensor = spconv.SparseConvTensor(
                features=features_batch,
                indices=torch.cat([batch.int(), indices_batch.int()], dim=1).contiguous(),
                spatial_shape=spatial_shape,
                batch_size=b,
            )
            sparse_tensor = self.subconvs[i * 2](sparse_tensor) + sparse_tensor
            sparse_tensor = self.subconvs[i*2+1](sparse_tensor) + sparse_tensor
            multi_scale_outputs.append(sparse_tensor.features+features_batch)
        final_features = torch.cat(multi_scale_outputs, dim=1)  # 将�?�尺度特征拼�?
        final_features = final_features.view(b, n, self.hidden_dim*self.length).contiguous()
        out = self.mlp_2(final_features)

        return out

def point_shift(pcd_coarse, k_curr, grid_size, is_tran=False):
    if is_tran:
        pcd_coarse_tran = pcd_coarse
    else:
        pcd_coarse_tran = pcd_coarse.transpose(1, 2).contiguous()
    order = serialization(pcd_coarse_tran, grid_size=grid_size)
    bs, n_p, _ = pcd_coarse_tran.size()
    pcd_coarse_tran = pcd_coarse_tran.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if k_curr == None:
        return pcd_coarse_tran.transpose(1, 2).contiguous()
    else:
        if is_tran:
            k_curr_tran = k_curr
        else:
            k_curr_tran = k_curr.transpose(1, 2).contiguous()
        k_curr_tran = k_curr_tran.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        return pcd_coarse_tran.transpose(1, 2).contiguous(), k_curr_tran.transpose(1, 2).contiguous()



def find_nearest_neighbors_with_feat(
    pcd_prev: torch.Tensor,
    seed_part: torch.Tensor,
    seed_feat: torch.Tensor
):
    """
    input:
        pcd_prev: Tensor, (B, 3, N)
        seed_part: Tensor, (B, 3, M)
        seed_feat: Tensor, (B, C, M)

    output:
        nearest_idx: LongTensor, (B, N)
        nearest_points: Tensor, (B, 3, N)
        nearest_feats: Tensor, (B, C, N)
    """
    B, _, N = pcd_prev.shape
    _, C, M = seed_feat.shape
    pcd_prev_t = pcd_prev.transpose(1, 2)  # (B, N, 3)
    seed_part_t = seed_part.transpose(1, 2)  # (B, M, 3)
    dist = torch.cdist(pcd_prev_t, seed_part_t)
    nearest_idx = torch.argmin(dist, dim=2)  # (B, N)
    nearest_points = torch.gather(
        seed_part_t, dim=1,
        index=nearest_idx.unsqueeze(-1).expand(B, N, 3)
    ).transpose(1, 2)
    nearest_feats = torch.gather(
        seed_feat, dim=2,
        index=nearest_idx.unsqueeze(1).expand(B, C, N)
    )
    return nearest_idx, nearest_points, nearest_feats


class Refine_Module(nn.Module):
    def __init__(self, hidden_dim=128, up_factor=6, grid_size=0.01, beta=1):
        super(Refine_Module, self).__init__()
        self.grid_size = grid_size
        self.up_factor = up_factor
        self.beta = beta
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_1 = MlpConv(in_channel=3, layer_dims=[hidden_dim//2, hidden_dim])
        self.mlp_2 = MlpConv(in_channel=hidden_dim * 2 + 512, layer_dims=[hidden_dim])
        self.serial_attn_r = Serial_ATTN(in_channel=hidden_dim, pos_channel=3, dim=64, cls=0)
        self.mlp_ps = MlpConv(in_channel=hidden_dim, layer_dims=[hidden_dim//2, hidden_dim//4])
        self.ps = nn.ConvTranspose1d(
            hidden_dim//4, hidden_dim, up_factor, up_factor, bias=False
        )  # point-wise splitting
        self.mlp_delta_feature = MlpRes(in_dim=hidden_dim * 2, hidden_dim=hidden_dim, out_dim=hidden_dim)

        self.mlp_delta = MlpConv(in_channel=hidden_dim, layer_dims=[hidden_dim//2, 3])

    def forward(self, pcd_prev, k_prev, p_part, part_feat, seed_part, seed_feat):
        xyz_prev = pcd_prev[:, :3, :].contiguous()
        feat_1 = self.mlp_1(xyz_prev)  # (B, 128, N_prev)

        idx, pts, feats = find_nearest_neighbors_with_feat(pcd_prev, seed_part, seed_feat)

        feat_1 = torch.cat(
            [
                feat_1,
                torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                # feat_global.repeat(1, 1, feat_1.size(2)),
                feats,
            ],
            1,
        )  # (B, 128*2 + 512, N_prev)
        value = self.mlp_2(feat_1)  # (B, 128, N_prev)
        hidden = self.serial_attn_r(
            pcd_prev, k_prev , value, p_part, part_feat
        )  # (B, 128, N_prev)

        feat_child = self.mlp_ps(hidden)  # (B, 32, N_prev)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        hidden_up = self.up_sampler(hidden)  # (B, 128, N_prev * up_factor)
        k_curr = self.mlp_delta_feature(
            torch.cat([feat_child, hidden_up], 1)
        )  # (B, 128, N_prev * up_factor)

        pcd_child = self.up_sampler(pcd_prev)  # (B, 3, N_prev * up_factor)
        delta = torch.tanh(self.mlp_delta(torch.relu(k_curr))) * self.beta
        up_pcd = pcd_child + delta

        return up_pcd  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)


class SerialPD(nn.Module):
    def __init__(self, hidden_dim=128, grid_size=0.01):
        """Snowflake Point Deconvolution"""
        super(SerialPD, self).__init__()
        self.grid_size = grid_size
        self.mlp_1 = MlpConv(in_channel=3, layer_dims=[hidden_dim//2, hidden_dim])
        self.mlp_2 = MlpConv(in_channel=hidden_dim * 2 + 512, layer_dims=[hidden_dim * 2, hidden_dim])

        self.serial_attn = Serial_ATTN(in_channel=hidden_dim, pos_channel=3, dim=64, cls=0)
        self.mlp_noise = MlpRes(in_dim=hidden_dim, hidden_dim=hidden_dim//2, out_dim=3)

    def forward(self, pcd_prev, k_prev, p_part, part_feat, seed_part, seed_feat):
        xyz_prev = pcd_prev[:, :3, :].contiguous()
        feat_1 = self.mlp_1(xyz_prev)  # (B, 128, N_prev)

        idx, pts, feats = find_nearest_neighbors_with_feat(pcd_prev, seed_part, seed_feat)

        feat_1 = torch.cat(
            [
                feat_1,
                torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                # feat_global.repeat(1, 1, feat_1.size(2)),
                feats,
            ],
            1,
        )  # (B, 128*2 + 512, N_prev)
        value = self.mlp_2(feat_1)  # (B, 128, N_prev)


        hidden = self.serial_attn(
            pcd_prev, k_prev if k_prev is not None else value, value, p_part, part_feat
        )  # (B, 128, N_prev)

        noise_pre = self.mlp_noise(hidden)  # (B, 128, N_prev * up_factor)


        return noise_pre, hidden  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)


class Get_Kprev(nn.Module):
    def __init__(self, p_num, grid_size=0.01):
        """Encoder that encodes information of partial point cloud"""
        super(Get_Kprev, self).__init__()
        self.grid_size = grid_size

        self.sa_module_0 = Serialdownsampling(
            p_num, 8, 64, [64], end_k=4, group_all=False, if_bn=False, if_idx=True,
        )
        self.sa_module_1 = Serialdownsampling(
            2048, 16, 64, [128], end_k=4, group_all=False, if_bn=False, if_idx=True,
        )
        self.sa_module_2 = Serialdownsampling(
            512, 16, 128, [128, 256], end_k=4, group_all=False, if_bn=False, if_idx=True,
        )
        self.sa_module_3 = Serialdownsampling(
            128, 16, 256, [256, 512], end_k=4, group_all=False, if_bn=False, if_idx=True
        )
        self.subconv = MSSC(in_dim=3, hidden_dim=32, out_dim=64, kernel_size=3, grid_size=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28])

    def forward(self, pcd):
        # xyz = pcd
        center = pcd
        points = self.subconv(center)
        if not torch.all(center == 0):
            order = serialization(center, grid_size=self.grid_size)
            bs, n_p, _ = center.size()
            center = center.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
            points = points.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

        center = center.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1).contiguous()
        l0_xyz, l0_points, idx0 = self.sa_module_0(center, points)  # (B, 3, 512), (B, 128, 512)
        l0_xyz, l0_points = point_shift(l0_xyz, l0_points, self.grid_size*1)
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_xyz, l1_points = point_shift(l1_xyz, l1_points, self.grid_size*2)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_xyz, l2_points = point_shift(l2_xyz, l2_points, self.grid_size*4)
        l3_xyz, l3_points, idx3 = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 128), (B, 256, 512)
        # _, global_feat = self.sa_module_4(l3_xyz, l3_points)  # (B, 3, 1), (B, out_dim, 1)

        return l0_xyz.transpose(1, 2).contiguous(), l0_points.transpose(1, 2).contiguous(), l3_xyz, l3_points



class LiNeXt_N2C(nn.Module):
    def __init__(
        self,
        num_pin=18000,
        num_pfull=180000,
        hidden_dim=64,
    ):
        super(LiNeXt_N2C, self).__init__()
        self.cls = 0
        self.num_pfull = num_pfull
        self.num_pin = num_pin
        self.noise_predict = SerialPD(hidden_dim=hidden_dim, grid_size=0.01)
        self.get_partial_feat = Get_Kprev(num_pin, grid_size=0.05)
        self.full_conv = MSSC(in_dim=3, hidden_dim=32, out_dim=hidden_dim, kernel_size=3, grid_size=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28])

        self.embed_dim = hidden_dim


    def forward(self, p_full, partial, full_feat=None):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        p_part, part_feat, seed_part, seed_feat = self.get_partial_feat(partial)
        if full_feat == None:
            full_feat = self.full_conv(p_full)

        p_full, full_feat = point_shift(p_full, full_feat, 0.01, is_tran=True)
        p_part, part_feat = point_shift(p_part, part_feat, 0.02, is_tran=True)
        noise, hidden_f = self.noise_predict(p_full, full_feat, p_part, part_feat, seed_part, seed_feat)
        p_full = p_full - noise
        p_full = p_full.transpose(1, 2).contiguous()
        return {
            "p_full": p_full,
            "noise": noise,
            "hidden_f": hidden_f,
            "p_part": p_part,
            "part_feat": part_feat,
            "seed_part": seed_part,
            "seed_feat": seed_feat
        }



class LiNeXt_Refine(nn.Module):
    def __init__(
        self,
        num_pin=180000,
        num_pfull=1080000,
        hidden_dim=64,
        beta=1,
    ):
        super(LiNeXt_Refine, self).__init__()
        self.cls = 0
        self.num_pfull = num_pfull
        self.num_pin = num_pin
        self.refine = Refine_Module(up_factor=6, hidden_dim=hidden_dim, grid_size=0.01, beta=beta)
        self.embed_dim = hidden_dim

    def forward(self, p_full, full_feat, p_part, part_feat, seed_part, seed_feat):
        p_full = p_full.transpose(1, 2).contiguous()
        full_feat = full_feat.transpose(1, 2).contiguous()
        order = serialization(p_full, grid_size=0.01)
        bs, n_p, _ = p_full.size()
        p_full = p_full.flatten(0, 1)[order].reshape(bs, n_p, -1).transpose(1, 2).contiguous()
        full_feat = full_feat.flatten(0, 1)[order].reshape(bs, n_p, -1).transpose(1, 2).contiguous()
        p_full = self.refine(p_full, full_feat, p_part, part_feat, seed_part, seed_feat)

        return p_full.transpose(1, 2).contiguous()

