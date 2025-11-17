import torch
import linext.models.linextnet as linextnet

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from linext.utils.collations import *
from linext.utils.metrics import ChamferDistance, PrecisionRecall
from linext.models.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_dist = chamfer_3DDist()

def chamfer_sqrt(p1, p2):
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2, idx1

def chamfer_sqrt2(p1, p2):
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(torch.sqrt(d1)))
    d2 = torch.mean(torch.sqrt(torch.sqrt(d2)))
    return (d1 + d2) / 2, idx1


def chamfer(p1, p2):
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2), idx1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def repeat_by_distance_quartiles(pcd_part, repeats=(5, 8, 12, 15)):
    """
    Replicate points within each sample according to quartiles of squared
    Euclidean distance to the origin.

    Args:
        pcd_part (torch.Tensor): [B, P, C] point cloud tensor.
        repeats (tuple of 4 ints): repeat factors for the four quartiles.

    Returns:
        torch.Tensor: [B, P_repeated, C] tensor after replication.
    """
    B, P, C = pcd_part.shape
    device = pcd_part.device

    # 1) Compute squared distances and sort indices
    dist2 = (pcd_part ** 2).sum(dim=-1)                 # [B, P]
    _, idx_sort = torch.sort(dist2, dim=-1)             # [B, P]

    # 2) Sort points by distance
    idx_sort = idx_sort.unsqueeze(-1).expand_as(pcd_part)
    pcd_sorted = torch.gather(pcd_part, 1, idx_sort)    # [B, P, C]

    # 3) Split into quartiles
    q = P // 4
    sizes = [q, q, q, P - 3 * q]
    chunks = torch.split(pcd_sorted, sizes, dim=1)      # list of [B, size_i, C]

    # 4) Repeat each chunk as specified
    repeated_chunks = [
        chunk.repeat_interleave(rep, dim=1) for chunk, rep in zip(chunks, repeats)
    ]

    # 5) Concatenate along the point dimension
    pcd_repeated = torch.cat(repeated_chunks, dim=1)    # [B, ?, C]
    return pcd_repeated

class DenoisePoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        try:
            self.alpha_noise = self.hparams['model']['alpha_noise']
        except (KeyError, TypeError):
            self.alpha_noise = 1
        try:
            self.repeats = self.hparams['model']['repeats']
        except (KeyError, TypeError):
            self.repeats = (5, 8, 12, 15)

        self.model = linextnet.LiNeXt_N2C()

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.cd_ls = chamfer_sqrt
        self.cnt = 0


    def forward(self, x_full, x_part):
        output = self.model(x_full, x_part)
        p_full = output["p_full"]
        torch.cuda.empty_cache()
        return p_full


    def training_step(self, batch:dict, batch_idx):
        torch.cuda.empty_cache()
        b_shape = batch['pcd_full'].shape
        pre_shape = b_shape
        noise = torch.randn(pre_shape, device=self.device) * self.alpha_noise


        gt_end = batch['pcd_full']


        input_part = repeat_by_distance_quartiles(batch['pcd_part'], repeats=self.repeats) + noise
        x_full = input_part
        x_part = batch['pcd_part']


        p_full = self.forward(x_full, x_part)

        loss_cd, _ = self.cd_ls(gt_end, p_full)

        loss = loss_cd

        self.log('train/loss_cd', loss_cd)
        self.log('train/loss', loss)
        self.cnt = self.cnt + 1
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch:dict, batch_idx):
        return

    def test_step(self, batch:dict, batch_idx):
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]

#######################################
# Modules
#######################################
