import torch
import linext.models.linextnet as linextnet

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from linext.utils.collations import *
from linext.utils.metrics import ChamferDistance, PrecisionRecall

from linext.models.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()


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

def chamfer_sqrt(p1, p2):
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2, idx1


class RefineModule(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
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
        try:
            self.n2c_path = self.hparams['model']['n2c_ckpt']
        except (KeyError, TypeError):
            self.n2c_path = None

        n2c_ckpt = torch.load(self.n2c_path)
        self.save_hyperparameters(n2c_ckpt['hyper_parameters'])
        self.model = linextnet.LiNeXt_N2C()
        self.load_state_dict(n2c_ckpt['state_dict'], strict=False)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.model_refine = linextnet.LiNeXt_Refine()

        self.chamfer_distance = chamfer_sqrt
        self.precision_recall = PrecisionRecall(0.001,0.01,100)

    def forward_refine(self, x_full, x_part):
        output = self.model(x_full, x_part)
        x_full = output["p_full"]
        full_hidden = output["hidden_f"]
        x_part = output["p_part"]
        part_feat = output["part_feat"]
        seed_part = output["seed_part"]
        seed_feat = output["seed_feat"]

        x_full = x_full.transpose(1, 2).detach()
        feat_full = full_hidden.detach()
        x_part = x_part.detach()
        part_feat = part_feat.detach()
        seed_part = seed_part.detach()
        seed_feat = seed_feat.detach()
        return self.model_refine(x_full, feat_full, x_part, part_feat, seed_part, seed_feat)

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        b_shape = batch['pcd_full'].shape
        pre_shape = (b_shape[0], b_shape[1], b_shape[2])
        noise = torch.randn(pre_shape, device=self.device) * self.alpha_noise

        input_part = repeat_by_distance_quartiles(batch['pcd_part'], repeats=self.repeats) + noise
        x_full = input_part
        x_part = batch['pcd_part']

        p_full = self.forward_refine(x_full, x_part)
        loss, _ = self.chamfer_distance(p_full, torch.tensor(batch['pcd_full']))
        self.log('train/cd_loss', loss)
        # torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        return

    def test_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_refine.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))

        return optimizer

#######################################
# Modules
#######################################
