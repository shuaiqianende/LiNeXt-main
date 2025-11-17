import torch
from torch.utils.data import Dataset
from linext.utils.pcd_preprocess import clusterize_pcd, visualize_pcd_clusters, point_set_to_coord_feats, overlap_clusters, aggregate_pcds
from linext.utils.pcd_transforms import *
from linext.utils.data_map import learning_map
from linext.utils.collations import point_set_to_sparse_refine
import os
import numpy as np
import MinkowskiEngine as ME
from natsort import natsorted

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################


def point_set_to_refine(p_full, p_part, n_full, n_part, resolution, filename):
    p_part = torch.tensor(p_part)
    p_full = torch.tensor(p_full)
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0)
    p_std = p_full.std(axis=0)
    return [p_full, p_mean, p_std, p_part, filename]

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, scan_window, seqs, split, resolution, num_points, mode):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = 'segments_views'

        self.n_clusters = 50
        self.resolution = resolution
        self.scan_window = scan_window
        self.num_points = num_points
        self.seg_batch = True

        self.split = split
        self.seqs = seqs
        self.mode = mode

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()

        self.nr_data = len(self.points_datapath)
        self.input_path = ''

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def _datapath_list(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            for file_num in range(0, len(point_seq_bin)):
                # we guarantee that the end of sequence will not generate single scans as aggregated pcds
                end_file = file_num + self.scan_window if len(point_seq_bin) - file_num > 1.5 * self.scan_window else len(point_seq_bin)
                self.points_datapath.append([os.path.join(point_seq_path, point_file) for point_file in point_seq_bin[file_num:end_file] ])
                if end_file == len(point_seq_bin):
                    break

    def datapath_list(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, seq)
            point_seq_gt = natsorted(os.listdir(os.path.join(point_seq_path, 'gt')))
            for file_num in range(0, len(point_seq_gt)):
                self.points_datapath.append(os.path.join(point_seq_path, 'gt', point_seq_gt[file_num]))
        #self.points_datapath = self.points_datapath[:200]

    def transforms(self, points):
        points = points[None,...]

        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return points[0]

    def __getitem__(self, index):
        p_part = np.load(self.points_datapath[index].replace('gt', 'input'))
        if self.split != 'test':
            pass
            p_full = np.load(self.points_datapath[index])
            # trans = pose[:-1,-1]
            # dist_full = np.sum((p_map - trans)**2, -1)**.5
            # p_full = p_map[dist_full < self.max_range]
            # p_full = np.concatenate((p_full, np.ones((len(p_full),1))), axis=-1)
            # p_full = (p_full @ np.linalg.inv(pose).T)[:,:3]
            # p_full = p_full[p_full[:,2] > -4.]
        else:
            p_full = p_part

        p_part = p_part.reshape((-1,3))
        p_full = p_full.reshape((-1,3))
        if self.split == 'train':
            p_concat = np.concatenate((p_full, p_part), axis=0)
            p_concat = self.transforms(p_concat)

            p_full = p_concat[:-len(p_part)]
            p_part = p_concat[-len(p_part):]

        # patial pcd has 1/10 of the complete pcd size
        n_part = int(self.num_points / 10.)

        return point_set_to_refine(
            p_full,
            p_part,
            self.num_points,
            n_part,
            self.resolution,
            self.points_datapath[index],
        )                                       

    def __len__(self):
        #print('DATA SIZE: ', np.floor(self.nr_data / self.sampling_window), self.nr_data % self.sampling_window)
        return self.nr_data

##################################################################################################
