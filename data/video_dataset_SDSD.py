import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os


class Video_Dataset_SDSD(data.Dataset):

    def __init__(self, config):
        super(Video_Dataset_SDSD, self).__init__()
        self.config = config
        self.cache_data = config['cache_data']
        # path_to_dataset/indoor_np/GT
        self.GT_root, self.LQ_root = config['dataroot_GT'], config['dataroot_LQ']
        self.data_type = self.config['data_type']
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        self.folders = []
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = [], []

        # read data:
        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_GT)

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)

            max_idx = len(img_paths_LQ) # default 7
            assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
            self.folders.append(subfolder_name)

            if self.cache_data:
                self.imgs_LQ.append(img_paths_LQ)
                self.imgs_GT.append(img_paths_GT)

    def __getitem__(self, index):
        imgs_LQ_paths = self.imgs_LQ[index]
        imgs_GT_paths = self.imgs_GT[index]

        imgs_LQ = util.read_img_seq2(imgs_LQ_paths, self.config['train_size'])
        imgs_GT = util.read_img_seq2(imgs_GT_paths, self.config['train_size'])
        img_GT_example = imgs_GT[0]

        img_LQ_l = list(imgs_LQ.unbind(0))
        img_GT_l = list(imgs_GT.unbind(0))

        if self.config['phase'] == 'train':
            GT_size = self.config['GT_size']

            _, H, W = img_GT_example.shape  # real img size

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ_l = [v[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size] for v in img_LQ_l]
            img_GT_l = [v[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size] for v in img_GT_l]

            # augmentation - flip, rotate
            img_LQ_l.extend(img_GT_l)
            rlt = util.augment_torch(img_LQ_l, self.config['use_flip'], self.config['use_rot'])
            img_LQ_l = rlt[0:self.config['N_frames']]
            img_GT_l = rlt[self.config['N_frames']:]

        return {
            'LQs': torch.stack(img_LQ_l).permute(1, 0, 2, 3).contiguous(),  # shape: [C, N, H, W]
            'GTs': torch.stack(img_GT_l).permute(1, 0, 2, 3).contiguous(),  # shape: [C, N, H, W]
            'folder': self.folders[index],
        }

    def __len__(self):
        return len(self.folders)
