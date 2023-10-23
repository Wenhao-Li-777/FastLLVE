import cv2

import os.path as osp
import logging
import time
import argparse

import torch.nn as nn
import options.option as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    model = create_model(opt)

    print('mkdir finish')

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}
        psnr_rlt_all = []
        ssim_rlt = {}
        ssim_rlt_all = []

        for val_data in val_loader:
            folder = val_data['folder'][0]

            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []
            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []

            model.feed_data(val_data)
            model.test()

            visuals = model.get_current_visuals()

            rlts = visuals['rlts'].permute(1, 0, 2, 3).contiguous()
            GTs = visuals['GTs'].permute(1, 0, 2, 3).contiguous()
            n, _, _, _ = rlts.shape

            for frames_number in range(n):
                rlt_img = util.tensor2img(rlts[frames_number])
                gt_img = util.tensor2img(GTs[frames_number])

                psnr = util.calculate_psnr(rlt_img, gt_img)
                psnr_rlt[folder].append(psnr)

                ssim = util.calculate_ssim(rlt_img, gt_img)
                ssim_rlt[folder].append(ssim)

            pbar.update('Test {}'.format(folder))

        for k, v in psnr_rlt.items():
            if opt['datasets']['if_mod'] == 0:
                if '_5' in k:
                    psnr_rlt_all.append(v[5])
                    psnr_rlt_all.append(v[6])
                else:
                    for frame_psnr in v:
                        psnr_rlt_all.append(frame_psnr)
            else:
                for frame_psnr in v:
                    psnr_rlt_all.append(frame_psnr)

        for k, v in ssim_rlt.items():
            if opt['datasets']['if_mod'] == 0:
                if '_5' in k:
                    ssim_rlt_all.append(v[5])
                    ssim_rlt_all.append(v[6])
                else:
                    for frame_ssim in v:
                        ssim_rlt_all.append(frame_ssim)
            else:
                for frame_ssim in v:
                    ssim_rlt_all.append(frame_ssim)

        psnr_total_avg = sum(psnr_rlt_all) / len(psnr_rlt_all)
        ssim_total_avg = sum(ssim_rlt_all) / len(ssim_rlt_all)
        print('PSNR = ', psnr_total_avg)
        print('SSIM = ', ssim_total_avg)


if __name__ == '__main__':
    main()
