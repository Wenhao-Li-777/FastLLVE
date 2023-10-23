import cv2

import os.path as osp
import logging
import time
import argparse

import options.option as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import os

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    model = create_model(opt)
    save_folder = './results/{}'.format(opt['name'])
    output_folder = osp.join(save_folder, 'images/output')
    input_folder = osp.join(save_folder, 'images/input')
    GT_folder = osp.join(save_folder, 'images/GT')
    util.mkdirs(save_folder)
    util.mkdirs(output_folder)
    util.mkdirs(input_folder)
    util.mkdirs(GT_folder)

    print('mkdir finish')

    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))

        for val_data in val_loader:
            folder = val_data['folder'][0]

            model.feed_data(val_data)

            model.test()

            visuals = model.get_current_visuals()
            LQs = visuals['LQs'].permute(1, 0, 2, 3).contiguous()
            rlts = visuals['rlts'].permute(1, 0, 2, 3).contiguous()
            GTs = visuals['GTs'].permute(1, 0, 2, 3).contiguous()
            n, _, _, _ = rlts.shape

            for frames_number in range(n):
                tag = '{}'.format(val_data['folder'])
                print(osp.join(output_folder, '{}.png'.format(tag + '_' + str(frames_number))))
                input_img = util.tensor2img(LQs[frames_number])
                rlt_img = util.tensor2img(rlts[frames_number])
                GT_img = util.tensor2img(GTs[frames_number])
                cv2.imwrite(osp.join(output_folder, '{}-{}.png'.format(tag, frames_number)), rlt_img)
                cv2.imwrite(osp.join(input_folder, '{}-{}.png'.format(tag, frames_number)), input_img)
                cv2.imwrite(osp.join(GT_folder, '{}-{}.png'.format(tag, frames_number)), GT_img)

            pbar.update('Test {}'.format(folder))


if __name__ == '__main__':
    main()
