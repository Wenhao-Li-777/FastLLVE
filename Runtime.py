import cv2

import os.path as osp
import logging
import time
import argparse

import options.option as opt
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import os

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-conf', type=str, required=True, help='Path to options YMAL file.')
conf = opt.parse(parser.parse_args().conf, is_train=False)
conf = opt.dict_to_nonedict(conf)


def main():
    model = create_model(conf)

    for phase, dataset_conf in conf['datasets'].items():
        val_set = create_dataset(dataset_conf)
        val_loader = create_dataloader(val_set, dataset_conf, conf, None)

        for val_data in val_loader:
            start = time.time()

            for count in range(100):
                model.feed_data(val_data, need_GT=False)
                model.test()

            end = time.time()
            runtime = (end - start) / 700
            print('runtime = ', runtime)


if __name__ == '__main__':
    main()