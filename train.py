import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.option as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to opt YAML file.', default='./options/train/train_in_sdsd.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # opt loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='./tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            print('total_epochs = ', total_epochs)
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['dist']:
                    # multi-GPU testing
                    psnr_rlt = {}  # with border and center frames
                    ssim_rlt = {}

                    dist_val_set_len = (len(val_set) // world_size) * world_size
                    for idx in range(rank, dist_val_set_len, world_size):
                        val_data = val_set[idx]
                        val_data['LQs'].unsqueeze_(0)
                        val_data['GTs'].unsqueeze_(0)
                        folder = val_data['folder']
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = torch.zeros(opt['datasets']['val']['N_frames'], dtype=torch.float32, device='cuda')
                        if ssim_rlt.get(folder, None) is None:
                            ssim_rlt[folder] = torch.zeros(opt['datasets']['val']['N_frames'], dtype=torch.float32, device='cuda')
                        model.feed_data(val_data)
                        model.test()
                        visuals = model.get_current_visuals()
                        rlts = visuals['clear_rlts'].permute(1, 0, 2, 3).contiguous()
                        GTs = visuals['GTs'].permute(1, 0, 2, 3).contiguous()
                        n, _, _, _ = rlts.shape
                        for frames_number in range(n):
                            rlt_img = util.tensor2img(rlts[frames_number])
                            gt_img = util.tensor2img(GTs[frames_number])

                            psnr_rlt[folder][frames_number] = util.calculate_psnr(rlt_img, gt_img)
                            ssim_rlt[folder][frames_number] = util.calculate_ssim(rlt_img, gt_img)

                    # collect data
                    for _, v in psnr_rlt.items():
                        dist.reduce(v, 0)
                    for _, v in ssim_rlt.items():
                        dist.reduce(v, 0)
                    dist.barrier()

                    if rank == 0:
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        ssim_rlt_avg = {}
                        ssim_total_avg = 0.
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                            psnr_total_avg += psnr_rlt_avg[k]
                        for k, v in ssim_rlt.items():
                            ssim_rlt_avg[k] = sum(v) / len(v)
                            ssim_total_avg += ssim_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        ssim_total_avg /= len(ssim_rlt)
                        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                        logger.info(log_s)
                        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
                            tb_logger.add_scalar('ssim_avg', ssim_total_avg, current_step)
                            for k, v in ssim_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
                else:
                    pbar = util.ProgressBar(len(val_loader))
                    psnr_rlt = {}  # with border and center frames
                    ssim_rlt = {}
                    psnr_rlt_all = []
                    ssim_rlt_all = []
                    for val_data in val_loader:
                        folder = val_data['folder'][0]
                        # border = val_data['border'].item()
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = []
                        if ssim_rlt.get(folder, None) is None:
                            ssim_rlt[folder] = []

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        rlts = visuals['clear_rlts'].permute(1, 0, 2, 3).contiguous()
                        GTs = visuals['GTs'].permute(1, 0, 2, 3).contiguous()
                        n, _, _, _ = rlts.shape

                        # calculate PSNR
                        for frames_number in range(n):
                            rlt_img = util.tensor2img(rlts[frames_number])
                            gt_img = util.tensor2img(GTs[frames_number])

                            psnr = util.calculate_psnr(rlt_img, gt_img)
                            psnr_rlt[folder].append(psnr)
                            ssim = util.calculate_ssim(rlt_img, gt_img)
                            ssim_rlt[folder].append(ssim)

                        pbar.update('Test {}'.format(folder))
                    for k, v in psnr_rlt.items():
                        if opt['datasets']['val']['if_mod'] == 0:
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
                        if opt['datasets']['val']['if_mod'] == 0:
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
                    log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                    logger.info(log_s)
                    log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                    logger.info(log_s)
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                        tb_logger.add_scalar('ssim_avg', ssim_total_avg, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()
