import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, conf):
        super(VideoBaseModel, self).__init__(conf)

        if conf['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_conf = conf['train']
        self.lut_dim = conf['network_G']['n_vertices_4d']

        # define network and load pretrained models
        self.netG = networks.define_G(conf).to(self.device)
        if conf['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # load pretrained model
        self.load_path_G = conf['path']['pretrain_model_G']
        if conf['path']['strict_load'] == None:
            self.strict_load = True
        else:
            self.strict_load = False
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            self.loss_type = train_conf['pixel_criterion']
            if self.loss_type == 'l1':
                self.loss_recons = nn.L1Loss(reduction='mean').to(self.device)
            elif self.loss_type == 'l2':
                self.loss_recons = nn.MSELoss(reduction='mean').to(self.device)
            elif self.loss_type == 'Charbonnier':
                self.loss_recons = CharbonnierLoss().to(self.device)
            elif self.loss_type == 'smooth_l1':
                self.loss_recons = nn.SmoothL1Loss(reduction='mean', beta=1 / 9).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(self.loss_type))
            self.sparse_factor = train_conf['sparse_factor']
            
            #### optimizers
            wd_G = train_conf['weight_decay_G'] if train_conf['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            'Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_conf['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_conf['beta1'], train_conf['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_conf['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_conf['lr_steps'],
                                                         restarts=train_conf['restarts'],
                                                         weights=train_conf['restart_weights'],
                                                         gamma=train_conf['lr_gamma'],
                                                         clear_state=train_conf['clear_state']))
            elif train_conf['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_conf['T_period'], eta_min=train_conf['eta_min'],
                            restarts=train_conf['restarts'], weights=train_conf['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GTs'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        self.fake_H, self.fake_H_denoise, self.weights, self.reg_smoothness, self.reg_monotonicity = self.netG(self.var_L)

        ## loss
        loss = 0.
        if self.sparse_factor > 0 and self.weights is not None:
            loss_sparse = self.sparse_factor * torch.mean(self.weights.pow(2))
            loss += loss_sparse
        loss += self.loss_recons(self.fake_H, self.real_H) + self.reg_smoothness + self.reg_monotonicity
        loss += self.loss_recons(self.fake_H_denoise, self.real_H)

        loss.backward()
        self.optimizer_G.step()
        self.log_dict['loss'] = loss.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            _, self.fake_H_denoise, _, _, _ = self.netG(self.var_L, if_train=False)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQs'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlts'] = self.fake_H_denoise.detach()[0].float().cpu()
        if need_GT:
            out_dict['GTs'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def load(self):
        load_path_G = self.load_path_G
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.strict_load)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)