import torch
import models.IALUT.IALUT_model as enhance_model

def define_G(conf):
    conf_net = conf['network_G']
    which_model = conf_net['which_model_G']

    if which_model == 'IALUT_LLVE':
        if conf['datasets'].get('train', None):
            train_resolution = [conf['datasets']['train']['LQ_size'], conf['datasets']['train']['LQ_size']]
            netG = enhance_model.IALUT_LLVE(input_resolution=conf_net['input_resolution'],
                                              train_resolution=train_resolution,
                                              n_ranks=conf_net['n_ranks'],
                                              n_vertices_4d=conf_net['n_vertices_4d'],
                                              n_base_feats=conf_net['n_base_feats'],
                                              smooth_factor=conf_net['smooth_factor'],
                                              monotonicity_factor=conf_net['monotonicity_factor'])
        else:
            netG = enhance_model.IALUT_LLVE(input_resolution=conf_net['input_resolution'],
                                              train_resolution=conf_net['input_resolution'],
                                              n_ranks=conf_net['n_ranks'],
                                              n_vertices_4d=conf_net['n_vertices_4d'],
                                              n_base_feats=conf_net['n_base_feats'],
                                              smooth_factor=conf_net['smooth_factor'],
                                              monotonicity_factor=conf_net['monotonicity_factor'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG