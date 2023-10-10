import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(config_path, is_train=True):
    with open(config_path, mode='r') as f:
        config = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in config['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    config['is_train'] = is_train
    if config['distortion'] == 'sr':
        scale = config['scale']

    # datasets
    for phase, dataset in config['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if config['distortion'] == 'sr':
            dataset['scale'] = scale
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')
    
    # path
    for key, path in config['path'].items():
        if path and key in config['path'] and key != 'strict_load':
            config['path'][key] = osp.expanduser(path)
    # find root path
    if config['path']['root'] is None:
        config['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(config['path']['root'], 'experiments', config['name'])
        config['path']['experiments_root'] = experiments_root
        config['path']['models'] = osp.join(experiments_root, 'models')
        config['path']['training_state'] = osp.join(experiments_root, 'training_state')
        config['path']['log'] = experiments_root
        config['path']['val_images'] = osp.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in config['name']:
            config['train']['val_freq'] = 8
            config['logger']['print_freq'] = 1
            config['logger']['save_checkpoint_freq'] = 8
    else: # test
        results_root = osp.join(config['path']['root'], 'results', config['name'])
        config['path']['results_root'] = results_root
        config['path']['log'] = results_root
    
    # network
    if config['distortion'] == 'sr':
        config['network_G']['scale'] = scale

    return config


def dict2str(config, indent_l=1):
    msg = ''
    for k, v in config.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(config):
    if isinstance(config, dict):
        new_config = dict()
        for key, sub_config in config.items():
            new_config[key] = dict_to_nonedict(sub_config)
        return NoneDict(**new_config)
    elif isinstance(config, list):
        return [dict_to_nonedict(sub_config) for sub_config in config]
    else:
        return config


def check_resume(config, resume_iter):
    logger = logging.getLogger('base')
    if config['path']['resume_state']:
        if config['path'].get('pretrain_model_G', None) is not None or config['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        config['path']['pretrain_model_G'] = osp.join(config['path']['models'], '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + config['path']['pretrain_model_G'])
        if 'gan' in config['model']:
            config['path']['pretrain_model_D'] = osp.join(config['path']['models'], '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + config['path']['pretrain_model_D'])
