import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_conf, conf=None, sampler=None):
    phase = dataset_conf['phase']
    if phase == 'train':
        if conf['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_conf['n_workers']
            assert dataset_conf['batch_size'] % world_size == 0
            batch_size = dataset_conf['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_conf['n_workers'] * len(conf['gpu_ids'])
            batch_size = dataset_conf['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_conf):
    mode = dataset_conf['mode']
    # datasets for image restoration
    if mode == 'video_SDSD':
        from data.video_dataset_SDSD import Video_Dataset_SDSD as D
    elif mode == 'video_SDSD_test':
        from  data.video_dataset_SDSD_test import Video_Dataset_SDSD_test as D
    elif mode == 'video_SMID':
        from data.video_dataset_SMID import Video_Dataset_SMID as D
    elif mode == 'video_SMID_test':
        from data.video_dataset_SMID_test import Video_Dataset_SMID_test as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_conf)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_conf['name']))
    return dataset