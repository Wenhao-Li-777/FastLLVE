import logging
logger = logging.getLogger('base')


def create_model(conf):
    model = conf['model']
    if model == 'video_base':
        from .Video_model import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(conf)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
