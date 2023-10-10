import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class DistIterSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, ratio=100):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas # 进程个数 默认等于world_size(GPU个数)
        self.rank = rank # 当前属于哪个进程/哪块GPU
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas)) # 每个进程的样本个数
        self.total_size = self.num_samples * self.num_replicas # 数据集总样本的个数

    def __iter__(self):
        # Shuffle处理：打乱数据集顺序
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dsize = len(self.dataset)
        indices = [v % dsize for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
