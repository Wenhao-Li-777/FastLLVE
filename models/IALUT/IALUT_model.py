import torch
import torch.nn as nn

from .backbone import LightBackbone3D, denoise
from .lut import LUT4DGenerator
from IALUT import IALUT_transform

class IALUT_LLVE(nn.Module):
    """Intensity-Aware Lookup Tables for Low-Light Video Enhancement.

    Args:
        n_ranks (int, optional): Number of ranks for IA-LUT (or the number of basis
            LUTs). Default: 3.
        n_vertices_4d (int, optional): Size of the IA-LUT. If `n_vertices_4d` <= 0,
            the IA-LUT will be disabled. Default: 33.
        n_base_feats (int, optional): The channel multiplier of the backbone network.
            Default: 8.
    """

    def __init__(self, 
        input_resolution,
        train_resolution,
        n_ranks = 3,
        n_vertices_4d = 33,
        n_base_feats = 8,
        smooth_factor = 0.,
        monotonicity_factor = 10.0):
        super(IALUT_LLVE, self).__init__()
        self.n_ranks = n_ranks
        self.n_vertices_4d = n_vertices_4d
        self.n_base_feats = n_base_feats
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor

        self.backbone = LightBackbone3D(input_resolution=input_resolution, train_resolution=train_resolution,
                                        extra_pooling=True, n_base_feats=n_base_feats)
        self.denoise = denoise()

        self.LUTGenerator = LUT4DGenerator(4, 3, n_vertices_4d, self.backbone.out_channels, n_ranks)

        self.init_weights()

    def init_weights(self):
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)
        self.LUTGenerator.init_weights()
    
    def forward(self, videos, if_train=True):
        min_max = (0, 1)

        codes, intensity_map = self.backbone(videos, if_train)
        weights, luts = self.LUTGenerator(codes)
        
        context_videos = torch.cat((videos, intensity_map), dim=1)
        outputs = IALUT_transform(context_videos, luts)
        outputs_clamp = torch.clamp(outputs, min_max[0], min_max[1])
        clear_outputs = self.denoise(outputs_clamp)

        if if_train:
            reg_smoothness, reg_monotonicity = self.LUTGenerator.regularizations(self.smooth_factor, self.monotonicity_factor)
            reg_smoothness = max(0, reg_smoothness)
            reg_monotonicity = max(0, reg_monotonicity)
        else:
            reg_smoothness = 0
            reg_monotonicity = 0

        return outputs, clear_outputs, weights, reg_smoothness, reg_monotonicity