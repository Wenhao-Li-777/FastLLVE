import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class dfus_block(nn.Module):
    def __init__(self):
        super(dfus_block, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(inplace=True))

        self.convc3 = nn.Sequential(nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.convd3 = nn.Sequential(nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                    nn.ReLU(inplace=True))
        self.convc3d3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                      nn.ReLU(inplace=True))
        self.convd3c3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        fea1 = self.conv1(x)

        feac3 = self.convc3(fea1)
        fead3 = self.convd3(fea1)
        feac3d3 = self.convc3d3(feac3)
        fead3c3 = self.convd3c3(fead3)

        fea = torch.cat([feac3, fead3, feac3d3, fead3c3], dim=1)
        fea = self.conv2(fea)

        return torch.cat([fea1, fea], dim=1)


class denoise(nn.Module):
    def __init__(self):
        super(denoise, self).__init__()
        # ddfn Feature _extraction
        self.convc3 = nn.Sequential(nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.convd3 = nn.Sequential(nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                    nn.ReLU(inplace=True))
        self.convc3d3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(1, 2, 2), dilation=(1, 2, 2)),
                                      nn.ReLU(inplace=True))
        self.convd3c3 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        # ddfn Feature_integration
        dfus_block_generator = functools.partial(dfus_block)
        self.dfus = make_layer(dfus_block_generator, 1)

        # ddfn Reconstruction
        self.Reconstruction = nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feac3 = self.convc3(x)
        fead3 = self.convd3(x)
        feac3d3 = self.convc3d3(feac3)
        fead3c3 = self.convd3c3(fead3)
        fea = torch.cat([feac3, fead3, feac3d3, fead3c3], dim=1)

        fea = self.dfus(fea)
        fea = self.Reconstruction(fea)

        return fea


class LightBackbone3D(nn.Module):
    def __init__(self, input_resolution, train_resolution, n_base_feats, extra_pooling=False, **kwargs):
        super(LightBackbone3D, self).__init__()
        self.n_feats = n_base_feats
        self.extra_pooling = extra_pooling

        # Conv_net
        self.conv = nn.Sequential(nn.Conv3d(3, self.n_feats, kernel_size=3, stride=(1, 2, 2), padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats, affine=True),
                                  nn.Conv3d(self.n_feats, self.n_feats * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats * 2, affine=True),
                                  nn.Conv3d(self.n_feats * 2, self.n_feats * 4, kernel_size=3, stride=(1, 2, 2),
                                            padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats * 4, affine=True),
                                  nn.Conv3d(self.n_feats * 4, self.n_feats * 8, kernel_size=3, stride=(1, 2, 2),
                                            padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(self.n_feats * 8, affine=True),
                                  nn.Conv3d(self.n_feats * 8, self.n_feats * 8, kernel_size=3, stride=(1, 2, 2),
                                            padding=1),
                                  nn.LeakyReLU(0.2))

        # dropout and pooling for LUT geneator
        self.drop = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.out_channels = self.n_feats * 8 * (16 if extra_pooling else 7 * (input_resolution[0] // 32) * (input_resolution[1] // 32))
        self.pool_intensity = nn.AdaptiveAvgPool3d((1, input_resolution[1], input_resolution[0]))
        self.pool_intensity_train = nn.AdaptiveAvgPool3d((1, train_resolution[1], train_resolution[0]))

        # Deconv_net
        self.deconv = nn.Sequential(nn.InstanceNorm3d(self.n_feats * 8, affine=True),
            nn.ConvTranspose3d(self.n_feats * 8, self.n_feats * 8, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 8, affine=True),
            nn.ConvTranspose3d(self.n_feats * 8, self.n_feats * 4, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 4, affine=True),
            nn.ConvTranspose3d(self.n_feats * 4, self.n_feats * 2, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats * 2, affine=True),
            nn.ConvTranspose3d(self.n_feats * 2, self.n_feats, kernel_size=3, stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(self.n_feats, affine=True),
            nn.ConvTranspose3d(self.n_feats, 1, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)))

    def forward(self, videos, if_train):
        b, c, t, h, w = videos.shape

        # Extract feature
        x = self.conv(videos)
        if self.extra_pooling:
            codes = self.drop(x)
            codes = self.pool(codes).view(b, -1)
        else:
            codes = self.drop(x).view(b, -1)

        # Create intensity map
        intensity_map = self.deconv(x)

        intensity_map = intensity_map[:, :, 1:t, :, :]
        if if_train:
            intensity_map = self.pool_intensity_train(intensity_map)
        else:
            intensity_map = self.pool_intensity(intensity_map)
        # intensity_map = self.intensity_norm(b, intensity_map)
        intensity_map_list = []
        for i in range(t):
            intensity_map_list.append(intensity_map)
        intensity_map = torch.cat(intensity_map_list, dim=2)

        return codes, intensity_map

    def intensity_norm(self, batch, intensity_map):
        for i in range(batch):
            max = torch.max(intensity_map[i]).item()
            min = torch.min(intensity_map[i]).item()
            min_max = max - min
            intensity_map[i] = intensity_map[i].sub(min).div(min_max)

        return intensity_map
