"""Feature Splitting Module (FSM) shared by AIResNet and transfer-learning backbones."""

import torch
from torch import nn


class SPPModule(nn.Module):
    def __init__(self, pool_mode="avg", sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == "avg":
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == "max":
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList(
            [nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes]
        )

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule("avg", pool_size)
        self.max_spp = SPPModule("max", pool_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                dilation=1,
                groups=1,
                bias=False,
            ),
            # GroupNorm: BatchNorm2d fails when N*H*W==1 (e.g. batch_size=1 and 1x1 maps)
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )

        _channels = channels * int(sum([x**2 for x in pool_size]))
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False
            ),
            nn.GroupNorm(32, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_input = self.avg_spp(x) + self.max_spp(x)
        channel_scale = self.channel(channel_input)

        spatial_input = torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )
        spatial_scale = self.spatial(spatial_input)

        x_age = (x * channel_scale + x * spatial_scale) * 0.5

        x_id = x - x_age

        return x_id, x_age
