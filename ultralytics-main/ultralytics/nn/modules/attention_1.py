###################### SRM  attention  ####     START ##############################

"""
PyTorch implementation of Srm : A style-based recalibration module for
convolutional neural networks
As described in https://arxiv.org/pdf/1903.10829
SRM first extracts the style information from each channel of the feature maps by style pooling,
then estimates per-channel recalibration weight via channel-independent style integration.
By incorporating the relative importance of individual styles into feature maps,
SRM effectively enhances the representational ability of a CNN.
"""

import torch
from torch import nn


class SRM(nn.Module):
    def __init__(self, feature, channel):
        super().__init__()
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, groups=channel,
                             bias=False)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, h, w = x.shape
        # style pooling
        mean = x.reshape(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.reshape(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat([mean, std], dim=-1)
        # style integration
        z = self.cfc(u)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.reshape(b, c, 1, 1)
        return x * g.expand_as(x)

###################### SRM  attention  ####     END   ###############################