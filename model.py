import math
import torch
from torch import nn
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN
import torch.nn.functional as F
from et import EfficientTransformer


def swish(x):
    return F.relu(x)

class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1, res_scale=0.1):

        super(residualBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.dconv1 = DCN(64, 64, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(n)
        # self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.dconv2 = DCN(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(n)
        self.res_scale = res_scale

    def forward(self, x):
        y = self.dconv1(x)
        # y = self.bn1(y)
        y = swish(y)
        # y = self.dconv2(y).mul(self.res_scale)
        y = self.dconv2(y)
        # y = self.bn2(y)
        return y + x


class Generator(nn.Module):
    def __init__(self, n_residual_blocks,upsample_factor):
        upsample_block_num = int(math.log(upsample_factor, 2))
        super(Generator, self).__init__()
        self.upsample_factor = upsample_factor
        self.n_residual_blocks = n_residual_blocks
        # 提取浅层特征
        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.trans1 = EfficientTransformer(inChannels=64, mlpDim=85)
        #上采样层

        block = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        # block.append(nn.Dropout(p=0.1, inplace=False))
        block.append(nn.Conv2d(64, 1, 9, 1, 4))
        self.block = nn.Sequential(*block)
        # self.block = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        #########################original version########################
        x = self.conv1(x)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)
        # y = self.trans1(y)
        x = (self.conv2(y)) + x
        x = self.block(x)
        return x


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

