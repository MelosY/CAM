


import torch 
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
from ..utils import LayerNorm, GRN
from timm.models.layers import trunc_normal_, DropPath
import math

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x):
        return x*torch.sigmoid(x)



class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d, stride, deformable=False):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        stride_h, stride_w = stride
        if stride_h == 1:
            kernel_h = 1
            padding_h = 0
        elif stride_h == 2:
            kernel_h = 4
            padding_h = 1
        elif stride_h == 4:
            kernel_h = 4
            padding_h = 0

        if stride_w == 1:
            kernel_w = 1
            padding_w = 0
        elif stride_w == 2:
            kernel_w = 4
            padding_w = 1
        elif stride_w == 4:
            kernel_w = 4
            padding_w = 0

        conv = nn.Conv2d 
        

        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), bias=True)
        self.conv = nn.Sequential(
            conv(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            conv(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)

class DepthWiseUNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d, stride, deformable=False):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        stride_h, stride_w = stride
        if stride_h == 1:
            kernel_h = 1
            padding_h = 0
        elif stride_h == 2:
            kernel_h = 4
            padding_h = 1
        elif stride_h == 4:
            kernel_h = 4
            padding_h = 0

        if stride_w == 1:
            kernel_w = 1
            padding_w = 0
        elif stride_w == 2:
            kernel_w = 4
            padding_w = 1
        elif stride_w == 4:
            kernel_w = 4
            padding_w = 0

        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False, groups=cin),
            nn.Conv2d(cin, cin, kernel_size=1, stride=1, padding=0, bias=False),
            bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False, groups=cin),
            nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False),
            bn2d(cout),
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x.contiguous())
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class SFTLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Linear(dim_in, dim_in,)
        self.SFT_scale_conv1 = nn.Linear(dim_in, dim_out,)
        self.SFT_shift_conv0 = nn.Linear(dim_in, dim_in,)
        self.SFT_shift_conv1 = nn.Linear(dim_in, dim_out,)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift
    


    




class MoreUNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d, stride, deformable=False):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        stride_h, stride_w = stride
        if stride_h == 1:
            kernel_h = 1
            padding_h = 0
        elif stride_h == 2:
            kernel_h = 4
            padding_h = 1
        elif stride_h == 4:
            kernel_h = 4
            padding_h = 0

        if stride_w == 1:
            kernel_w = 1
            padding_w = 0
        elif stride_w == 2:
            kernel_w = 4
            padding_w = 1
        elif stride_w == 4:
            kernel_w = 4
            padding_w = 0

        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False, groups=cin),
            nn.Conv2d(cin, cin, kernel_size=1, stride=1, padding=0, bias=False),
            bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False, groups=cin),
            nn.Conv2d(cin, cin, kernel_size=1, stride=1, padding=0, bias=False),
            bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False, groups=cin),
            nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False),
            bn2d(cout), nn.ReLU6(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False, groups=cout),
            nn.Conv2d(cout, cout, kernel_size=1, stride=1, padding=0, bias=False),
            bn2d(cout)
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)