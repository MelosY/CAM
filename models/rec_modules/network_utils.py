import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from timm.models.layers import to_2tuple, trunc_normal_
class DepthWiseConv(nn.Module):
    def __init__(self,cin,cout) -> None:
        super().__init__()
        self.depthconv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False, groups=cin)
        self.pointconv = nn.Conv2d(cin,cout, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self,x):
        x = self.depthconv(x)
        x = self.pointconv(x)
        return x

class FPN(nn.Module):
    def __init__(self,in_channels) -> None:
        super().__init__()
        self.toplayer = nn.Conv2d(in_channels[2],in_channels[2],kernel_size=1,stride=1,padding=0)
        
        self.latlayer1 = nn.Conv2d(in_channels[0],in_channels[2],kernel_size=1,stride=1,padding=0)
        self.latlayer2 = nn.Conv2d(in_channels[1],in_channels[2],kernel_size=1,stride=1,padding=0)

        self.smooth1 = DepthWiseConv(in_channels[2],in_channels[2])
        self.smooth2 = DepthWiseConv(in_channels[2],in_channels[2])
        self.smooth3 = DepthWiseConv(in_channels[2],in_channels[2])
    def _upsample_add(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear') + y
    
    def fpn_head(self,input):
        output = input[0]
        for x in input[1:]:
            output =output + F.interpolate(x,size=output.shape[2:],mode="bilinear",align_corners=False) 
        return output

    def forward(self,x):
        c1,c2,c3 = x

        p3 = self.toplayer(c3)
        p2 = self._upsample_add(p3,self.latlayer2(c2))
        p1 = self._upsample_add(p2,self.latlayer1(c1))

        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        fuse_p = self.fpn_head((p1,p2,p3))
        return fuse_p


