# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm

from .rec_modules.decoder import *

from .rec_modules.fuse_utils import FuseModel
from .rec_modules.utils import *
from .rec_modules.decoder_utils import *
from .rec_modules.stn_head import STNHead
from .rec_modules.tps import TPSSpatialTransformer


def to_contiguous(tensor):
  if tensor.is_contiguous():
    return tensor
  else:
    return tensor.contiguous()
class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.

    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,
                 decoder_name='tf_decoder', max_seq_len=25, beam_width=0,
                 strides=[(4,4), (2,2), (2,2), (2,2)],
                 args=None,
                 ):
        super().__init__()
        self.strides = strides
        self.depths = depths
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=strides[0], stride=strides[0]),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=strides[i+1], stride=strides[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]




        self.mid_size = args.mid_size
        if self.mid_size:
            self.enc_downsample = nn.Sequential(
                nn.Conv2d(dims[-1], dims[-1]//2, kernel_size=1,stride=1),
                nn.SyncBatchNorm(dims[-1]//2),
                #nn.ReLU6(inplace=True),
                nn.Conv2d(dims[-1]//2, dims[-1]//2, kernel_size=3, stride=1, padding=1, bias=False, groups=dims[-1]//2),
                nn.Conv2d(dims[-1]//2, dims[-1]//2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.SyncBatchNorm(dims[-1]//2),
            )
            dims[-1] = dims[-1]//2
        # recognition decoder
            self.linear_enc2recog = nn.Sequential(
                    nn.Conv2d(dims[-1], dims[-1], kernel_size=1,stride=1,),
                    nn.SyncBatchNorm(dims[-1]),
                    #nn.ReLU6(inplace=True),
                    nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=1, padding=1, bias=False, groups=dims[-1]),
                    nn.Conv2d(dims[-1], dims[-1], kernel_size=1, stride=1, padding=0, bias=False),
                    nn.SyncBatchNorm(dims[-1]),
                )
        else:
            self.linear_enc2recog = nn.Sequential(
                    nn.Conv2d(dims[-1], dims[-1]//2, kernel_size=1,stride=1),
                    nn.SyncBatchNorm(dims[-1]//2),
                    #nn.ReLU6(inplace=True),
                    nn.Conv2d(dims[-1]//2, dims[-1], kernel_size=3,stride=1,padding=1),
                    nn.SyncBatchNorm(dims[-1]),
                )

        self.decoder = create_decoder(decoder_name, num_classes, max_seq_len)
        d_embedding = self.decoder.d_embedding
        self.linear_norm = nn.Sequential(
            nn.Linear(dims[-1], d_embedding),
            nn.LayerNorm(d_embedding, eps=1e-6),
        )

        self.beam_width = beam_width



        
        self.binary_decoder = BinaryDecoder(dims,num_classes,strides,args)
        
        self.fuse_model = FuseModel(dims,args)
    
        self.apply(self._init_weights)

        self.tps = TPSSpatialTransformer(
            output_image_size=(32,args.stn_output_width),
            num_control_points=20,
            margins=(0.05,0.05))
        self.stn_head = STNHead(
            in_planes=3,
            num_ctrlpoints=20,
            use_depthwise_stn=True,
            stn_lr=0.1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.SyncBatchNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
            return {}

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward_features(self, x):
       
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        return feats


    def forward(self, x):
        output = {}
        imgs, tgt, tgt_lens, binary_tgt = x

        
        # input images are downsampled before being fed into stn_head.
        stn_input = F.interpolate(imgs, (32,64), mode='bilinear', align_corners=True)
        stn_img_feat, ctrl_points = self.stn_head(stn_input)
        
        imgs, _ = self.tps(imgs, ctrl_points)
        if not self.training:
            # save for visualization
            output['ctrl_points'] = ctrl_points
            output['rectified_images'] = imgs


        enc_feats = self.forward_features(imgs)

        enc_feat = enc_feats[-1]
        if self.mid_size:
            enc_feat = self.enc_downsample(enc_feat)
        

        target = tgt
        if not self.training:
            tgt, tgt_lens = None, None

        
        output["enc_feat"] =enc_feat
        start_query_feat = None

        # binary mask

        pred_binary, binary_feats = self.binary_decoder(enc_feat)
        output['pred_binary'] = pred_binary
         
        reg_feat = self.linear_enc2recog(enc_feat)
        B, C, H, W = reg_feat.shape
        last_feat,binary_feat = self.fuse_model(reg_feat,binary_feats)
        
        dec_in = last_feat.reshape(B, C, H*W).permute(0, 2, 1)
        dec_in = self.linear_norm(dec_in)

        
        output["binary_feat"] = binary_feats[-1]

        dec_output, dec_attn_maps, dec_attn_output = self.decoder(dec_in,
                                                                  dec_in,
                                                                  targets=tgt,
                                                                  tgt_lens=tgt_lens,
                                                                  train_mode=self.training,
                                                                  beam_width=self.beam_width,
                                                                  start_query=start_query_feat)
        output['rec_output'] = dec_output
        output['dec_attn_output'] = dec_attn_output
        
        return output

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_vit(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
