import torch 
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
from ..utils import LayerNorm, GRN
from timm.models.layers import trunc_normal_, DropPath
from .utils import Block,SFTLayer
from .transformer_layer import TransformerCrossEncoderLayer
import numpy as np
class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')







class DAttentionFuse(nn.Module):

    def __init__(
        self, 
        q_size=(4,32),
        kv_size=(4,32), 
        n_heads=8, 
        n_head_channels=80, 
        n_groups=4,
        attn_drop=0.0, 
        proj_drop=0.0, 
        stride=2, 
        offset_range_factor=2,
        use_pe=True, 
        args = None
    ):
        '''
        stage_idx from 2 to 3
        '''

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.offset_range_factor = offset_range_factor
        stage_idx = args.stage_idx
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]
       
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.n_group_channels, 2*self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(2*self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(2*self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            self.rpe_table = nn.Parameter(
                torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
            )
            trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        return ref

    def forward(self, x,y):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        
        q_off = einops.rearrange(torch.cat((x,y),dim=1), 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=2*self.n_group_channels)

        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        q = self.proj_q(y)
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=False) # B * g, Cg, Hg, Wg
            
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
    
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        
        if self.use_pe:
            rpe_table = self.rpe_table
            rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
            
            q_grid = self._get_ref_points(H, W, B, dtype, device)
            
            displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
            
            attn_bias = F.grid_sample(
                input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                grid=displacement[..., (1, 0)],
                mode='bilinear', align_corners=False
            )
            
            attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            
            attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj_drop(self.proj_out(out))
        
        return out, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class FuseModel(nn.Module):
    def __init__(self,dims,args) -> None:
        super().__init__()

        
        channels = [dims[-1] // 2 ** i for i in range(4)]
        
        refine_conv = nn.Conv2d
        self.deform_stride=args.deform_stride

        k_size = [(2,2),(2,1),(2,1),(1,1)]
        q_size = (2,32)
        in_out_ch = [(-1,-2),(-2,-3),(-3,-4),(-4,-4)]   

        self.binary_condition_layer = DAttentionFuse(q_size=q_size,kv_size=q_size,stride=self.deform_stride,n_head_channels=dims[-1]//8,args=args)

        self.binary2refine_linear_norm = nn.ModuleList()
        for i in range(len(k_size)):
            self.binary2refine_linear_norm.append(nn.Sequential(
                Block(dim=channels[in_out_ch[i][0]]),
                LayerNorm(channels[in_out_ch[i][0]], eps=1e-6, data_format="channels_first"),
                refine_conv(channels[in_out_ch[i][0]], channels[in_out_ch[i][1]], kernel_size=k_size[i], stride=k_size[i])), # [8, 32]
            )

    

    def forward(self,recog_feat,binary_feats,dec_in=None):
        multi_feat = []
        binary_feat = binary_feats[-1]
        for i in range(len(self.binary2refine_linear_norm)):
            binary_feat = self.binary2refine_linear_norm[i](binary_feat)
            multi_feat.append(binary_feat)
        binary_feat = binary_feat + binary_feats[0]
        multi_feat[3] += binary_feats[0]
        binary_refined_feat,pos,_ = self.binary_condition_layer(recog_feat, binary_feat)
        binary_refined_feat =  binary_refined_feat + binary_feat
        return binary_refined_feat,binary_feat
