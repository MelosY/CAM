import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DepthWiseUNetBlock,UNetBlock,MoreUNetBlock


class BinaryDecoder(nn.Module):
    def __init__(self,dims,num_classes,strides,args) -> None:
        super().__init__()

        channels = [dims[-1] // 2 ** i for i in range(4)]
        self.linear_enc2binary = nn.Sequential(
                nn.Conv2d(dims[-1], dims[-1], kernel_size=3,stride=1,padding=1),
                nn.SyncBatchNorm(dims[-1]),
            )
        self.strides = strides
        self.use_deformable = False
        self.binary_decoder = nn.ModuleList()
        unet = DepthWiseUNetBlock if args.use_depthwise_unet else UNetBlock
        unet = MoreUNetBlock if args.use_more_unet else unet

        for i in range(3):
            up_sample_stride = self.strides[::-1][i]
            cin, cout = channels[i], channels[i+1]
            self.binary_decoder.append(unet(cin, cout, nn.SyncBatchNorm, up_sample_stride, self.use_deformable))

        last_stride = (self.strides[0][0]//2, self.strides[0][1]//2)
        self.binary_decoder.append(unet(cout, cout, nn.SyncBatchNorm, last_stride, self.use_deformable))
            


        if args.binary_loss_type == 'CrossEntropyDiceLoss' or args.binary_loss_type == 'BanlanceMultiClassCrossEntropyLoss':
            segm_num_cls = num_classes - 2
        else:
            segm_num_cls = num_classes - 3
        self.binary_pred = nn.Conv2d(channels[-1], segm_num_cls, kernel_size=1, stride=1, bias=True)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p_h, p_w = self.strides[0]
        p_h = p_h//2
        p_w = p_w//2
        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w

        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p_h, w, p_w))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h*p_w * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, patch_size**2, h, w)
        imgs: (N, 3, H, W)
        """
        p_h, p_w = self.strides[0]
        p_h = p_h//2
        p_w = p_w//2
        _, _, h, w = x.shape
        assert p_h * p_w == x.shape[1]
        
        x = x.permute(0, 2, 3, 1) # [N, h, w, 4*4]
        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w))
        x = torch.einsum('nhwpq->nhpwq', x)
        imgs = x.reshape(shape=(x.shape[0], h * p_h, w * p_w))
        return imgs
    def forward(self,x,time=None):
        """
          x: the encoder feat to init the query for binary prediction, usually this is equal to the `img`.
          img: the encoder feat.
          txt: the unnormmed text to get the length of predicted words.
          txt_feat: the text feat before character prediction.
          xs: the encoder feat from different stages
        """

        binary_feats = []
        x = self.linear_enc2binary(x)
        binary_feats.append(x.clone())

        for i, d in enumerate(self.binary_decoder):

            x = d(x)
            binary_feats.append(x.clone())
        #return None,binary_feats
        x = self.binary_pred(x)

        if self.training:
            return x, binary_feats
        else:
            # return torch.sigmoid(x), binary_feat
            return x.softmax(1), binary_feats
