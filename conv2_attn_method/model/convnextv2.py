# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN
from .priors import PriorGenerator

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim,  #输入输出通道一样，
                                kernel_size=7, padding=3, #大感受野
                                groups=dim) # depthwise conv #每个channel单独卷积-->channel之间没有信息交换 负责空间信息建模
        self.norm = LayerNorm(dim, eps=1e-6) #对通道进行归一化
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers ；升维，负责channel融合，通道建模
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        #在训练过程中，以一定的概率随机删除当前Block的残差分支，让网络学习多个不同深度的子网络，以提高模型泛化能力
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)，以适应layernorm
        x = self.norm(x) 
        x = self.pwconv1(x)
        x = self.act(x) #GELU激活函数，增强非线性映射能力
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    
    torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,
                    dilation=1,groups=1,bias=True,padding_mode="zeros",
                    device=None,dtype=None)
    torch.nn.Linear(in_features,out_features,bias=True) #对输入的最后一个维度进行线性变换
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.0,
                 prior_type="none",
                 prior_alpha=0.5,
                 **prior_kwargs,
                 ):
        super().__init__()
        self.depths = depths

        self.prior_generator=PriorGenerator(prior_type,**prior_kwargs)
        self.register_buffer("prior_alpha",torch.full((4,),float(prior_alpha)))

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first") #pytorch官方默认处理（N,H,W,C)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer) 

        #创建一个用于存放神经网络子模块的列表容器
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] #在0-drop_path_rate之间生成sum(depths)个均匀间隔的数
        #假设drop_path_rate=0.2,depths=[3,3,9,3],那么dp_rates=torch.linspace(0,0.2,18)
        cur = 0
        for i in range(4):
            stage = nn.Sequential(  #depths=[3, 3, 9, 3] dims=[96, 192, 384, 768], 
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])] #*参数解包
                #Stage1
                ##Block(dim=96,drop_path=dp_rates[0+0])cur=0
                ##Block(dim=96,drop_path=dp_rates[0+1])
                ##Block(dim=96,drop_path=dp_rates[0+2])
                #Stage2
                ##Block(dim=192,drop_path=dp_rates[3+0])cur=3
                ##Block(dim=192,drop_path=dp_rates[3+1])
                ##Block(dim=192,drop_path=dp_rates[3+2])
                #Stage3
                ##Block(dim=384,drop_path=dp_rates[6+0])cur=6
                ##Block(dim=384,drop_path=dp_rates[6+1])
                ##...
                ##Block(dim=384,drop_path=dp_rates[6+8])
                #Stage4
                ##Block(dim=768,drop_path=dp_rates[15+0])cur=15
                ##Block(dim=768,drop_path=dp_rates[15+1])
                ##Block(dim=768,drop_path=dp_rates[15+2])
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x,mask=None):
        for i in range(4):
            x = self.downsample_layers[i](x)
            # x = self.stages[i](x)
            attn=self.prior_generator(mask,x.shape[-2:])
            if attn is not None:
                x=x*(1.0+self.prior_alpha[i]*attn)
            #0805放在stage前
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C) #先GAP再LayerNorm

    def forward(self, x,mask=None):
        x = self.forward_features(x,mask=mask)
        x = self.head(x)
        return x

# def convnextv2_atto(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
#     return model

# def convnextv2_femto(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
#     return model

# def convnext_pico(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
#     return model

# def convnextv2_nano(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
#     return model

def convnextv2_tiny(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt"
        print(f"Loading pretrained weights from:{url}")
        state_dict=torch.hub.load_state_dict_from_url(url=url,map_location="cpu")

        if "model" in state_dict:
            state_dict=state_dict['model']
        num_classes = kwargs.get("num_classes", 1000)
        if num_classes != 1000:
            print(f"=> Removing 'head' weights because num_classes={num_classes} != 1000")
            for k in list(state_dict.keys()):
                if "head" in k:
                    del state_dict[k]
                    
        #  加载权重，strict=False 允许分类头权重不匹配
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)



    return model

def convnextv2_base(pretrained=False, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        # 官方 ImageNet-1k 预训练权重 
        url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt"
        print(f"Loading pretrained weights from: {url}")
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict['model']
        num_classes = kwargs.get("num_classes", 1000)
        if num_classes != 1000:
            print(f"=> Removing 'head' weights because num_classes={num_classes} != 1000")
            for k in list(state_dict.keys()):
                if "head" in k:
                    del state_dict[k]
                    
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    return model


def convnextv2_large(pretrained=False, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        # 官方 ImageNet-1k 预训练权重 
        url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt"
        print(f"Loading pretrained weights from: {url}")
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict['model']
        num_classes = kwargs.get("num_classes", 1000)
        if num_classes != 1000:
            print(f"=> Removing 'head' weights because num_classes={num_classes} != 1000")
            for k in list(state_dict.keys()):
                if "head" in k:
                    del state_dict[k]
                    
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    return model

# def convnextv2_huge(**kwargs):
#     model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
#     return model