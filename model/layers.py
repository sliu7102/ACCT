r"""    layers module for lvvit
author: sliu
readme: rewrite of token labeling

"""
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

import math

from model.droppath import DropPath


import warnings

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 5

class GroupLinear(nn.Module):
    '''
    Group Linear operator 
    '''
    def __init__(self, in_planes, out_channels, groups= 1, bias= True):
        super(GroupLinear, self).__init__()
        assert in_planes%groups == 0
        assert out_channels%groups == 0

        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups = groups
        self.bias = bias
        self.group_in_dim = int(self.in_dim/self.groups)
        self.group_out_dim = int(self.out_dim/self.groups)

        self.group_weight = nn.Parameter(torch.zeros(self.groups, self.group_in_dim, self.group_out_dim))
        self.group_bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        t,b,d = x.size()
        x = x.view(t,b,self.groups,int(d/self.groups))
        out = torch.einsum('tbgd,gdf->tbgf', (x, self.group_weight)).reshape(t,b,self.out_dim)+self.group_bias

        return out

    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Mlp(nn.Module):
    r'''
    MLP with support to use group linear operator
    '''
    def __init__(self, in_features, hidden_features= None, out_features= None, act_layer= nn.GELU, drop= 0., group= 1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if group == 1:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = GroupLinear(in_features, hidden_features)
            self.fc2 = GroupLinear(hidden_features, out_features)
        
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    r'''
    Multi-head self-attention

    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        with some modification to support different num_heads and head_dim.
    '''
    def __init__(self, dim, num_heads= 8, head_dim= None, qkv_bias= False, qk_scale= None, attn_drop= 0., proj_drop= 0.):
        super().__init__()

        self.num_heads = num_heads

        if head_dim is not None:
            self.head_dim=head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # * 同时获取 3 个矩阵: Q, K, V
        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias= qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim* self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, padding_mask = None):

        # ! 注意这里的输入为: B, N, C
        B, N, C = x.shape
        # * (b, n, c)  -->>  (b, n, 6*21*3=378)  -->>  (b, n, 3, 6, 21)  -->>  (3, b, 6, n, 21)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # B,heads,N,C/heads 
        q, k, v = qkv[0], qkv[1], qkv[2]

        # trick here to make q@k.t more stable
        attn = ((q * self.scale) @ k.transpose(-2, -1))

        if padding_mask is not None:
            attn = attn.view(B, self.num_heads, N, N)
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_float = attn.softmax(dim=-1, dtype=torch.float32)
            attn = attn_float.type_as(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim* self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x





class Block(nn.Module):
    r"""
    Pre-layernorm transformer block
    Args:
        dim (int): Number of input channels.

    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads= num_heads, head_dim= head_dim, qkv_bias= qkv_bias, qk_scale= qk_scale, attn_drop= attn_drop, proj_drop= drop
        )
        # ? nn.Identity() 的作用？  -- identity模块不改变输入
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features= dim, hidden_features= self.mlp_hidden_dim, act_layer= act_layer, drop= drop, group= group)

    def forward(self, x, padding_mask= None):
        x = x + self.drop_path(self.attn(self.norm1(x), padding_mask)) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam

        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        i = self.mlp_hidden_dim
        mha_block_flops = dict(
        kqv=3 * h * h  ,
        attention_scores=h * s,
        attn_softmax=SOFTMAX_FLOPS * s * heads,
        attention_dropout=DROPOUT_FLOPS * s * heads,
        attention_scale=s * heads,
        attention_weighted_avg_values=h * s,
        attn_output=h * h,
        attn_output_bias=h,
        attn_output_dropout=DROPOUT_FLOPS * h,
        attn_output_residual=h,
        attn_output_layer_norm=LAYER_NORM_FLOPS * h,)
        ffn_block_flops = dict(
        intermediate=h * i,
        intermediate_act=ACTIVATION_FLOPS * i,
        intermediate_bias=i,
        output=h * i,
        output_bias=h,
        output_dropout=DROPOUT_FLOPS * h,
        output_residual=h,
        output_layer_norm=LAYER_NORM_FLOPS * h,)
        
        return sum(mha_block_flops.values())*s + sum(ffn_block_flops.values())*s



class MHABlock(nn.Module):
    """
    Multihead Attention block with residual branch
    """
    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.skip_lam = skip_lam
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, padding_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x*self.skip_lam), padding_mask))/self.skip_lam
        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        block_flops = dict(
        kqv=3 * h * h ,
        attention_scores=h * s,
        attn_softmax=SOFTMAX_FLOPS * s * heads,
        attention_dropout=DROPOUT_FLOPS * s * heads,
        attention_scale=s * heads,
        attention_weighted_avg_values=h * s,
        attn_output=h * h,
        attn_output_bias=h,
        attn_output_dropout=DROPOUT_FLOPS * h,
        attn_output_residual=h,
        attn_output_layer_norm=LAYER_NORM_FLOPS * h,)

        return sum(block_flops.values())*s
class FFNBlock(nn.Module):
    """
    Feed forward network with residual branch
    """
    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.skip_lam = skip_lam
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop, group=group)
    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x*self.skip_lam)))/self.skip_lam
        return x
    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        i = self.mlp_hidden_dim
        block_flops = dict(
        intermediate=h * i,
        intermediate_act=ACTIVATION_FLOPS * i,
        intermediate_bias=i,
        output=h * i,
        output_bias=h,
        output_dropout=DROPOUT_FLOPS * h,
        output_residual=h,
        output_layer_norm=LAYER_NORM_FLOPS * h,)

        return sum(block_flops.values())*s


class PatchEmbedComplex(nn.Module):
    r""" 
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """
    def __init__(self, img_size= 2048, patch_size= 16, in_chans= 1, embed_dim= 128):
        super().__init__()

        new_patch_size = patch_size // 2

        num_patches = img_size // patch_size
        # * 定义全局变量
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        mid_chans = 96  # * 384

        self.conv1 = nn.Conv1d(in_chans, mid_chans, kernel_size= 17, stride= 2, padding= 8, bias= False)  # * N = 1024
        self.bn1 = nn.BatchNorm1d(mid_chans)
        self.relu = nn.LeakyReLU(inplace= True)
        self.conv2 = nn.Conv1d(mid_chans, mid_chans *2, kernel_size= 15, stride= 1, padding= 7, bias= False)  # * N = 1024
        self.bn2 = nn.BatchNorm1d(mid_chans *2)
        self.conv3 = nn.Conv1d(mid_chans *2, mid_chans *2, kernel_size= 9, stride= 1, padding= 4, bias= False)  # * N = 1024
        self.bn3 = nn.BatchNorm1d(mid_chans *2)
        
        self.conv4 = nn.Conv1d(mid_chans *2, mid_chans *3, kernel_size= 5, stride= 1, padding= 2, bias= False)  # * N = 1024
        self.bn4 = nn.BatchNorm1d(mid_chans *3)

        self.proj = nn.Conv1d(mid_chans *3, embed_dim, kernel_size= new_patch_size, stride= new_patch_size)

        self.downsample = nn.Sequential(nn.Conv1d(mid_chans, mid_chans * 3, kernel_size= 3, stride= 1, padding= 1, bias= False),
                                        nn.BatchNorm1d(mid_chans * 3)
                                        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        indenti = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = x + self.downsample(indenti)/1. 

        x = self.proj(x)


        return x

    def flops(self):
        mid_chans = 96
        img_size = self.img_size
        block_flops = dict(
        conv1=img_size/2*1*mid_chans*1*17,
        conv2=img_size/2*mid_chans*1*mid_chans*2*15,
        conv3=img_size/2*mid_chans*2*mid_chans*2*9,
        conv4=img_size/2*mid_chans*2*mid_chans*3*5,
        conv5= img_size/2*mid_chans*mid_chans*3*3,
        proj=img_size/2*mid_chans*3*self.embed_dim,
        )
        return sum(block_flops.values())


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    '''type: (Tensor, float, float, float, float) -> Tensor '''
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



