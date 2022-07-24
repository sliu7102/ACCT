
r"""
    @Author : sliu
    @README : Transformer blocks for fault diagnosis
    @Date   : 2022-07-25 02: 03: 35
    @Related: 
"""



import torch
import torch.nn as nn
import numpy as np 
from model.layers import *

import os
import sys
sys.path.append(os.getcwd())
from params import get_parameters

config = get_parameters()


def get_block(block_type, **kargs):
    if block_type=='mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type=='ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type=='tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    L = size[2]
    cut_rat = 1. - lam 
    
    cut_l = np.int(L * cut_rat  )

    # * uniform
    cx = np.random.randint(L)

    bbx1 = np.clip(cx - cut_l // 2, 0, L)
    bbx2 = np.clip(cx + cut_l // 2, 0, L)

    lam = 1. - (bbx2-bbx1)/L

    return bbx1, bbx2, lam


def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay == 'linear':
        # * linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # * stochastic depth decay rule
    elif drop_path_decay == 'fix':
        # * use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # * use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr



class TransBlocks(nn.Module):
    def __init__(self, img_size= 2048, patch_size= 32, in_chans= 1, num_classes= 9, embed_dim= 64, depth= 12,
                 num_heads= 12, mlp_ratio= 4., qkv_bias= False, qk_scale= None, drop_rate= 0., attn_drop_rate= 0.,
                 drop_path_rate= 0., drop_path_decay= 'linear', norm_layer= nn.LayerNorm, head_dim = None,
                 skip_lam = 1.0, order= None, mix_token= False, return_dense= False):
        super().__init__()
        self.num_classes = num_classes
        
        self.num_features = self.embed_dim = embed_dim  # * num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes  # * dim of output, should be num_classes

        patch_embed_fn = PatchEmbedComplex

        self.patch_embed = patch_embed_fn(img_size= img_size, patch_size= patch_size, in_chans= in_chans, embed_dim= embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p= drop_rate)

        if order is None:
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            print(dpr)    

            self.blocks = nn.ModuleList([
                Block(
                    dim= embed_dim, num_heads= num_heads, head_dim= head_dim, mlp_ratio= mlp_ratio, qkv_bias= qkv_bias, qk_scale= qk_scale,
                    drop= drop_rate, attn_drop= attn_drop_rate, drop_path= dpr[i], norm_layer= norm_layer, skip_lam= skip_lam, group= 1)
                for i in range(depth)
            ])
        else:
            # * use given order to sequentially generate modules
            dpr=get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i],
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(len(order))
            ])

        self.norm = norm_layer(embed_dim) 
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.Sequential(nn.Linear(embed_dim, num_classes), nn.Sigmoid()) if num_classes > 0 else nn.Identity()

        self.return_dense = return_dense
        self.mix_token = mix_token

        if return_dense:
            self.aux_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            # self.aux_head = nn.Sequential(nn.Linear(embed_dim, num_classes), nn.Sigmoid()) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        # * 参数初始化
        trunc_normal_(self.pos_embed, std= .02)
        trunc_normal_(self.cls_token, std= .02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std= .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std= .02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool = ''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    # * 输入源数据，获取特征向量 token
    def forward_emdeddings(self, x):
        x = self.patch_embed(x)

        return x

    def scaler(self, data):
        max_ = torch.max(data, axis=1)[0].reshape(-1 , 1)
        min_ = torch.min(data, axis=1)[0].reshape(-1 , 1)
        return (data - min_) / (max_ - min_)

        
    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim= 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        a = 0   # * 层数计数器
        x_star = torch.zeros(x.shape)
        x_end = torch.zeros(x.shape).cuda()

        for blk in self.blocks:
            a += 1
            x = blk(x)
            if self.training:
                if a == config.depth - config.CC_mid_layer: 
                    x_star = x[:, 1:]
                if a == config.depth:
                    x_end = x[:, 1:]
                else:
                    continue

        x = self.norm(x)

        return x, x_star, x_end



    # ! 前向主函数
    def forward(self, x):
        x = self.forward_emdeddings(x)
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta) *config.RB_lamda + (1. - config.RB_lamda)
            patch_n = x.shape[2]

            bbx1, bbx2, lam = rand_bbox(x.size(), lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2] = x.flip(0)[:, :, bbx1:bbx2] 
            x = temp_x

        else:
            bbx1, bbx2 = 0, 0 

        x = x.flatten(2).transpose(1, 2)
        if self.training and config.SL_switch:
            context_target = x.detach()

        x, x_star, x_end = self.forward_tokens(x)

        if self.training and config.SL_switch:
            patches = x[:, 1:].permute(0, 2, 1)
        x_feature = x[:, 0]
        atten_map = x[:, 1:]
        x_cls = self.head(x[:, 0])

        if self.return_dense:
            x_aux = self.aux_head(x[:, 1:])
            if not self.training:

                if config.tsne:
                    return x_cls + 0.5 * x_aux.max(1)[0]  , x_feature
                    # , atten_map
                else:
                    return x_cls + 0.5 * x_aux.max(1)[0]
                
            
            # * recover the mixed part
            if self.mix_token and self.training:
                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, :] = x_aux.flip(0)[:, bbx1:bbx2, :] 
                x_aux = temp_x

                return x_cls, x_aux, (bbx1, bbx2), lam, x_star, x_end




def vit_acct(pretrained=False, **kwargs): 
    model = TransBlocks(patch_size=config.patch_size, embed_dim=config.embed_dim, depth=config.depth,
                   num_heads=config.num_heads, mlp_ratio=3.,
                   p_emb='complex', skip_lam=2., return_dense=True, mix_token=True, **kwargs)
    return model


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(2048//config.patch_size, 256, kernel_size= 5, stride= 2, padding= 2, bias=False), 
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(0.1),
                                        )
                                        
        self.conv2 = nn.Sequential(nn.Conv1d(256, 512, kernel_size= 3, stride= 1, padding= 1, bias=False), 
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(0.1),
                                        )

                                        
        self.avgpoool = nn.AdaptiveAvgPool1d((1)) 

        self.fc = nn.Sequential(nn.Linear(512, 128),
                                    nn.LeakyReLU(0.1), 
                                    nn.Linear(128, 1),
                                    )

        # ^ Params initialization 
        for m in self.modules(): 
            if isinstance(m, (nn.Conv1d, nn.Linear)): 
                trunc_normal_(m.weight, std= .02) 

                if isinstance(m, nn.Conv1d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                



    def T_embed(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpoool(x).reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x_star, x_end):

        B, C, _ = x_star.shape

        random_index = torch.randperm(C)
        random_index2 = torch.randperm(C)
        x_end_shuffle = x_end[:, random_index]
        x_end_shuffle2 = x_end[:, random_index2]

        T0 = self.T_embed(torch.cat([x_star, x_end], dim = -1)) 
        T1 = self.T_embed(torch.cat([x_star, x_end_shuffle], dim = -1)) 
        T2 = self.T_embed(torch.cat([x_star, x_end_shuffle2], dim = -1)) 
        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() + T2.logsumexp(dim = 1).mean() - np.log(B)) 
        lower_bound = torch.clamp(lower_bound, min= 1e-5 , max= None) 
        return -torch.log(lower_bound )


    def flops(self):
        img_size = 128 * 2
        block_flops = dict(
        conv1=img_size/2*128*128*2*7,
        conv2=img_size/4*128*2*128*4*3,
        intermediate = 512 * 128,
        out = 128 * 1
        )
        return sum(block_flops.values())

