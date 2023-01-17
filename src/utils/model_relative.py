import glob

import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import torch
import torch.nn.functional as F
from cnn.my_stgcn import STGCN4Seg
from cnn.my_stgcn_vit import STGCN4Seg_vit
from cnn.resnet import generate_r3d_model
from cnn.resnet2d import my_resnet
from cnn.resnet2d_vit import my_resnet_vit
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from utils.model import OpenPackBase


# https://github.com/chinhsuanwu/coatnet-pytorch/blob/master/coatnet.py
class RelativeAttention(nn.Module):
    def __init__(self, dim, num_patches, heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.num_patches = num_patches

        self.heads = heads
        self.scale = dim_head**-0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.num_patches - 1), heads))

        coords = torch.meshgrid((torch.arange(self.num_patches)))
        coords = torch.flatten(torch.stack(coords), 1)  # 1, 50
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.num_patches - 1  # 負をなくす
        # relative_coords[0] *= self.num_patches - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, "(h w) c -> 1 c h w", h=self.num_patches, w=self.num_patches)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RelativeTransformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, RelativeAttention(dim, num_patches, heads, dim_head, dropout), nn.LayerNorm
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), nn.LayerNorm),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class myRelativeTransformer(OpenPackBase):
    def __init__(
        self,
        num_patches,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        emb_dropout,
        dropout,
        dim_head=64,
        *args,
        **kargs,
    ):

        super().__init__(num_classes=num_classes, *args, **kargs)
        cfg = kargs["cfg"]

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = RelativeTransformer(
            dim, cfg.model.num_patches, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        # print("!!! init xavier_normal_ !!!")
        # for n, p in self.named_parameters():
        #     if p.dim() >= 2:
        #         nn.init.xavier_normal_(p)

    def forward(
        self,
        imu,
        keypoint,
        ht,
        printer,
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
        return self.linear_head(x)
