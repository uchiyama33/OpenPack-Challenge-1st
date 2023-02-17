import glob

import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from vit_pytorch.vit import Attention, FeedForward, PreNorm


class FeedForwardEmbedding(nn.Module):
    def __init__(self, indim, outdim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outdim),
        )

    def forward(self, x):
        return self.net(x)


class Embedding(nn.Module):
    def __init__(self, cfg, num_classes=11, concat=True):
        super().__init__()
        self.set_param(cfg)
        self.concat = concat

        self.imu_patch_embedding = nn.Linear(self.imu_input_dim, self.imu_embedding_dim)
        self.e4acc_patch_embedding = nn.Linear(self.e4acc_input_dim, self.e4acc_embedding_dim)
        self.bbox_patch_embedding = nn.Linear(self.bbox_input_dim, self.bbox_embedding_dim)
        self.keypoint_patch_embedding = nn.Linear(self.keypoint_input_dim, self.keypoint_embedding_dim)

        self.ht_patch_embedding = nn.Embedding(2, self.ht_embedding_dim)
        self.printer_patch_embedding = nn.Embedding(2, self.printer_embedding_dim)


    def set_param(self, cfg):
        assert not (cfg.model.use_substitute_image and cfg.model.use_substitute_emb)
        assert not (cfg.model.resnet and cfg.model.use_cnn_feature)
        assert not (cfg.model.resnet and cfg.model.mbconv)

        self.use_substitute_image = cfg.model.use_substitute_image
        self.use_substitute_emb = cfg.model.use_substitute_emb
        self.add_defect_info = cfg.model.add_defect_info

        self.dim = cfg.model.dim
        self.imu_input_dim = cfg.dataset.stream.imu_dim * int(
            cfg.dataset.stream.frame_rate_imu * cfg.model.time_step_width / 1000
        )
        self.imu_embedding_dim = cfg.model.imu_dim
        self.keypoint_input_dim = cfg.dataset.stream.keypoint_dim * int(
            cfg.dataset.stream.frame_rate_keypoint * cfg.model.time_step_width / 1000
        )
        self.keypoint_embedding_dim = cfg.model.keypoint_dim
        self.e4acc_input_dim = cfg.dataset.stream.e4acc_dim * int(
            cfg.dataset.stream.frame_rate_e4acc * cfg.model.time_step_width / 1000
        )
        self.e4acc_embedding_dim = cfg.model.e4acc_dim
        self.bbox_input_dim = cfg.dataset.stream.bbox_dim * int(
            cfg.dataset.stream.frame_rate_keypoint * cfg.model.time_step_width / 1000
        )
        self.bbox_embedding_dim = cfg.model.bbox_dim
        self.ht_embedding_dim = cfg.model.ht_dim
        self.printer_embedding_dim = cfg.model.printer_dim

        if hasattr(cfg.model, "embedding_method"):
            self.embedding_method = cfg.model.embedding_method
        else:
            self.embedding_method = "linear"

        self.concat_dim = (
            self.imu_embedding_dim
            + self.keypoint_embedding_dim
            + self.e4acc_embedding_dim
            + self.bbox_embedding_dim
            + self.ht_embedding_dim
            + self.printer_embedding_dim
        )

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        b = imu.shape[0]
        t = imu.shape[1]
        x_list = []
        name_list = []

        imu = rearrange(imu, "b t f d -> b t (f d)")
        x_imu = self.imu_patch_embedding(imu)
        x_list.append(x_imu)
        name_list.append("imu")

        e4acc = rearrange(e4acc, "b t f d -> b t (f d)")
        x_e4acc = self.e4acc_patch_embedding(e4acc)
        x_list.append(x_e4acc)
        name_list.append("e4acc")

        bbox = rearrange(bbox, "b t f d -> b t (f d)")
        x_bbox = self.bbox_patch_embedding(bbox)
        x_list.append(x_bbox)
        name_list.append("bbox")

        keypoint = rearrange(keypoint, "b t f d n -> b t (f d n)")
        x_keypoint = self.keypoint_patch_embedding(keypoint)
        x_list.append(x_keypoint)
        name_list.append("keypoint")

        if self.ht_embedding_dim != 0:
            x_ht = self.ht_patch_embedding(ht)
            x_list.append(x_ht)
            name_list.append("ht")
        if self.printer_embedding_dim != 0:
            x_printer = self.printer_patch_embedding(printer)
            x_list.append(x_printer)
            name_list.append("printer")

        if self.concat:
            return torch.concat(x_list, dim=2)
        else:
            return x_list


class OpenPackBase(nn.Module):
    def __init__(self, cfg, num_classes=11, concat=True):
        super().__init__()
        self.embedding = Embedding(cfg, num_classes, concat)
        self.set_param(cfg)
        self.num_classes = num_classes

    def set_param(self, cfg):
        assert not (cfg.model.use_substitute_image and cfg.model.use_substitute_emb)
        assert not (cfg.model.resnet and cfg.model.use_cnn_feature)

        self.num_patches = cfg.model.num_patches
        self.depth = cfg.model.depth
        self.heads = cfg.model.heads
        self.mlp_dim = cfg.model.mlp_dim
        self.dim_head = cfg.model.dim_head
        self.use_pe = cfg.model.use_pe
        self.emb_dropout_p = cfg.model.emb_dropout
        self.dropout_p = cfg.model.dropout

        if cfg.model.dim == -1:
            self.dim = self.embedding.concat_dim
        else:
            self.dim = cfg.model.dim

        self.use_substitute_image = cfg.model.use_substitute_image
        self.use_substitute_emb = cfg.model.use_substitute_emb
        self.add_defect_info = cfg.model.add_defect_info

        self.imu_input_dim = cfg.dataset.stream.imu_dim * int(
            cfg.dataset.stream.frame_rate_imu * cfg.model.time_step_width / 1000
        )
        self.imu_embedding_dim = cfg.model.imu_dim
        self.keypoint_input_dim = cfg.dataset.stream.keypoint_dim * int(
            cfg.dataset.stream.frame_rate_keypoint * cfg.model.time_step_width / 1000
        )
        self.keypoint_embedding_dim = cfg.model.keypoint_dim
        self.e4acc_input_dim = cfg.dataset.stream.e4acc_dim * int(
            cfg.dataset.stream.frame_rate_e4acc * cfg.model.time_step_width / 1000
        )
        self.e4acc_embedding_dim = cfg.model.e4acc_dim
        self.bbox_embedding_dim = cfg.model.bbox_dim
        self.ht_embedding_dim = cfg.model.ht_dim
        self.printer_embedding_dim = cfg.model.printer_dim


class mySimpleTransformer(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.simple_vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim)
        if self.use_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.to_latent = nn.Identity()
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Linear(self.dim, num_classes)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.transformer(x)
        x = self.to_latent(x)

        feat = self.ln(x)
        return self.linear_head(feat), feat


class myTransformer(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.dropout = nn.Dropout(self.emb_dropout_p)
        self.transformer = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        if self.use_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.to_latent = nn.Identity()
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Linear(self.dim, num_classes)

        # print("!!! init xavier_normal_ !!!")
        # for n, p in self.named_parameters():
        #     if p.dim() >= 2:
        #         nn.init.xavier_normal_(p)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        feat = self.ln(x)
        return self.linear_head(feat), feat


class myConvTransformer(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.dropout = nn.Dropout(self.emb_dropout_p)
        self.transformer = ConvTransformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.to_latent = nn.Identity()
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Linear(self.dim, num_classes)


    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        feat = self.ln(x)
        return self.linear_head(feat), feat


class ConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, conv_k=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # PreNorm(dim, DConvAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        PreNorm(dim, ConvLayer(dim, conv_k, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        # for attn, ff, conv in self.layers:
        for attn, conv in self.layers:
            x = attn(x) + x
            # x = ff(x) + x
            x = conv(x) + x
        return x


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ConvLayer(nn.Module):
    def __init__(self, dim, conv_k, dropout=0.0, expansion=4):
        super().__init__()
        # self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding="same")
        hidden_dim = int(dim * expansion)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=conv_k, padding="same"),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(dim, dim, kernel_size=1, padding="same"),
                nn.Dropout(dropout),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(dim, hidden_dim, kernel_size=1, padding="same"),
                nn.GroupNorm(10, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_k, padding="same"),
                nn.GroupNorm(10, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, dim, kernel_size=1, padding="same"),
                nn.GroupNorm(10, dim),
            )

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x


class DConvAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Primer: Searching for Efficient Transformers for Language Modeling
        self.dconv_q = nn.Conv2d(heads, heads, (3, 1), stride=1, padding="same", groups=heads)
        self.dconv_k = nn.Conv2d(heads, heads, (3, 1), stride=1, padding="same", groups=heads)
        self.dconv_v = nn.Conv2d(heads, heads, (3, 1), stride=1, padding="same", groups=heads)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = self.dconv_q(q)
        k = self.dconv_k(k)
        v = self.dconv_v(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class myTransformerPlusLSTM(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.dropout = nn.Dropout(self.emb_dropout_p)
        self.transformer = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        if self.use_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.to_latent = nn.Identity()
        self.ln = nn.LayerNorm(self.dim)
        self.lstm = nn.LSTM(
            self.dim, self.dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        self.linear_head = nn.Linear(self.dim * 2, num_classes)


    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        feat = self.ln(x)
        feat = self.lstm(feat)[0]
        return self.linear_head(feat), feat


from conformer.encoder import ConformerBlock


class myConformer(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.dropout = nn.Dropout(self.emb_dropout_p)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )

        self.inner_module = InterResModule(self.dim, num_classes)

        if self.use_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.to_latent = nn.Identity()
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Linear(self.dim, num_classes)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.depth // 2:
                x, inner_logit = self.inner_module(x)
        x = self.to_latent(x)

        feat = self.ln(x)
        return self.linear_head(feat), inner_logit


class InterResModule(nn.Module):
    def __init__(self, dim_model, out_size):
        super(InterResModule, self).__init__()

        self.proj_1 = nn.Linear(dim_model, out_size)
        self.proj_2 = nn.Linear(out_size, dim_model)

    def forward(self, x):
        logits = self.proj_1(x)
        x = x + self.proj_2(logits.softmax(dim=-1))
        return x, logits


class myConformerAVEC(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):

        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.imu_embedding_dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.inner_module_imu = InterResModule(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.keypoint_embedding_dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.inner_module_keypoint = InterResModule(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.bbox_embedding_dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.inner_module_bbox = InterResModule(self.bbox_embedding_dim, num_classes)

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )

        if self.use_pe:
            self.pos_embedding_imu = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.imu_embedding_dim)
            )
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_bbox = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.bbox_embedding_dim)
            )

        self.fusion = FusionModule(self.dim, self.dim)

        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Linear(self.dim, num_classes)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        assert len(x_list) == 6 and x_list[1].nelement() == 0

        x_imu = x_list[0]
        x_bbox = x_list[2]
        x_keypoint = x_list[3]
        x_ht = x_list[4]
        x_printer = x_list[5]
        inner_logit_list = []

        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
        x_imu = self.dropout_imu(x_imu)
        for layer in self.layers_imu:
            x_imu = layer(x_imu)
        x_imu, inner_logit = self.inner_module_imu(x_imu)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_keypoint += self.pos_embedding_keypoint[:, :t]
        x_keypoint = self.dropout_keypoint(x_keypoint)
        for layer in self.layers_keypoint:
            x_keypoint = layer(x_keypoint)
        x_keypoint, inner_logit = self.inner_module_keypoint(x_keypoint)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_bbox += self.pos_embedding_bbox[:, :t]
        x_bbox = self.dropout_bbox(x_bbox)
        for layer in self.layers_bbox:
            x_bbox = layer(x_bbox)
        x_bbox, inner_logit = self.inner_module_bbox(x_bbox)
        inner_logit_list.append(inner_logit)

        x = self.fusion([x_imu, x_keypoint, x_ht, x_printer, x_bbox])

        for layer in self.layers:
            x = layer(x)

        feat = self.ln(x)
        return self.linear_head(feat), torch.stack(inner_logit_list).mean(0)


class FusionModule(nn.Module):
    def __init__(self, dim_in, dim_out, ff_ratio=4):
        super(FusionModule, self).__init__()

        dim_ffn = ff_ratio * dim_out

        # Layers
        self.layers = nn.Sequential(
            nn.Linear(
                dim_in,
                dim_ffn,
            ),
            nn.SiLU(),
            nn.Linear(dim_ffn, dim_out),
        )

    def forward(self, x_list):

        x = torch.cat(x_list, dim=-1)
        x = self.layers(x)

        return x


class myTransformerAVEC(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = Transformer(
            self.imu_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_imu = InterResModule(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = Transformer(
            self.keypoint_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_keypoint = InterResModule(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = Transformer(
            self.bbox_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_bbox = InterResModule(self.bbox_embedding_dim, num_classes)

        if self.use_pe:
            self.pos_embedding_imu = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.imu_embedding_dim)
            )
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_bbox = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.bbox_embedding_dim)
            )

        self.fusion = FusionModule(self.dim, self.dim)
        self.layers = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Linear(self.dim, num_classes)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        assert len(x_list) == 6 and x_list[1].nelement() == 0

        x_imu = x_list[0]
        x_bbox = x_list[2]
        x_keypoint = x_list[3]
        x_ht = x_list[4]
        x_printer = x_list[5]
        inner_logit_list = []

        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
        x_imu = self.dropout_imu(x_imu)
        x_imu = self.layers_imu(x_imu)
        x_imu, inner_logit = self.inner_module_imu(x_imu)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_keypoint += self.pos_embedding_keypoint[:, :t]
        x_keypoint = self.dropout_keypoint(x_keypoint)
        x_keypoint = self.layers_keypoint(x_keypoint)
        x_keypoint, inner_logit = self.inner_module_keypoint(x_keypoint)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_bbox += self.pos_embedding_bbox[:, :t]
        x_bbox = self.dropout_bbox(x_bbox)
        x_bbox = self.layers_bbox(x_bbox)
        x_bbox, inner_logit = self.inner_module_bbox(x_bbox)
        inner_logit_list.append(inner_logit)

        x = self.fusion([x_imu, x_keypoint, x_ht, x_printer, x_bbox])

        x = self.layers(x)

        feat = self.ln(x)
        return self.linear_head(feat), torch.stack(inner_logit_list).mean(0)


class myConformerAVECplusLSTM(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):

        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.imu_embedding_dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.inner_module_imu = InterResModuleLSTM(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.keypoint_embedding_dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.inner_module_keypoint = InterResModuleLSTM(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.bbox_embedding_dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )
        self.inner_module_bbox = InterResModuleLSTM(self.bbox_embedding_dim, num_classes)

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=self.dim,
                    num_attention_heads=self.heads,
                    feed_forward_expansion_factor=4,
                    conv_expansion_factor=2,
                    feed_forward_dropout_p=self.dropout_p,
                    attention_dropout_p=self.dropout_p,
                    conv_dropout_p=self.dropout_p,
                    conv_kernel_size=5,
                    half_step_residual=True,
                )
                for _ in range(self.depth)
            ]
        )

        if self.use_pe:
            self.pos_embedding_imu = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.imu_embedding_dim)
            )
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_bbox = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.bbox_embedding_dim)
            )

        self.fusion = FusionModule(self.dim, self.dim)

        self.ln = nn.LayerNorm(self.dim)
        self.lstm = nn.LSTM(
            self.dim, self.dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        self.linear_head = nn.Linear(self.dim * 2, num_classes)


    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        assert len(x_list) == 6 and x_list[1].nelement() == 0

        x_imu = x_list[0]
        x_bbox = x_list[2]
        x_keypoint = x_list[3]
        x_ht = x_list[4]
        x_printer = x_list[5]
        inner_logit_list = []

        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
        x_imu = self.dropout_imu(x_imu)
        for layer in self.layers_imu:
            x_imu = layer(x_imu)
        x_imu, inner_logit = self.inner_module_imu(x_imu)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_keypoint += self.pos_embedding_keypoint[:, :t]
        x_keypoint = self.dropout_keypoint(x_keypoint)
        for layer in self.layers_keypoint:
            x_keypoint = layer(x_keypoint)
        x_keypoint, inner_logit = self.inner_module_keypoint(x_keypoint)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_bbox += self.pos_embedding_bbox[:, :t]
        x_bbox = self.dropout_bbox(x_bbox)
        for layer in self.layers_bbox:
            x_bbox = layer(x_bbox)
        x_bbox, inner_logit = self.inner_module_bbox(x_bbox)
        inner_logit_list.append(inner_logit)

        x = self.fusion([x_imu, x_keypoint, x_ht, x_printer, x_bbox])

        for layer in self.layers:
            x = layer(x)

        feat = self.ln(x)
        feat = self.lstm(feat)[0]
        return self.linear_head(feat), torch.stack(inner_logit_list).mean(0)


class myTransformerAVECplusLSTM(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = Transformer(
            self.imu_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_imu = InterResModuleLSTM(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = Transformer(
            self.keypoint_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_keypoint = InterResModuleLSTM(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = Transformer(
            self.bbox_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_bbox = InterResModuleLSTM(self.bbox_embedding_dim, num_classes)

        if self.use_pe:
            self.pos_embedding_imu = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.imu_embedding_dim)
            )
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_bbox = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.bbox_embedding_dim)
            )

        self.fusion = FusionModule(self.dim, self.dim)
        self.layers = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln = nn.LayerNorm(self.dim)
        self.lstm = nn.LSTM(
            self.dim, self.dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        self.linear_head = nn.Linear(self.dim * 2, num_classes)


    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        assert len(x_list) == 6 and x_list[1].nelement() == 0

        x_imu = x_list[0]
        x_bbox = x_list[2]
        x_keypoint = x_list[3]
        x_ht = x_list[4]
        x_printer = x_list[5]
        inner_logit_list = []

        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
        x_imu = self.dropout_imu(x_imu)
        x_imu = self.layers_imu(x_imu)
        x_imu, inner_logit = self.inner_module_imu(x_imu)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_keypoint += self.pos_embedding_keypoint[:, :t]
        x_keypoint = self.dropout_keypoint(x_keypoint)
        x_keypoint = self.layers_keypoint(x_keypoint)
        x_keypoint, inner_logit = self.inner_module_keypoint(x_keypoint)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_bbox += self.pos_embedding_bbox[:, :t]
        x_bbox = self.dropout_bbox(x_bbox)
        x_bbox = self.layers_bbox(x_bbox)
        x_bbox, inner_logit = self.inner_module_bbox(x_bbox)
        inner_logit_list.append(inner_logit)

        x = self.fusion([x_imu, x_keypoint, x_ht, x_printer, x_bbox])

        x = self.layers(x)

        feat = self.ln(x)
        feat = self.lstm(feat)[0]
        return self.linear_head(feat), torch.stack(inner_logit_list).mean(0)


class InterResModuleLSTM(nn.Module):
    def __init__(self, dim_model, out_size):
        super(InterResModuleLSTM, self).__init__()

        self.proj_1 = nn.Linear(dim_model * 2, out_size)
        self.proj_2 = nn.Linear(out_size, dim_model)
        self.lstm = nn.LSTM(
            dim_model, dim_model, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )

    def forward(self, x):

        logits = self.proj_1(self.lstm(x)[0])
        x = x + self.proj_2(logits.softmax(dim=-1))
        return x, logits


class myConvTransformerAVEC(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = ConvTransformer(
            self.imu_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_imu = InterResModule(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = ConvTransformer(
            self.keypoint_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_keypoint = InterResModule(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = ConvTransformer(
            self.bbox_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_bbox = InterResModule(self.bbox_embedding_dim, num_classes)

        if self.use_pe:
            self.pos_embedding_imu = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.imu_embedding_dim)
            )
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_bbox = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.bbox_embedding_dim)
            )

        self.fusion = FusionModule(self.dim, self.dim)
        self.layers = ConvTransformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln = nn.LayerNorm(self.dim)
        self.linear_head = nn.Linear(self.dim, num_classes)


    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        assert len(x_list) == 6 and x_list[1].nelement() == 0

        x_imu = x_list[0]
        x_bbox = x_list[2]
        x_keypoint = x_list[3]
        x_ht = x_list[4]
        x_printer = x_list[5]
        inner_logit_list = []

        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
        x_imu = self.dropout_imu(x_imu)
        x_imu = self.layers_imu(x_imu)
        x_imu, inner_logit = self.inner_module_imu(x_imu)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_keypoint += self.pos_embedding_keypoint[:, :t]
        x_keypoint = self.dropout_keypoint(x_keypoint)
        x_keypoint = self.layers_keypoint(x_keypoint)
        x_keypoint, inner_logit = self.inner_module_keypoint(x_keypoint)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_bbox += self.pos_embedding_bbox[:, :t]
        x_bbox = self.dropout_bbox(x_bbox)
        x_bbox = self.layers_bbox(x_bbox)
        x_bbox, inner_logit = self.inner_module_bbox(x_bbox)
        inner_logit_list.append(inner_logit)

        x = self.fusion([x_imu, x_keypoint, x_ht, x_printer, x_bbox])

        x = self.layers(x)

        feat = self.ln(x)
        return self.linear_head(feat), torch.stack(inner_logit_list).mean(0)
    

class myConvTransformerAVECplusLSTM(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = ConvTransformer(
            self.imu_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_imu = InterResModuleLSTM(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = ConvTransformer(
            self.keypoint_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_keypoint = InterResModuleLSTM(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = ConvTransformer(
            self.bbox_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_bbox = InterResModuleLSTM(self.bbox_embedding_dim, num_classes)

        if self.use_pe:
            self.pos_embedding_imu = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.imu_embedding_dim)
            )
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_bbox = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.bbox_embedding_dim)
            )

        self.fusion = FusionModule(self.dim, self.dim)
        self.layers = ConvTransformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln = nn.LayerNorm(self.dim)
        self.lstm = nn.LSTM(
            self.dim, self.dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        self.linear_head = nn.Linear(self.dim * 2, num_classes)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        assert len(x_list) == 6 and x_list[1].nelement() == 0

        x_imu = x_list[0]
        x_bbox = x_list[2]
        x_keypoint = x_list[3]
        x_ht = x_list[4]
        x_printer = x_list[5]
        inner_logit_list = []

        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
        x_imu = self.dropout_imu(x_imu)
        x_imu = self.layers_imu(x_imu)
        x_imu, inner_logit = self.inner_module_imu(x_imu)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_keypoint += self.pos_embedding_keypoint[:, :t]
        x_keypoint = self.dropout_keypoint(x_keypoint)
        x_keypoint = self.layers_keypoint(x_keypoint)
        x_keypoint, inner_logit = self.inner_module_keypoint(x_keypoint)
        inner_logit_list.append(inner_logit)

        if self.use_pe:
            x_bbox += self.pos_embedding_bbox[:, :t]
        x_bbox = self.dropout_bbox(x_bbox)
        x_bbox = self.layers_bbox(x_bbox)
        x_bbox, inner_logit = self.inner_module_bbox(x_bbox)
        inner_logit_list.append(inner_logit)

        x = self.fusion([x_imu, x_keypoint, x_ht, x_printer, x_bbox])

        x = self.layers(x)

        feat = self.ln(x)
        feat = self.lstm(feat)[0]
        return self.linear_head(feat), torch.stack(inner_logit_list).mean(0)