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

from cnn.mbconv_cnn import my_mbconv
from cnn.mbconv_cnn_gn import my_mbconv_gn
from cnn.mbconv_cnn_vit import my_mbconv_vit
from cnn.my_stgcn import STGCN4Seg
from cnn.my_stgcn_vit import STGCN4Seg_vit
from cnn.ori_stgcn import OriSTGCN
from cnn.resnet import generate_r3d_model
from cnn.resnet1d_vit import my_resnet1d_vit
from cnn.resnet2d import my_resnet
from cnn.resnet2d_cifar import my_resnet_cifar
from cnn.resnet2d_vit import my_resnet_vit


def get_inplanes(embedding_dim, n_layers):
    if n_layers == 1:
        return [embedding_dim, 1, 1, 1]
    if n_layers == 2:
        return [embedding_dim // 2, embedding_dim, 1, 1]
    if n_layers == 3:
        return [embedding_dim // 4, embedding_dim // 2, embedding_dim, 1]
    if n_layers == 4:
        return [embedding_dim // 8, embedding_dim // 4, embedding_dim // 2, embedding_dim]


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

        # self.ff_embedding = PreNorm(
        #     self.concat_dim,
        #     FeedForwardEmbedding(
        #         self.concat_dim, cfg.model.dim, cfg.model.mlp_dim, dropout=cfg.model.emb_dropout
        #     ),
        # )

        if self.resnet1d:
            if cfg.resnet1d.type == "vit":
                self.imu_patch_embedding = nn.Sequential(
                    my_resnet1d_vit(
                        inplanes=self.imu_inplanes,
                        n_input_channel=cfg.dataset.stream.imu_dim,
                        n_layers=cfg.resnet1d.n_layers,
                        num_groups=self.imu_num_groups
                        # norm_layer=nn.Identity,
                    ),
                    nn.Linear(self.imu_inplanes[-1], self.imu_embedding_dim),
                )
                self.e4acc_patch_embedding = my_resnet1d_vit(
                    inplanes=self.e4acc_inplanes,
                    n_input_channel=cfg.dataset.stream.e4acc_dim,
                    n_layers=cfg.resnet1d.n_layers,
                    num_groups=self.e4acc_num_groups
                    # norm_layer=nn.Identity,
                )
        elif self.embedding_method == "conv":
            self.imu_patch_embedding = nn.Sequential(
                nn.Conv1d(
                    cfg.dataset.stream.imu_dim,
                    self.imu_embedding_dim,
                    5,
                    padding="same",
                    groups=cfg.dataset.stream.imu_dim,
                ),
                nn.Conv1d(
                    self.imu_embedding_dim,
                    self.imu_embedding_dim,
                    5,
                    padding="same",
                    groups=self.imu_embedding_dim,
                ),
                nn.AdaptiveAvgPool1d(1),
            )
            self.e4acc_patch_embedding = nn.Sequential(
                nn.Conv1d(
                    cfg.dataset.stream.e4acc_dim,
                    self.e4acc_embedding_dim,
                    3,
                    padding="same",
                    groups=cfg.dataset.stream.e4acc_dim,
                ),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.imu_patch_embedding = nn.Linear(self.imu_input_dim, self.imu_embedding_dim)
            self.e4acc_patch_embedding = nn.Linear(self.e4acc_input_dim, self.e4acc_embedding_dim)

        self.bbox_patch_embedding = nn.Linear(self.bbox_input_dim, self.bbox_embedding_dim)

        if self.st_gcn:
            Ks = cfg.st_gcn.Ks
            Kt = cfg.st_gcn.Kt
            A = optorch.models.keypoint.get_adjacency_matrix(layout="MSCOCO", hop_size=Ks - 1)
            if cfg.st_gcn.type == "vit":
                self.keypoint_patch_embedding = STGCN4Seg_vit(2, cfg.st_gcn.out_channels, Ks=Ks, Kt=Kt, A=A)
            elif cfg.st_gcn.type == "ori":
                self.keypoint_patch_embedding = OriSTGCN(
                    2, cfg.model.keypoint_dim, Ks, Kt, A, edge_importance_weighting=True
                )
            else:
                self.keypoint_patch_embedding = STGCN4Seg(2, cfg.st_gcn.out_channels, Ks=Ks, Kt=Kt, A=A)
        elif self.embedding_method == "conv":
            self.keypoint_patch_embedding = nn.Sequential(
                nn.Conv1d(
                    cfg.dataset.stream.keypoint_dim,
                    self.keypoint_embedding_dim,
                    5,
                    padding="same",
                    groups=cfg.dataset.stream.keypoint_dim // 2,
                ),
                nn.Conv1d(
                    self.keypoint_embedding_dim,
                    self.keypoint_embedding_dim,
                    5,
                    padding="same",
                    groups=self.keypoint_embedding_dim,
                ),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.keypoint_patch_embedding = nn.Linear(self.keypoint_input_dim, self.keypoint_embedding_dim)

        self.ht_patch_embedding = nn.Embedding(2, self.ht_embedding_dim)
        self.printer_patch_embedding = nn.Embedding(2, self.printer_embedding_dim)

        # CNNの前
        if self.use_substitute_image:
            self.kinect_depth_null = nn.Parameter(
                torch.randn(1, self.kinect_depth_image_size, self.kinect_depth_image_size)
            )
            # self.rs02_depth_null = nn.Parameter(
            #     torch.randn(3, self.rs02_depth_image_size, self.rs02_depth_image_size)
            # )

        # CNNの後
        if self.use_substitute_emb:
            self.kinect_depth_null = nn.Parameter(torch.randn(1, self.kinect_depth_embedding_dim))
            self.rs02_depth_null = nn.Parameter(torch.randn(1, self.rs02_depth_embedding_dim))

        if self.add_defect_info:
            self.kinect_depth_defect = nn.Parameter(torch.randn(self.kinect_depth_embedding_dim))
            self.kinect_depth_exist = nn.Parameter(torch.randn(self.kinect_depth_embedding_dim))
            self.rs02_depth_defect = nn.Parameter(torch.randn(self.rs02_depth_embedding_dim))
            self.rs02_depth_exist = nn.Parameter(torch.randn(self.rs02_depth_embedding_dim))

        # self.kinect_depth_patch_embedding = nn.Linear(kinect_depth_input_dim, kinect_depth_embedding_dim)
        # self.rs02_depth_patch_embedding = nn.Linear(rs02_depth_input_dim, rs02_depth_embedding_dim)

        # 3DCNN
        # self.kinect_depth_patch_embedding = generate_r3d_model(
        #     10, get_inplanes(kinect_depth_embedding_dim), 1
        # )
        # self.rs02_depth_patch_embedding = generate_r3d_model(10, get_inplanes(rs02_depth_embedding_dim), 3)

        # 2DCNN
        # self.kinect_resnet = create_feature_extractor(resnet18(pretrained=use_pretrained_resnet), ["avgpool"])
        # self.kinect_depth_patch_embedding = nn.Linear(512, kinect_depth_embedding_dim)
        # self.rs02_resnet = create_feature_extractor(resnet18(pretrained=use_pretrained_resnet), ["avgpool"])
        # self.rs02_depth_patch_embedding = nn.Linear(512, rs02_depth_embedding_dim)

        # simple 2DCNN
        if self.resnet:

            if cfg.resnet.type == "original":
                self.kinect_depth_patch_embedding = my_resnet(
                    inplanes=self.kinect_inplanes,
                    n_input_channel=1,
                    n_layers=cfg.resnet.n_layers,
                    # norm_layer=nn.Identity,
                )
                self.rs02_depth_patch_embedding = my_resnet(
                    inplanes=self.rs02_inplanes,
                    n_input_channel=3,
                    n_layers=cfg.resnet.n_layers,
                    # norm_layer=nn.Identity,
                )
            if cfg.resnet.type == "vit":
                self.kinect_depth_patch_embedding = my_resnet_vit(
                    inplanes=self.kinect_inplanes,
                    n_input_channel=1,
                    n_layers=cfg.resnet.n_layers,
                    num_groups=self.kinect_num_groups
                    # norm_layer=nn.Identity,
                )
                self.rs02_depth_patch_embedding = my_resnet_vit(
                    inplanes=self.rs02_inplanes,
                    n_input_channel=3,
                    n_layers=cfg.resnet.n_layers,
                    num_groups=self.rs02_num_groups
                    # norm_layer=nn.Identity,
                )
            # self.rs02_depth_patch_embedding = my_resnet(
            #     inplanes=get_inplanes(self.rs02_depth_embedding_dim), n_input_channel=3
            # )
        elif self.mbconv:
            if cfg.mbconv.type == "original":
                self.kinect_depth_patch_embedding = my_mbconv(
                    channels=get_inplanes(self.kinect_depth_embedding_dim, cfg.mbconv.n_layers),
                    in_channels=1,
                    n_layers=cfg.mbconv.n_layers,
                )
                self.rs02_depth_patch_embedding = my_mbconv(
                    channels=get_inplanes(self.kinect_depth_embedding_dim, cfg.mbconv.n_layers),
                    in_channels=3,
                    n_layers=cfg.mbconv.n_layers,
                )
            if cfg.mbconv.type == "gn":
                self.kinect_depth_patch_embedding = my_mbconv_gn(
                    channels=get_inplanes(self.kinect_depth_embedding_dim, cfg.mbconv.n_layers),
                    in_channels=1,
                    n_layers=cfg.mbconv.n_layers,
                )
                self.rs02_depth_patch_embedding = my_mbconv_gn(
                    channels=get_inplanes(self.kinect_depth_embedding_dim, cfg.mbconv.n_layers),
                    in_channels=3,
                    n_layers=cfg.mbconv.n_layers,
                )
            if cfg.mbconv.type == "vit":
                self.kinect_depth_patch_embedding = my_mbconv_vit(
                    channels=get_inplanes(self.kinect_depth_embedding_dim, cfg.mbconv.n_layers),
                    in_channels=1,
                    n_layers=cfg.mbconv.n_layers,
                )
                self.rs02_depth_patch_embedding = my_mbconv_vit(
                    channels=get_inplanes(self.kinect_depth_embedding_dim, cfg.mbconv.n_layers),
                    in_channels=3,
                    n_layers=cfg.mbconv.n_layers,
                )
        elif self.resnet_cifar:
            assert self.kinect_depth_embedding_dim == 128
            self.kinect_depth_patch_embedding = my_resnet_cifar(
                n_blocks=cfg.resnet_cifar.n_blocks,
                in_channels=1,
            )
            self.rs02_depth_patch_embedding = my_resnet_cifar(
                n_blocks=cfg.resnet_cifar.n_blocks,
                in_channels=3,
            )
        elif self.use_cnn_feature:
            self.kinect_depth_patch_embedding = nn.Linear(
                cfg.dataset.stream.kinect_feature_dim, self.kinect_depth_embedding_dim
            )
        else:
            self.kinect_depth_patch_embedding = nn.Linear(
                self.kinect_depth_input_dim, self.kinect_depth_embedding_dim
            )
            # self.rs02_depth_patch_embedding = nn.Linear(
            #     self.rs02_depth_input_dim, self.rs02_depth_embedding_dim
            # )

    def set_param(self, cfg):
        # assert (
        #     cfg.model.imu_dim
        #     + cfg.model.keypoint_dim
        #     + cfg.model.e4acc_dim
        #     + cfg.model.ht_dim
        #     + cfg.model.printer_dim
        #     + cfg.model.kinect_depth_dim
        #     + cfg.model.rs02_depth_dim
        #     == cfg.model.dim
        # )
        assert not (cfg.model.use_substitute_image and cfg.model.use_substitute_emb)
        assert not (cfg.model.resnet and cfg.model.use_cnn_feature)
        assert not (cfg.model.resnet and cfg.model.mbconv)

        self.imu_inplanes = get_inplanes(cfg.model.imu_dim, cfg.resnet1d.n_layers)
        self.imu_num_groups = 16
        self.imu_embedding_dim = cfg.model.imu_dim
        if cfg.model.e4acc_dim == -1:
            self.e4acc_inplanes = [64, 128, 256, 512]
            self.e4acc_num_groups = 32
            self.e4acc_embedding_dim = self.rs02_inplanes[cfg.resnet1d.n_layers - 1]
        else:
            self.e4acc_inplanes = get_inplanes(cfg.model.e4acc_dim, cfg.resnet1d.n_layers)
            self.e4acc_num_groups = 5
            self.e4acc_embedding_dim = cfg.model.e4acc_dim

        if cfg.model.kinect_depth_dim == -1:
            self.kinect_inplanes = [64, 128, 256, 512]
            self.kinect_num_groups = 32
            self.kinect_depth_embedding_dim = self.kinect_inplanes[cfg.resnet.n_layers - 1]
        else:
            self.kinect_inplanes = get_inplanes(cfg.model.kinect_depth_dim, cfg.resnet.n_layers)
            self.kinect_num_groups = 5
            self.kinect_depth_embedding_dim = cfg.model.kinect_depth_dim
        if cfg.model.rs02_depth_dim == -1:
            self.rs02_inplanes = [64, 128, 256, 512]
            self.rs02_num_groups = 32
            self.rs02_depth_embedding_dim = self.rs02_inplanes[cfg.resnet.n_layers - 1]
        else:
            self.rs02_inplanes = get_inplanes(cfg.model.rs02_depth_dim, cfg.resnet.n_layers)
            self.rs02_num_groups = 5
            self.rs02_depth_embedding_dim = cfg.model.rs02_depth_dim

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
        # self.kinect_depth_embedding_dim = cfg.model.kinect_depth_dim
        self.kinect_depth_input_dim = cfg.dataset.stream.kinect_depth_dim
        self.kinect_depth_image_size = cfg.model.image_size
        # self.rs02_depth_embedding_dim = cfg.model.rs02_depth_dim
        self.rs02_depth_input_dim = cfg.dataset.stream.rs02_depth_dim
        self.rs02_depth_image_size = cfg.model.image_size
        self.use_pretrained_resnet = cfg.model.use_pretrained_resnet
        self.st_gcn = cfg.model.st_gcn
        self.resnet = cfg.model.resnet
        self.resnet_cifar = cfg.model.resnet_cifar
        self.mbconv = cfg.model.mbconv
        self.use_cnn_feature = cfg.model.use_cnn_feature
        self.cutout_p = cfg.train.dataaug.cutout_p
        self.resnet1d = cfg.model.resnet1d

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
            + self.kinect_depth_embedding_dim
            + self.rs02_depth_embedding_dim
        )

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        b = imu.shape[0]
        t = imu.shape[1]
        x_list = []
        name_list = []

        if self.resnet1d:
            if self.imu_embedding_dim != 0:
                imu = rearrange(imu, "b t f d -> (b t) d f")
                x_imu = self.imu_patch_embedding(imu)
                x_imu = rearrange(x_imu, "(b t) d -> b t d", t=t)
                x_list.append(x_imu)
                name_list.append("imu")

            if self.e4acc_embedding_dim != 0:
                e4acc = rearrange(e4acc, "b t f d -> (b t) d f")
                x_e4acc = self.e4acc_patch_embedding(e4acc)
                x_e4acc = rearrange(x_e4acc, "(b t) d -> b t d", t=t)
                x_list.append(x_e4acc)
                name_list.append("e4acc")
        elif self.embedding_method == "conv":
            if self.imu_embedding_dim != 0:
                imu = rearrange(imu, "b t f d -> (b t) d f")
                x_imu = self.imu_patch_embedding(imu).squeeze()
                x_imu = rearrange(x_imu, "(b t) d -> b t d", t=t)
                x_list.append(x_imu)
                name_list.append("imu")

            if self.e4acc_embedding_dim != 0:
                e4acc = rearrange(e4acc, "b t f d -> (b t) d f")
                x_e4acc = self.e4acc_patch_embedding(e4acc).squeeze()
                x_e4acc = rearrange(x_e4acc, "(b t) d -> b t d", t=t)
                x_list.append(x_e4acc)
                name_list.append("e4acc")
        else:
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

        if self.st_gcn:
            keypoint = rearrange(keypoint, "b t f d n -> (b t) d f n")
            x_keypoint = self.keypoint_patch_embedding(keypoint)
            x_keypoint = rearrange(x_keypoint, "(b t) d -> b t d", t=t)
        elif self.embedding_method == "conv":
            keypoint = rearrange(keypoint, "b t f d n -> (b t) (d n) f")
            x_keypoint = self.keypoint_patch_embedding(keypoint).squeeze()
            x_keypoint = rearrange(x_keypoint, "(b t) d -> b t d", t=t)
        else:
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

        if self.use_substitute_image:
            kinect_depth[torch.logical_not(exist_data_kinect_depth)] = self.kinect_depth_null.unsqueeze(
                0
            ).repeat(kinect_depth[torch.logical_not(exist_data_kinect_depth)].shape[0], 1, 1, 1)
            rs02_depth[torch.logical_not(exist_data_rs02_depth)] = self.rs02_depth_null.unsqueeze(0).repeat(
                rs02_depth[torch.logical_not(exist_data_rs02_depth)].shape[0], 1, 1, 1
            )

        # kinect_depth = rearrange(kinect_depth, "b t f c h w -> b (t f) (c h w)")
        # x_kinect_depth = self.kinect_depth_patch_embedding(kinect_depth)
        # x_kinect_depth = rearrange(x_kinect_depth, "b (t f) d -> b t (f d)", t=t)
        # rs02_depth = rearrange(rs02_depth, "b t f c h w -> b (t f) (c h w)")
        # x_rs02_depth = self.rs02_depth_patch_embedding(rs02_depth)
        # x_rs02_depth = rearrange(x_rs02_depth, "b (t f) d -> b t (f d)", t=t)

        if self.resnet or self.mbconv or self.resnet_cifar:
            # 3dcnn
            # kinect_depth = rearrange(kinect_depth, "b t f c h w -> (b t) c f h w")
            # x_kinect_depth = self.kinect_depth_patch_embedding(kinect_depth)
            # x_kinect_depth = rearrange(x_kinect_depth, "(b t) c -> b t c", t=t)

            # 2dcnn
            # kinect_depth = rearrange(kinect_depth, "b t f c h w -> (b t) (c f) h w", f=1).repeat(1, 3, 1, 1)
            # x_kinect_depth = self.kinect_resnet(kinect_depth)["avgpool"].squeeze()
            # x_kinect_depth = rearrange(x_kinect_depth, "(b t) d -> b t d", t=t)
            # x_kinect_depth = self.kinect_depth_patch_embedding(x_kinect_depth)

            # simple 2dcnn
            if self.kinect_depth_embedding_dim != 0:
                kinect_depth = rearrange(kinect_depth, "b t f c h w -> (b t) (c f) h w", f=1)
                x_kinect_depth = self.kinect_depth_patch_embedding(kinect_depth).squeeze()
                x_kinect_depth = rearrange(x_kinect_depth, "(b t) c -> b t c", t=t)

                if self.use_substitute_emb:
                    assert exist_data_kinect_depth.shape[2] == 1
                    _exist_data_kinect_depth = exist_data_kinect_depth[:, :, 0]
                    x_kinect_depth[
                        torch.logical_not(_exist_data_kinect_depth)
                    ] = self.kinect_depth_null.repeat(
                        x_kinect_depth[torch.logical_not(_exist_data_kinect_depth)].shape[0], 1
                    )

                # add defect info
                if self.add_defect_info:
                    kinect_exist_per = (
                        (exist_data_kinect_depth.sum(2) / exist_data_kinect_depth.shape[2])
                        .unsqueeze(-1)
                        .repeat(1, 1, self.kinect_depth_exist.shape[0])
                    )
                    x_kinect_depth = (
                        x_kinect_depth
                        + kinect_exist_per * self.kinect_depth_exist
                        + (1 - kinect_exist_per) * self.kinect_depth_defect
                    )
                x_list.append(x_kinect_depth)
                name_list.append("kinect_depth")

            # 3dcnn
            # rs02_depth = rearrange(rs02_depth, "b t f c h w -> (b t) c f h w")
            # x_rs02_depth = self.rs02_depth_patch_embedding(rs02_depth)
            # x_rs02_depth = rearrange(x_rs02_depth, "(b t) c -> b t c", t=t)
            # add defect info

            # 2dcnn
            # rs02_depth = rearrange(rs02_depth, "b t f c h w -> (b t) (c f) h w", f=1)
            # x_rs02_depth = self.rs02_resnet(rs02_depth)["avgpool"].squeeze()
            # x_rs02_depth = rearrange(x_rs02_depth, "(b t) d -> b t d", t=t)
            # x_rs02_depth = self.rs02_depth_patch_embedding(x_rs02_depth)

            # simple 2dcnn
            if self.rs02_depth_embedding_dim != 0:
                rs02_depth = rearrange(rs02_depth, "b t f c h w -> (b t) (c f) h w", f=1)
                x_rs02_depth = self.rs02_depth_patch_embedding(rs02_depth).squeeze()
                x_rs02_depth = rearrange(x_rs02_depth, "(b t) c -> b t c", t=t)

                if self.use_substitute_emb:
                    assert exist_data_rs02_depth.shape[2] == 1
                    _exist_data_rs02_depth = exist_data_rs02_depth[:, :, 0]
                    x_rs02_depth[torch.logical_not(_exist_data_rs02_depth)] = self.rs02_depth_null.repeat(
                        x_rs02_depth[torch.logical_not(_exist_data_rs02_depth)].shape[0], 1
                    )

                if self.add_defect_info:
                    rs02_exist_per = (
                        (exist_data_rs02_depth.sum(2) / exist_data_rs02_depth.shape[2])
                        .unsqueeze(-1)
                        .repeat(1, 1, self.rs02_depth_exist.shape[0])
                    )
                    x_rs02_depth = (
                        x_rs02_depth
                        + rs02_exist_per * self.rs02_depth_exist
                        + (1 - rs02_exist_per) * self.rs02_depth_defect
                    )
                x_list.append(x_rs02_depth)
                name_list.append("rs02_depth")

        elif self.use_cnn_feature:
            x_kinect_depth = self.kinect_depth_patch_embedding(kinect_depth)
            # add defect info
            if self.add_defect_info:
                if exist_data_kinect_depth.dtype == torch.bool:
                    exist_data_kinect_depth = exist_data_kinect_depth.to(torch.int)
                exist_data_kinect_depth = exist_data_kinect_depth.unsqueeze(-1).repeat(
                    1, 1, self.kinect_depth_exist.shape[0]
                )
                x_kinect_depth = (
                    x_kinect_depth
                    + exist_data_kinect_depth * self.kinect_depth_exist
                    + (1 - exist_data_kinect_depth) * self.kinect_depth_defect
                )
            x_list.append(x_kinect_depth)
            name_list.append("kinect_depth")
        else:
            kinect_depth = rearrange(kinect_depth, "b t f c h w -> (b t) (f c h w)")
            x_kinect_depth = self.kinect_depth_patch_embedding(kinect_depth)
            x_kinect_depth = rearrange(x_kinect_depth, "(b t) d -> b t d", t=t)
            x_list.append(x_kinect_depth)
            name_list.append("kinect_depth")
            # rs02_depth = rearrange(rs02_depth, "b t f c h w -> (b t) (f c h w)")
            # x_rs02_depth = self.rs02_depth_patch_embedding(rs02_depth)
            # x_rs02_depth = rearrange(x_rs02_depth, "(b t) d -> b t d", t=t)
            # x_list.append(x_rs02_depth)

        if self.cutout_p > 0 and self.training:
            name_list = np.array(name_list)
            n_target = int(b * self.cutout_p)
            target_batch = torch.randperm(b)[:n_target]
            select_data = torch.randint(0, 2, (n_target,))
            keypoint_ind = np.argmax(name_list == "keypoint")
            kinect_ind = np.argmax(name_list == "kinect_depth")

            for i in range(n_target):
                if select_data[i] == 0:
                    x_list[keypoint_ind][target_batch[i]] = torch.zeros_like(
                        x_list[keypoint_ind][target_batch[i]]
                    )

                elif select_data[i] == 1:
                    x_list[kinect_ind][target_batch[i]] = torch.zeros_like(
                        x_list[kinect_ind][target_batch[i]]
                    )

        if self.concat:
            return torch.concat(x_list, dim=2)
        else:
            return x_list
        # return self.ff_embedding(torch.concat(x_list, dim=2))


class OpenPackBase(nn.Module):
    def __init__(self, cfg, num_classes=11, concat=True):
        super().__init__()
        self.embedding = Embedding(cfg, num_classes, concat)
        self.set_param(cfg)
        self.num_classes = num_classes

    def set_param(self, cfg):
        # assert (
        #     cfg.model.imu_dim
        #     + cfg.model.keypoint_dim
        #     + cfg.model.e4acc_dim
        #     + cfg.model.ht_dim
        #     + cfg.model.printer_dim
        #     + cfg.model.kinect_depth_dim
        #     + cfg.model.rs02_depth_dim
        #     == cfg.model.dim
        # )
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
        self.kinect_depth_embedding_dim = cfg.model.kinect_depth_dim
        self.kinect_depth_input_dim = cfg.dataset.stream.kinect_depth_dim
        self.kinect_depth_image_size = cfg.model.image_size
        self.rs02_depth_embedding_dim = cfg.model.rs02_depth_dim
        self.rs02_depth_input_dim = cfg.dataset.stream.rs02_depth_dim
        self.rs02_depth_image_size = cfg.model.image_size
        self.use_pretrained_resnet = cfg.model.use_pretrained_resnet
        self.st_gcn = cfg.model.st_gcn
        self.resnet = cfg.model.resnet
        self.use_cnn_feature = cfg.model.use_cnn_feature


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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.transformer(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
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
            # self.conv = nn.Sequential(
            #     nn.Conv1d(dim, hidden_dim, kernel_size=1, padding="same"),
            #     nn.GELU(),
            #     nn.Dropout(dropout),
            #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_k, padding="same"),
            #     nn.GELU(),
            #     nn.Conv1d(hidden_dim, dim, kernel_size=1, padding="same"),
            #     nn.Dropout(dropout),
            # )
            self.conv = nn.Sequential(
                nn.Conv1d(dim, hidden_dim, kernel_size=1, padding="same"),
                nn.GroupNorm(10, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_k, padding="same"),
                nn.GroupNorm(10, hidden_dim),
                nn.GELU(),
                # SE(dim, hidden_dim),
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


class myB2TTransformer(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.dropout = nn.Dropout(self.emb_dropout_p)
        self.transformer = B2TTransformer(
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
        feat = self.ln(x)
        return self.linear_head(feat), feat


class B2TTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, conv_k=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        nn.LayerNorm(dim),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                        nn.LayerNorm(dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ln1, ff, ln2 in self.layers:
            input_x = x
            x = attn(x) + x
            x = ln1(x)
            x = ff(x) + x + input_x
            x = ln2(x)
        return x


class myB2TConvTransformer(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.dropout = nn.Dropout(self.emb_dropout_p)
        self.transformer = B2TConvTransformer(
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
        feat = self.ln(x)
        return self.linear_head(feat), feat


class B2TConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, conv_k=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        DConvAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        nn.LayerNorm(dim),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                        nn.LayerNorm(dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ln1, ff, ln2 in self.layers:
            input_x = x
            x = attn(x) + x
            x = ln1(x)
            x = ff(x) + x + input_x
            x = ln2(x)
        return x


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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x += self.pos_embedding[:, :t]
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.depth // 2:
                x, inner_logit = self.inner_module(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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


class myB2TConvTransformerAVEC(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = B2TConvTransformer(
            self.imu_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_imu = InterResModule(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = B2TConvTransformer(
            self.keypoint_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_keypoint = InterResModule(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = B2TConvTransformer(
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
        self.layers = B2TConvTransformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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


class myB2TConvTransformerAVECplusLSTM(OpenPackBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = B2TConvTransformer(
            self.imu_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_imu = InterResModuleLSTM(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = B2TConvTransformer(
            self.keypoint_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_keypoint = InterResModuleLSTM(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = B2TConvTransformer(
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
        self.layers = B2TConvTransformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln = nn.LayerNorm(self.dim)
        self.lstm = nn.LSTM(
            self.dim, self.dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        self.linear_head = nn.Linear(self.dim * 2, num_classes)

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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
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