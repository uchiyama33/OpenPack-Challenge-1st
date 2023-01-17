import glob

import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import torch
import torch.nn.functional as F
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
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from vit_pytorch.vit import Attention, FeedForward, PreNorm


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


class EmbeddingBase(nn.Module):
    def __init__(self, cfg, num_classes=11):
        super().__init__()
        self.set_param(cfg)

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

        self.num_patches = cfg.model.num_patches

        if cfg.model.imu_dim == -1:
            self.imu_inplanes = [64, 128, 256, 512]
            self.imu_num_groups = 32
            self.imu_embedding_dim = self.imu_inplanes[cfg.resnet1d.n_layers - 1]
        else:
            self.imu_inplanes = get_inplanes(cfg.model.imu_dim, cfg.resnet1d.n_layers)
            self.imu_num_groups = 5
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

        self.concat_dim = (
            self.imu_embedding_dim
            + self.keypoint_embedding_dim
            + self.e4acc_embedding_dim
            + self.ht_embedding_dim
            + self.printer_embedding_dim
            + self.kinect_depth_embedding_dim
            + self.rs02_depth_embedding_dim
        )


class EmbeddingStep(EmbeddingBase):
    def __init__(self, cfg, num_classes=11):
        super().__init__(cfg)

        # self.ff_embedding = PreNorm(
        #     self.concat_dim,
        #     FeedForwardEmbedding(
        #         self.concat_dim, cfg.model.dim, cfg.model.mlp_dim, dropout=cfg.model.emb_dropout
        #     ),
        # )

        if self.resnet1d:
            if cfg.resnet1d.type == "vit":
                self.imu_patch_embedding = my_resnet1d_vit(
                    inplanes=self.imu_inplanes,
                    n_input_channel=cfg.dataset.stream.imu_dim,
                    n_layers=cfg.resnet1d.n_layers,
                    num_groups=self.imu_num_groups
                    # norm_layer=nn.Identity,
                )
                self.e4acc_patch_embedding = my_resnet1d_vit(
                    inplanes=self.e4acc_inplanes,
                    n_input_channel=cfg.dataset.stream.e4acc_dim,
                    n_layers=cfg.resnet1d.n_layers,
                    num_groups=self.e4acc_num_groups
                    # norm_layer=nn.Identity,
                )
        else:
            self.imu_patch_embedding = nn.Linear(self.imu_input_dim, self.imu_embedding_dim)
            self.e4acc_patch_embedding = nn.Linear(self.e4acc_input_dim, self.e4acc_embedding_dim)

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

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
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
        else:
            imu = rearrange(imu, "b t f d -> b t (f d)")
            x_imu = self.imu_patch_embedding(imu)
            x_list.append(x_imu)
            name_list.append("imu")

            e4acc = rearrange(e4acc, "b t f d -> b t (f d)")
            x_e4acc = self.e4acc_patch_embedding(e4acc)
            x_list.append(x_e4acc)
            name_list.append("e4acc")

        if self.st_gcn:
            keypoint = rearrange(keypoint, "b t f d n -> (b t) d f n")
            x_keypoint = self.keypoint_patch_embedding(keypoint)
            x_keypoint = rearrange(x_keypoint, "(b t) d -> b t d", t=t)
        else:
            keypoint = rearrange(keypoint, "b t f d n -> b t (f d n)")
            x_keypoint = self.keypoint_patch_embedding(keypoint)
        x_list.append(x_keypoint)
        name_list.append("keypoint")

        x_ht = self.ht_patch_embedding(ht)
        x_list.append(x_ht)
        name_list.append("ht")
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

        return torch.concat(x_list, dim=2)
        # return self.ff_embedding(torch.concat(x_list, dim=2))


class EmbeddingChannel(EmbeddingBase):
    def __init__(self, cfg, num_classes=11):
        super().__init__(cfg)

        self.imu_patch_embeddings = nn.ModuleList(
            [
                nn.Linear(
                    self.imu_input_dim // cfg.dataset.stream.imu_dim * self.num_patches, self.concat_dim
                )
                for i in range(cfg.dataset.stream.imu_dim)
            ]
        )

        # self.keypoint_patch_embedding = nn.Linear(self.keypoint_input_dim * self.num_patches, self.concat_dim)
        self.keypoint_patch_embeddings = nn.ModuleList(
            [
                nn.Linear(
                    self.keypoint_input_dim // cfg.dataset.stream.keypoint_dim * 2 * self.num_patches,
                    self.concat_dim,
                )
                for i in range(cfg.dataset.stream.keypoint_dim)
            ]
        )

        self.ht_patch_embedding = nn.Linear(self.num_patches, self.concat_dim)
        self.printer_patch_embedding = nn.Linear(self.num_patches, self.concat_dim)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
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

        # imu = rearrange(imu, "b t f d -> b (t f d)")
        # x_imu = self.imu_patch_embedding(imu)
        # x_list.append(x_imu)
        # name_list.append("imu")

        imu = rearrange(imu, "b t f d -> b d (t f)")
        for l in range(imu.shape[1]):
            x_imu = self.imu_patch_embeddings[l](imu[:, l])
            x_list.append(x_imu)
            name_list.append("imu")

        # keypoint = rearrange(keypoint, "b t f d n -> b (t f d n)")
        # x_keypoint = self.keypoint_patch_embedding(keypoint)
        # x_list.append(x_keypoint)
        # name_list.append("keypoint")

        keypoint = rearrange(keypoint, "b t f d n -> b n (t f d)")
        for l in range(keypoint.shape[1]):
            x_keypoint = self.keypoint_patch_embeddings[l](keypoint[:, l])
            x_list.append(x_keypoint)
            name_list.append("keypoint")

        x_ht = self.ht_patch_embedding(ht.to(dtype=torch.float32))
        x_list.append(x_ht)
        name_list.append("ht")
        x_printer = self.printer_patch_embedding(printer.to(dtype=torch.float32))
        x_list.append(x_printer)
        name_list.append("printer")

        return torch.stack(x_list, dim=1)


class OpenPackTwoTowerBase(nn.Module):
    def __init__(self, cfg, num_classes=11):
        super().__init__()
        self.embedding_step = EmbeddingStep(cfg, num_classes)
        self.embedding_channel = EmbeddingChannel(cfg, num_classes)
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
            self.dim = self.embedding_step.concat_dim
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


class myTransformerTwoTower(OpenPackTwoTowerBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)

        self.dropout_step = nn.Dropout(self.emb_dropout_p)
        self.transformer_step = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        if self.use_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        self.dropout_channel = nn.Dropout(self.emb_dropout_p)
        self.transformer_channel = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln_channel = nn.LayerNorm(self.dim)
        self.fc_channel = nn.Linear(self.dim, self.num_patches)

        self.ln = nn.LayerNorm(self.dim + 31)
        self.gate = nn.Linear(self.dim + 31, 2)
        self.linear_head = nn.Linear(self.dim + 31, num_classes)

        # test
        # self.linear_head_test = nn.Linear(31, num_classes)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        ht,
        printer,
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_step = self.embedding_step(
            imu,
            keypoint,
            e4acc,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x_step += self.pos_embedding[:, :t]
        x_step = self.dropout_step(x_step)
        x_step = self.transformer_step(x_step)

        x_channel = self.embedding_channel(
            imu,
            keypoint,
            e4acc,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        x_channel = self.dropout_channel(x_channel)
        x_channel = self.transformer_channel(x_channel)
        x_channel = self.ln_channel(x_channel)
        x_channel = self.fc_channel(x_channel).transpose(1, 2)

        gate = F.softmax(self.gate(torch.concat([x_step, x_channel], dim=-1)), dim=-1)
        feat = self.ln(torch.concat([x_step * gate[:, :, [0]], x_channel * gate[:, :, [1]]], dim=-1))
        return self.linear_head(feat), feat

    # only channel test
    # def forward(
    #     self,
    #     imu,
    #     keypoint,
    #     e4acc,
    #     ht,
    #     printer,
    #     kinect_depth,
    #     rs02_depth,
    #     exist_data_kinect_depth,
    #     exist_data_rs02_depth,
    # ):
    #     t = imu.shape[1]

    #     x_channel = self.embedding_channel(
    #         imu,
    #         keypoint,
    #         e4acc,
    #         ht,
    #         printer,
    #         kinect_depth,
    #         rs02_depth,
    #         exist_data_kinect_depth,
    #         exist_data_rs02_depth,
    #     )
    #     x_channel = self.dropout_channel(x_channel)
    #     x_channel = self.transformer_channel(x_channel)
    #     x_channel = self.ln_channel(x_channel)
    #     x_channel = self.fc_channel(x_channel).transpose(1, 2)

    #     return self.linear_head_test(x_channel), None


class myTransformerTwoTowerTwoStep(OpenPackTwoTowerBase):
    def __init__(
        self,
        num_classes,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_step = nn.Dropout(self.emb_dropout_p)
        self.transformer_step = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        if self.use_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        self.dropout_channel = nn.Dropout(self.emb_dropout_p)
        self.transformer_channel = Transformer(
            self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln_channel = nn.LayerNorm(self.dim)
        self.fc_channel = nn.Linear(self.dim, self.num_patches)

        # self.dropout = nn.Dropout(self.emb_dropout_p)
        self.transformer = Transformer(
            self.dim + 31, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )

        self.ln = nn.LayerNorm(self.dim + 31)
        self.linear_head = nn.Linear(self.dim + 31, num_classes)

        # test
        # self.linear_head_test = nn.Linear(31, num_classes)

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        ht,
        printer,
        kinect_depth,
        rs02_depth,
        exist_data_kinect_depth,
        exist_data_rs02_depth,
    ):
        t = imu.shape[1]
        x_step = self.embedding_step(
            imu,
            keypoint,
            e4acc,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x_step += self.pos_embedding[:, :t]
        x_step = self.dropout_step(x_step)
        x_step = self.transformer_step(x_step)

        x_channel = self.embedding_channel(
            imu,
            keypoint,
            e4acc,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        x_channel = self.dropout_channel(x_channel)
        x_channel = self.transformer_channel(x_channel)
        x_channel = self.ln_channel(x_channel)
        x_channel = self.fc_channel(x_channel).transpose(1, 2)

        x = torch.concat([x_step, x_channel], dim=-1)
        # x = self.dropout(x)
        x = self.transformer(x)
        feat = self.ln(x)
        return self.linear_head(feat), feat
