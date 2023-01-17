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
from cnn.resnet import generate_r3d_model
from cnn.resnet2d import my_resnet
from cnn.resnet2d_cifar import my_resnet_cifar
from cnn.resnet2d_vit import my_resnet_vit
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor


def get_inplanes(embedding_dim, n_layers):
    if n_layers == 1:
        return [embedding_dim, 1, 1, 1]
    if n_layers == 2:
        return [embedding_dim // 2, embedding_dim, 1, 1]
    if n_layers == 3:
        return [embedding_dim // 4, embedding_dim // 2, embedding_dim, 1]
    if n_layers == 4:
        return [embedding_dim // 8, embedding_dim // 4, embedding_dim // 2, embedding_dim]


# TODO  class Embedding(nn.Module)みたいなのを作って、以下のOpenPackBase内のメンバにするー＞グループ化をさせる
class Embedding(nn.Module):
    def __init__(self, cfg, num_classes=11):
        super().__init__()
        self.set_param(cfg)

        self.imu_patch_embedding = nn.Linear(self.imu_input_dim, self.imu_embedding_dim)

        if self.st_gcn:
            if cfg.st_gcn.load_pretrain:
                chk_dir = "/workspace/pretrain_model/st_gcn/" + cfg.st_gcn.issue + "/checkpoints_k0/*.ckpt"
                chk_path = glob.glob(chk_dir)[0]
                plmodel = STGCNPL.load_from_checkpoint(chk_path, cfg=cfg)
                if cfg.st_gcn.freeze:
                    plmodel.freeze()
                self.keypoint_patch_embedding = plmodel.net.net
                del plmodel
            else:
                Ks = 3
                Kt = 3
                A = optorch.models.keypoint.get_adjacency_matrix(layout="MSCOCO", hop_size=Ks - 1)
                if cfg.st_gcn.type == "vit":
                    self.keypoint_patch_embedding = STGCN4Seg_vit(
                        2, cfg.st_gcn.out_channels, Ks=Ks, Kt=Kt, A=A
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
                    inplanes=get_inplanes(self.kinect_depth_embedding_dim, cfg.resnet.n_layers),
                    n_input_channel=1,
                    n_layers=cfg.resnet.n_layers,
                    # norm_layer=nn.Identity,
                )
                self.rs02_depth_patch_embedding = my_resnet(
                    inplanes=get_inplanes(self.kinect_depth_embedding_dim, cfg.resnet.n_layers),
                    n_input_channel=3,
                    n_layers=cfg.resnet.n_layers,
                    # norm_layer=nn.Identity,
                )
            if cfg.resnet.type == "vit":
                self.kinect_depth_patch_embedding = my_resnet_vit(
                    inplanes=get_inplanes(self.kinect_depth_embedding_dim, cfg.resnet.n_layers),
                    n_input_channel=1,
                    n_layers=cfg.resnet.n_layers,
                    num_groups=5
                    # norm_layer=nn.Identity,
                )
                self.rs02_depth_patch_embedding = my_resnet_vit(
                    inplanes=get_inplanes(self.rs02_depth_embedding_dim, cfg.resnet.n_layers),
                    n_input_channel=3,
                    n_layers=cfg.resnet.n_layers,
                    num_groups=5
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
        assert (
            cfg.model.imu_dim
            + cfg.model.keypoint_dim
            + cfg.model.ht_dim
            + cfg.model.printer_dim
            + cfg.model.kinect_depth_dim
            + cfg.model.rs02_depth_dim
            == cfg.model.dim
        )
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
        self.resnet_cifar = cfg.model.resnet_cifar
        self.mbconv = cfg.model.mbconv
        self.use_cnn_feature = cfg.model.use_cnn_feature
        self.cutout_p = cfg.train.dataaug.cutout_p

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
        b = imu.shape[0]
        t = imu.shape[1]
        x_list = []
        name_list = []

        imu = rearrange(imu, "b t f d -> b t (f d)")
        x_imu = self.imu_patch_embedding(imu)
        x_list.append(x_imu)
        name_list.append("imu")

        if self.st_gcn:
            keypoint = rearrange(keypoint, "b t f d n -> (b t) d f n")
            x_keypoint = self.keypoint_patch_embedding(keypoint)
            x_keypoint = rearrange(x_keypoint, "(b t) c f d -> b t (c f d)", t=t)

            # GAP版
            # x_keypoint = rearrange(x_keypoint, "(b t) d -> b t d", t=t)
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

        return x_imu, x_keypoint, x_kinect_depth, x_ht, x_printer


class OpenPackBase(nn.Module):
    def __init__(self, cfg, num_classes=11):
        super().__init__()
        self.set_param(cfg)
        self.embedding = Embedding(cfg, num_classes)

    def set_param(self, cfg):
        assert (
            cfg.model.imu_dim
            + cfg.model.keypoint_dim
            + cfg.model.ht_dim
            + cfg.model.printer_dim
            + cfg.model.kinect_depth_dim
            + cfg.model.rs02_depth_dim
            == cfg.model.dim
        )
        assert not (cfg.model.use_substitute_image and cfg.model.use_substitute_emb)
        assert not (cfg.model.resnet and cfg.model.use_cnn_feature)

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


class myTransformerTwoStep(OpenPackBase):
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
        use_pe=True,
        *args,
        **kargs,
    ):
        from vit_pytorch.vit import Transformer

        super().__init__(num_classes=num_classes, *args, **kargs)
        self.use_pe = use_pe

        self.dropout_imu = nn.Dropout(emb_dropout)
        self.dropout_keypoint = nn.Dropout(emb_dropout)
        self.dropout_kinect = nn.Dropout(emb_dropout)
        self.dropout_all = nn.Dropout(emb_dropout)

        self.transformer_imu = Transformer(self.imu_embedding_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer_keypoint = Transformer(
            self.keypoint_embedding_dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.transformer_kinect = Transformer(
            self.kinect_depth_embedding_dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.transformer_all = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        if use_pe:
            self.pos_embedding_imu = nn.Parameter(torch.randn(1, num_patches + 1, self.imu_embedding_dim))
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_kinect = nn.Parameter(
                torch.randn(1, num_patches + 1, self.kinect_depth_embedding_dim)
            )
            self.pos_embedding_all = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.to_latent = nn.Identity()
        self.linear_head_imu = nn.Sequential(
            nn.LayerNorm(self.imu_embedding_dim), nn.Linear(self.imu_embedding_dim, num_classes)
        )
        self.linear_head_keypoint = nn.Sequential(
            nn.LayerNorm(self.keypoint_embedding_dim), nn.Linear(self.keypoint_embedding_dim, num_classes)
        )
        self.linear_head_kinect = nn.Sequential(
            nn.LayerNorm(self.kinect_depth_embedding_dim),
            nn.Linear(self.kinect_depth_embedding_dim, num_classes),
        )
        self.linear_head_all = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        # print("!!! init xavier_normal_ !!!")
        # for n, p in self.named_parameters():
        #     if p.dim() >= 2:
        #         nn.init.xavier_normal_(p)

    def _forward_first(
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
        assert self.rs02_depth_embedding_dim == 0
        t = imu.shape[1]
        x_imu, x_keypoint, x_kinect, x_ht, x_printer = self.embedding(
            imu,
            keypoint,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
            x_keypoint += self.pos_embedding_keypoint[:, :t]
            x_kinect += self.pos_embedding_kinect[:, :t]
        x_imu = self.dropout_imu(x_imu)
        x_keypoint = self.dropout_keypoint(x_keypoint)
        x_kinect = self.dropout_kinect(x_kinect)

        x_imu = self.transformer_imu(x_imu)
        x_keypoint = self.transformer_keypoint(x_keypoint)
        x_kinect = self.transformer_kinect(x_kinect)

        return x_imu, x_keypoint, x_kinect, x_ht, x_printer

    def forward_pretrain(
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
        x_imu, x_keypoint, x_kinect, x_ht, x_printer = self._forward_first(
            imu,
            keypoint,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        x_imu = self.linear_head_imu(x_imu)
        x_keypoint = self.linear_head_keypoint(x_keypoint)
        x_kinect = self.linear_head_kinect(x_kinect)

        return x_imu, x_keypoint, x_kinect

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
        x_imu, x_keypoint, x_kinect, x_ht, x_printer = self._forward_first(
            imu,
            keypoint,
            ht,
            printer,
            kinect_depth,
            rs02_depth,
            exist_data_kinect_depth,
            exist_data_rs02_depth,
        )
        x = torch.concat([x_imu, x_keypoint, x_kinect, x_ht, x_printer], dim=2)
        if self.use_pe:
            x += self.pos_embedding_all[:, :t]
        x = self.dropout_all(x)
        x = self.transformer_all(x)
        x = self.to_latent(x)

        # TODO MLP headは一個を共有すればいい？別々に用意した方がいい？
        return self.linear_head_all(x)
