import math

import numpy as np
import torch
import torch.nn.functional as F
from tools import *
from torch import nn, einsum
from torch.autograd import Variable
from torch import linalg as LA

from .lib import ST_RenovateNet
from einops import rearrange, repeat

from .modules import *


def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]


class InfoGCN(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_frame=64,
        num_person=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,
        num_head=3,
        noise_ratio=0.1,
        k=0,
        gain=1,
        cl_mode=None,
        multi_cl_weights=None,
        cl_version="V0",
        pred_threshold=0,
        use_p_map=True,
    ):
        super(InfoGCN, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,17,17
        self.A_vector = self.get_A(graph, k).float()

        base_channel = 64
        self.base_channel = base_channel
        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(num_class, base_channel * 4)
        self.gain = gain
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        self.l1 = EncodingBlock(base_channel, base_channel, A)
        self.l2 = EncodingBlock(base_channel, base_channel, A)
        self.l3 = EncodingBlock(base_channel, base_channel, A)
        self.l4 = EncodingBlock(base_channel, base_channel * 2, A, stride=2)
        self.l5 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l6 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l7 = EncodingBlock(base_channel * 2, base_channel * 4, A, stride=2)
        self.l8 = EncodingBlock(base_channel * 4, base_channel * 4, A)
        self.l9 = EncodingBlock(base_channel * 4, base_channel * 4, A)
        self.fc = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_mu = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_logvar = nn.Linear(base_channel * 4, base_channel * 4)
        self.decoder = nn.Linear(base_channel * 4, num_class)
        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(
            self.fc_logvar.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        if self.cl_mode is not None:
            self.build_cl_blocks()
        if self.multi_cl_weights is None:
            self.multi_cl_weights = [0.1, 0.2, 0.5, 1]

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(
                self.base_channel,
                self.num_frame,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
            self.ren_mid = ST_RenovateNet(
                self.base_channel * 2,
                self.num_frame // 2,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
            self.ren_high = ST_RenovateNet(
                self.base_channel * 4,
                self.num_frame // 4,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
            self.ren_fin = ST_RenovateNet(
                self.base_channel * 4,
                self.num_frame // 4,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")

    def get_ST_Multi_Level_cl_output(
        self, x, feat_low, feat_mid, feat_high, feat_fin, label
    ):
        logits = self.decoder(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = (
            cl_low * self.multi_cl_weights[0]
            + cl_mid * self.multi_cl_weights[1]
            + cl_high * self.multi_cl_weights[2]
            + cl_fin * self.multi_cl_weights[3]
        )
        return cl_loss

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k)).float()

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x, label=None, get_cl_loss=False):
        N, C, T, V, M = x.size()
        x = rearrange(x, "n c t v m -> (n m t) v c", m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N * M * T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, : self.num_point]
        x = rearrange(x, "(n m t) v c -> n (m v c) t", m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, "n (m v c) t -> (n m) c t v", m=M, v=V).contiguous()
        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return (
                y_hat,
                z,
                self.get_ST_Multi_Level_cl_output(
                    z, feat_low, feat_mid, feat_high, feat_fin, label
                ),
            )

        return y_hat, z


class Model_lst_4part_uav(nn.Module):

    def __init__(
        self,
        num_class=155,
        num_point=17,
        num_frame=300,
        num_person=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,
        num_head=3,
        head=["ViT-B/32"],
        noise_ratio=0.1,
        k=0,
        gain=1,
        cl_mode=None,
        multi_cl_weights=None,
        cl_version="V0",
        pred_threshold=0,
        use_p_map=True,
    ):
        super(Model_lst_4part_uav, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,17,17
        self.A_vector = self.get_A(graph, k).float()

        base_channel = 64
        self.base_channel = base_channel
        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(num_class, base_channel * 4)
        self.gain = gain
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        self.l1 = EncodingBlock(base_channel, base_channel, A)
        self.l2 = EncodingBlock(base_channel, base_channel, A)
        self.l3 = EncodingBlock(base_channel, base_channel, A)
        self.l4 = EncodingBlock(base_channel, base_channel * 2, A, stride=2)
        self.l5 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l6 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l7 = EncodingBlock(base_channel * 2, base_channel * 4, A, stride=2)
        self.l8 = EncodingBlock(base_channel * 4, base_channel * 4, A)
        self.l9 = EncodingBlock(base_channel * 4, base_channel * 4, A)
        self.fc = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_mu = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_logvar = nn.Linear(base_channel * 4, base_channel * 4)
        self.decoder = nn.Linear(base_channel * 4, num_class)
        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(
            self.fc_logvar.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1, 5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256, 512))

        self.head = head
        if "ViT-B/32" in self.head:
            self.linear_head["ViT-B/32"] = nn.Linear(256, 512)
            conv_init(self.linear_head["ViT-B/32"])

        if "ViT-B/16" in self.head:
            self.linear_head["ViT-B/16"] = nn.Linear(256, 512)
            conv_init(self.linear_head["ViT-B/16"])
        if "ViT-L/14" in self.head:
            self.linear_head["ViT-L/14"] = nn.Linear(256, 768)
            conv_init(self.linear_head["ViT-L/14"])
        if "ViT-L/14@336px" in self.head:
            self.linear_head["ViT-L/14@336px"] = nn.Linear(256, 768)
            conv_init(self.linear_head["ViT-L/14@336px"])

        if "RN50x64" in self.head:
            self.linear_head["RN50x64"] = nn.Linear(256, 1024)
            conv_init(self.linear_head["RN50x64"])

        if "RN50x16" in self.head:
            self.linear_head["RN50x16"] = nn.Linear(256, 768)
            conv_init(self.linear_head["RN50x16"])

        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        if self.cl_mode is not None:
            self.build_cl_blocks()
        if self.multi_cl_weights is None:
            self.multi_cl_weights = [0.1, 0.2, 0.5, 1]

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(
                self.base_channel,
                self.num_frame,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
            self.ren_mid = ST_RenovateNet(
                self.base_channel * 2,
                self.num_frame // 2,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
            self.ren_high = ST_RenovateNet(
                self.base_channel * 4,
                self.num_frame // 4,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
            self.ren_fin = ST_RenovateNet(
                self.base_channel * 4,
                self.num_frame // 4,
                self.num_point,
                self.num_person,
                n_class=self.num_class,
                version=self.cl_version,
                pred_threshold=self.pred_threshold,
                use_p_map=self.use_p_map,
            )
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")

    def get_ST_Multi_Level_cl_output(
        self, x, feat_low, feat_mid, feat_high, feat_fin, label
    ):
        logits = self.decoder(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = (
            cl_low * self.multi_cl_weights[0]
            + cl_mid * self.multi_cl_weights[1]
            + cl_high * self.multi_cl_weights[2]
            + cl_fin * self.multi_cl_weights[3]
        )
        return cl_loss

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k)).float()

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x, label=None, get_cl_loss=False):
        N, C, T, V, M = x.size()
        x = rearrange(x, "n c t v m -> (n m t) v c", m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N * M * T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, : self.num_point]
        x = rearrange(x, "(n m t) v c -> n (m v c) t", m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, "n (m v c) t -> (n m) c t v", m=M, v=V).contiguous()
        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N, M, c_new, T // 4, V)
        head_list = torch.Tensor([0, 1, 2, 3, 4]).long()  # 2,3,20
        hand_list = torch.Tensor(
            [5, 6, 7, 8, 9, 10]
        ).long()  # 4,5,6,7,8,9,10,11,21,22,23,24
        foot_list = torch.Tensor(
            [11, 12, 13, 14, 15, 16]
        ).long()  # 12,13,14,15,16,17,18,19
        hip_list = torch.Tensor([5, 6, 11, 12]).long()  # 0,1,2,12,16
        head_feature = self.part_list[0](
            feature[:, :, :, :, head_list].mean(4).mean(3).mean(1)
        )
        hand_feature = self.part_list[1](
            feature[:, :, :, :, hand_list].mean(4).mean(3).mean(1)
        )
        foot_feature = self.part_list[2](
            feature[:, :, :, :, foot_list].mean(4).mean(3).mean(1)
        )
        hip_feature = self.part_list[3](
            feature[:, :, :, :, hip_list].mean(4).mean(3).mean(1)
        )

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)

        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return (
                y_hat,
                z,
                self.get_ST_Multi_Level_cl_output(
                    z, feat_low, feat_mid, feat_high, feat_fin, label
                ),
            )

        return y_hat, z, feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]
