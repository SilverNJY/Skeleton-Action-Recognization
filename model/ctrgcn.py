import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import clip
from Text_Prompt import *
from tools import *
from einops import rearrange, repeat
from model.modules import *
from model.lib import ST_RenovateNet


head_index = [0, 1, 2, 3, 4]
arm_index  = [5, 6, 7, 8, 9, 10]
leg_index  = [13, 14, 15, 16]
hip_index  = [11, 12]
body_parts = [head_index, arm_index, leg_index, hip_index]


def get_used_joint_mask(x, used_joints_list):

    NM, C, T, V = x.shape 
    N = len(used_joints_list)

    mask = 0.0001*torch.ones((NM, C, T, V), dtype=torch.float32)
    
    # 将 num_joints_list 指示的位置设为 1
    for n in range(NM):
        i = n
        if n>=N:
            i = n - N
        for v in used_joints_list[i]:
            mask[n, :, :, v] = 1
    # mask[:, :, :, num_joints_list] = 1
    return mask


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)


class Model_lst_4part_uav(nn.Module):

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

    def __init__(
        self,
        num_class=155,
        num_point=17,
        num_person=2,
        num_frame=128,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,
        adaptive=True,
        head=["ViT-B/32"],
        k=0,
        cl_mode=None,
        multi_cl_weights=[1, 1, 1, 1],
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

        A = self.graph.A # 3,17,17
        self.A_vector = self.get_A(graph, k).float()

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.num_frame = num_frame
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.base_channel = base_channel
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])

        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])

        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
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

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def get_ST_Multi_Level_cl_output(
        self, x, feat_low, feat_mid, feat_high, feat_fin, label
    ):
        logits = self.fc(x)
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
        return logits, cl_loss

    def forward(self, x, label=None, get_cl_loss=None, used_parts_list=[]):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x, _ = self.l1(x)
        feat_low = x.clone()

        x, _ = self.l2(x)
        x, _ = self.l3(x)
        x, _ = self.l4(x)
        x, _ = self.l5(x)
        feat_mid = x.clone()

        x, _ = self.l6(x)
        x, _ = self.l7(x)
        x, _ = self.l8(x)
        feat_high = x.clone()

        x, _ = self.l9(x)
        x, graph = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([0,1,2,3,4]).long() # 2,3,20
        hand_list = torch.Tensor([5,6,7,8,9,10]).long() # 4,5,6,7,8,9,10,11,21,22,23,24
        foot_list = torch.Tensor([11,12,13,14,15,16]).long() # 12,13,14,15,16,17,18,19
        hip_list = torch.Tensor([5,6,11,12]).long() # 0,1,2,12,16
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)

        x = self.drop_out(x)
        graph2 = graph.view(N, M, -1, c_new, V, V)
        # graph4 = torch.einsum('n m k c u v, n m k c v l -> n m k c u l', graph2, graph2)
        graph2 = graph2.view(N, M, -1, c_new, V, V).mean(1).mean(2).view(N, -1)
        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            logits, cl_loss = self.get_ST_Multi_Level_cl_output(
                x, feat_low, feat_mid, feat_high, feat_fin, label
            )
            return logits, cl_loss, feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]

        return self.fc(x), graph2, feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_uav_bone(nn.Module):

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

    def __init__(
        self,
        num_class=155,
        num_point=17,
        num_person=2,
        num_frame=128,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,
        adaptive=True,
        head=["ViT-B/32"],
        k=1,
        cl_mode=None,
        multi_cl_weights=[1, 1, 1, 1],
        cl_version="V0",
        pred_threshold=0,
        use_p_map=True,
    ):
        super(Model_lst_4part_uav_bone, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,17,17
        self.A_vector = self.get_A(graph, k).float()

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.num_frame = num_frame
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.base_channel = base_channel
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])

        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
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

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def get_ST_Multi_Level_cl_output(
        self, x, feat_low, feat_mid, feat_high, feat_fin, label
    ):
        logits = self.fc(x)
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
        return logits, cl_loss

    def forward(self, x, label=None, get_cl_loss=None):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
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
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([0,1,2,3,4]).long() 
        hand_list = torch.Tensor([5,6,7,8,9,10]).long() 
        foot_list = torch.Tensor([13,14,15,16]).long() 
        hip_list = torch.Tensor([11,12]).long() 
        # head_list = torch.Tensor([2,3]).long()
        # hand_list = torch.Tensor([4,5,6,7,8,9,10,11,20,22,23,24]).long()
        # foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        # hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)

        x = self.drop_out(x)
        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return (
                self.get_ST_Multi_Level_cl_output(
                    x, feat_low, feat_mid, feat_high, feat_fin, label
                ),
                feature_dict,
                self.logit_scale,
                [head_feature, hand_feature, hip_feature, foot_feature],
            )
        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_uav_used_parts(nn.Module):

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

    def __init__(self, num_class=155, num_point=17, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=0, cl_mode=None, multi_cl_weights=None):
        super(Model_lst_4part_uav_used_parts, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,17,17
        self.A_vector = self.get_A(graph, k).float()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])

        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])

        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        if self.cl_mode is not None:
            self.build_cl_blocks()

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label=None, get_cl_loss=False, used_joints_tag=[], istrain=False):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        feat_low = x.clone()
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        if istrain:
            used_joints_list = [[j for j, val in enumerate(row) if val == 1] for row in used_joints_tag]
            useful_mask = get_used_joint_mask(x, used_joints_list)
            x = x*useful_mask.to(x.device)
        feat_mid = x.clone()
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()
        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([0,1,2,3,4]).long() # 2,3,20
        hand_list = torch.Tensor([5,6,7,8,9,10]).long() # 4,5,6,7,8,9,10,11,21,22,23,24
        foot_list = torch.Tensor([11,12,13,14,15,16]).long() # 12,13,14,15,16,17,18,19
        hip_list = torch.Tensor([5,6,11,12]).long() # 0,1,2,12,16
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)

        x = self.drop_out(x)
        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_uav_bone_used_parts(nn.Module):
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

    def __init__(self, num_class=155, num_point=17, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, head=['ViT-B/32'], k=1):
        super(Model_lst_4part_uav_bone_used_parts, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,17,17
        self.A_vector = self.get_A(graph, k).float()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])

        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        if self.cl_mode is not None:
            self.build_cl_blocks()

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x, used_parts_list=[], istrain=False):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        feat_low = x.clone()
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        if istrain:
            useful_mask = get_used_joint_mask(x, used_parts_list)
            x = x*useful_mask.to(x.device) 
        x = self.l5(x)
        feat_mid = x.clone()
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()
        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([0,1,2,3,4]).long() 
        hand_list = torch.Tensor([5,6,7,8,9,10]).long() 
        foot_list = torch.Tensor([13,14,15,16]).long() 
        hip_list = torch.Tensor([11,12]).long() 

        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)

        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]
