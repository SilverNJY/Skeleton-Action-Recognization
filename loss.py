import torch.nn.functional as F
import torch.nn as nn
from importlib_metadata import requires
import torch
from torch import einsum, positive
import math
import random


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print("=========using KL Loss=and has temperature and * bz==========")
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


class InfoNCEGraph(nn.Module):
    def __init__(self, in_channels=128, out_channels=256, mem_size=512, positive_num=4, negative_num=16, T=0.8, class_num=155, label_all=[]):
        super(InfoNCEGraph, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mem_size = mem_size
        self.positive_num = positive_num
        self.negative_num = negative_num
        self.T = T
        self.trans = nn.Linear(in_channels, out_channels)
        self.Bank = nn.Parameter(
            torch.zeros((mem_size, out_channels)), requires_grad=False
        )
        self.label_all = torch.from_numpy(label_all)
        nn.init.normal_(self.trans.weight, 0, math.sqrt(2. / class_num))
        nn.init.zeros_(self.trans.bias)
        self.bank_flag = nn.Parameter(
            torch.zeros(len(self.label_all)), requires_grad=False
            ) 
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, f, label, input_index):
        # f: n c label: n
        n, _ = f.size()
        f = self.trans(f)
        f_norm = f.norm(dim=-1, p=2, keepdim=True)
        f_normed = f / f_norm
        self.Bank[input_index] = f_normed.detach()
        self.bank_flag[input_index] = 1

        all_pairs = einsum('n c, m c -> n m', f_normed, self.Bank)
        bank_label = self.label_all.to(label.device) # mem_size
        positive_mask = (label.view(n, 1) == bank_label.view(1, -1)).view(n, self.mem_size) # n mem_size
        negative_mask = (1-positive_mask.float())

        positive_mask = positive_mask * self.bank_flag
        negative_mask = negative_mask * self.bank_flag

        combined_pairs_list = []

        for i in range(n):
            if (positive_mask[i].sum(dim=-1) < self.positive_num) or (negative_mask[i].sum(dim=-1) < self.negative_num):
                continue
            positive_pairs = torch.masked_select(all_pairs[i], mask=positive_mask[i].bool()).view(-1)
            positive_pairs_hard = positive_pairs.sort(dim=-1, descending=False)[0][:self.positive_num].view(1, self.positive_num, 1)

            negative_pairs = torch.masked_select(all_pairs[i], mask=negative_mask[i].bool()).view(-1)
            negative_pairs_hard = negative_pairs.sort(dim=-1, descending=True)[0][:self.negative_num].view(1, 1, self.negative_num)\
                .expand(-1, self.positive_num, -1)

            idx = random.sample(list(range(len(negative_pairs))), k=self.negative_num)
            negative_pairs_random = negative_pairs[idx].view(1, 1, self.negative_num).expand(-1, self.positive_num, -1)

            combined_pairs_hard2hard = torch.cat([positive_pairs_hard, negative_pairs_hard], -1).view(self.positive_num, -1)
            combined_pairs_hard2random = torch.cat([positive_pairs_hard, negative_pairs_random], -1).view(self.positive_num, -1)
            combined_pairs = torch.cat([combined_pairs_hard2hard, combined_pairs_hard2random], 0)
            combined_pairs_list.append((combined_pairs))

        if len(combined_pairs_list) == 0:
            return torch.zeros(1, device=f.device)

        combined_pairs = torch.cat(combined_pairs_list, 0)
        combined_label = torch.zeros(combined_pairs.size(0), device=f.device).long()
        loss = self.cross_entropy(combined_pairs/self.T, combined_label)

        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, reduction="mean"):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            alpha = [1] * num_class
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
