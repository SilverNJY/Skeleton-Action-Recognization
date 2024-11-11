import torch.nn.functional as F
import torch.nn as nn
from importlib_metadata import requires
import torch
from torch import einsum, positive
import math
import random
from torch import linalg as LA

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
    def __init__(
        self,
        in_channels=128,
        out_channels=256,
        mem_size=512,
        positive_num=8,
        negative_num=32,
        T=0.8,
        class_num=155,
        label_all=[],
    ):
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
        nn.init.normal_(self.trans.weight, 0, math.sqrt(2.0 / class_num))
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

        all_pairs = einsum("n c, m c -> n m", f_normed, self.Bank)
        bank_label = self.label_all.to(label.device)  # mem_size

        positive_mask = (label.view(n, 1) == bank_label.view(1, -1)).view(
            n, self.mem_size
        )  # n mem_size
        negative_mask = 1 - positive_mask.float()

        positive_mask = positive_mask * self.bank_flag
        negative_mask = negative_mask * self.bank_flag

        combined_pairs_list = []

        for i in range(n):
            if (positive_mask[i].sum(dim=-1) < self.positive_num) or (
                negative_mask[i].sum(dim=-1) < self.negative_num
            ):
                continue
            positive_pairs = torch.masked_select(
                all_pairs[i], mask=positive_mask[i].bool()
            ).view(-1)
            positive_pairs_hard = positive_pairs.sort(dim=-1, descending=False)[0][
                : self.positive_num
            ].view(1, self.positive_num, 1)

            negative_pairs = torch.masked_select(
                all_pairs[i], mask=negative_mask[i].bool()
            ).view(-1)
            negative_pairs_hard = (
                negative_pairs.sort(dim=-1, descending=True)[0][: self.negative_num]
                .view(1, 1, self.negative_num)
                .expand(-1, self.positive_num, -1)
            )

            idx = random.sample(list(range(len(negative_pairs))), k=self.negative_num)
            negative_pairs_random = (
                negative_pairs[idx]
                .view(1, 1, self.negative_num)
                .expand(-1, self.positive_num, -1)
            )

            combined_pairs_hard2hard = torch.cat(
                [positive_pairs_hard, negative_pairs_hard], -1
            ).view(self.positive_num, -1)
            combined_pairs_hard2random = torch.cat(
                [positive_pairs_hard, negative_pairs_random], -1
            ).view(self.positive_num, -1)
            combined_pairs = torch.cat(
                [combined_pairs_hard2hard, combined_pairs_hard2random], 0
            )
            combined_pairs_list.append((combined_pairs))

        if len(combined_pairs_list) == 0:
            return torch.zeros(1, device=f.device)

        combined_pairs = torch.cat(combined_pairs_list, 0)
        combined_label = torch.zeros(combined_pairs.size(0), device=f.device).long()
        loss = self.cross_entropy(combined_pairs / self.T, combined_label)

        return loss


def get_loss_func(loss_func, loss_args):
    if loss_func == "LabelSmoothingCrossEntropy":
        loss = LabelSmoothingCrossEntropy(
            smoothing=loss_args["smoothing"], temperature=loss_args["temperature"]
        )
    elif loss_func == "CrossEntropy":
        loss = nn.CrossEntropyLoss()
    elif loss_func == "CE_MBMMD":
        loss = CE_MBMMD(
            CrossEntropy=nn.CrossEntropyLoss(),
            MMDLoss=MMDLoss(),
            weights=loss_args["weights"],
        )
    elif loss_func == "LSCE_MBMMD":
        loss = LSCE_MBMMD(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(
                smoothing=loss_args["smoothing"], temperature=loss_args["temperature"]
            ),
            MMDLoss=MMDLoss(),
            weights=loss_args["weights"],
        )
    elif loss_func == "InfoGCN_Loss":
        loss = InfoGCN_Loss(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(
                smoothing=loss_args["smoothing"], temperature=loss_args["temperature"]
            ),
            weights=loss_args["weights"],
            class_num=loss_args["class_num"],
            out_channels=loss_args["out_channels"],
            gain=loss_args["gain"],
        )
    elif loss_func == "InfoGCN_Loss_MBMMD":
        loss = InfoGCN_Loss_MBMMD(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(
                smoothing=loss_args["smoothing"], temperature=loss_args["temperature"]
            ),
            MMDLoss=MMDLoss(),
            weights=loss_args["weights"],
            class_num=loss_args["class_num"],
            out_channels=loss_args["out_channels"],
            gain=loss_args["gain"],
        )
    elif loss_func == "LSCE_GROUP":
        loss = LSCE_GROUP(
            LSCE=LabelSmoothingCrossEntropy(
                smoothing=loss_args["smoothing"], temperature=loss_args["temperature"]
            )
        )
    elif loss_func == "LSCE_MBMMD_GROUP":
        loss = LSCE_MBMMD_GROUP(
            LSCE_GROUP=LSCE_GROUP(
                LSCE=LabelSmoothingCrossEntropy(
                    smoothing=loss_args["smoothing"],
                    temperature=loss_args["temperature"],
                )
            ),
            MMDLoss=MMDLoss(),
            weights=loss_args["weights"],
        )
    elif loss_func == "InfoGCN_Loss_GROUP":
        loss = InfoGCN_Loss_GROUP(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(
                smoothing=loss_args["smoothing"], temperature=loss_args["temperature"]
            ),
            weights=loss_args["weights"],
            class_num=loss_args["class_num"],
            out_channels=loss_args["out_channels"],
            gain=loss_args["gain"],
        )
    elif loss_func == "InfoGCN_Loss_MBMMD_GROUP":
        loss = InfoGCN_Loss_MBMMD_GROUP(
            LabelSmoothingCrossEntropy=LabelSmoothingCrossEntropy(
                smoothing=loss_args["smoothing"], temperature=loss_args["temperature"]
            ),
            MMDLoss=MMDLoss(),
            weights=loss_args["weights"],
            class_num=loss_args["class_num"],
            out_channels=loss_args["out_channels"],
            gain=loss_args["gain"],
        )
    else:
        print("Loss Not Included")
        loss = None

    return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        if isinstance(x, tuple):
            x = x[0]
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(
            dim=-1
        )
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MMDLoss(nn.Module):
    """
    Params:
    source: (n * len(x))
    target: (m * len(y))
    kernel_mul:
    kernel_num:
    fix_sigma:
    Return:
    loss: MMD loss
    """

    def __init__(
        self, kernel_type="rbf", kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs
    ):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == "linear":
            return self.linear_mmd2(source, target)
        elif self.kernel_type == "rbf":
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source,
                target,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma,
            )
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class CE_MBMMD(nn.Module):
    def __init__(self, CrossEntropy, MMDLoss, weights=[1.0, 0.1]):
        super(CE_MBMMD, self).__init__()

        assert len(weights) == 2

        self.CE = CrossEntropy
        self.MMD = MMDLoss
        self.weights = weights

    def forward(self, x_tuple, target):

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(
            x_tuple[-2].view(N * M, C).contiguous(),
            x_tuple[-1].view(N * M, C).contiguous(),
        )

        return self.weights[0] * self.CE(x_tuple[0], target) + self.weights[1] * mpmmd


class LSCE_MBMMD(nn.Module):
    def __init__(self, LabelSmoothingCrossEntropy, MMDLoss, weights=[1.0, 0.1]):
        super(LSCE_MBMMD, self).__init__()

        assert len(weights) == 2

        self.LSCE = LabelSmoothingCrossEntropy
        self.MMD = MMDLoss
        self.weights = weights

    def forward(self, x_tuple, target):

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(
            x_tuple[-2].view(N * M, C).contiguous(),
            x_tuple[-1].view(N * M, C).contiguous(),
        )

        return self.weights[0] * self.LSCE(x_tuple[0], target) + self.weights[1] * mpmmd


def get_mmd_loss_infogcn(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid])
    return mmd_loss, l2_z_mean, z_mean[y_valid]


class InfoGCN_Loss(nn.Module):
    def __init__(
        self,
        LabelSmoothingCrossEntropy,
        weights=[1.0, 0.1, 0.0001],
        class_num=26,
        out_channels=256,
        gain=3,
    ):
        super(InfoGCN_Loss, self).__init__()

        assert len(weights) == 3

        self.LSCE = LabelSmoothingCrossEntropy
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(
            x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num
        )

        return (
            self.weights[0] * self.LSCE(x_tuple[0], target)
            + self.weights[1] * info_mmd_loss
            + self.weights[2] * l2_z_mean
        )


class InfoGCN_Loss_MBMMD(nn.Module):
    def __init__(
        self,
        LabelSmoothingCrossEntropy,
        MMDLoss,
        weights=[1.0, 0.1, 0.0001, 0.1],
        class_num=26,
        out_channels=256,
        gain=3,
    ):
        super(InfoGCN_Loss_MBMMD, self).__init__()

        assert len(weights) == 4

        self.LSCE = LabelSmoothingCrossEntropy
        self.MMD = MMDLoss
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(
            x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num
        )

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(
            x_tuple[-2].view(N * M, C).contiguous(),
            x_tuple[-1].view(N * M, C).contiguous(),
        )

        return (
            self.weights[0] * self.LSCE(x_tuple[0], target)
            + self.weights[1] * info_mmd_loss
            + self.weights[2] * l2_z_mean
            + self.weights[-1] * mpmmd
        )


class LSCE_GROUP(nn.Module):
    def __init__(self, LSCE):
        super().__init__()
        self.LSCE = LSCE

    def forward(self, x, target, target_person):
        N, M, C = x[1].size()
        return self.LSCE(x[0], target) + self.LSCE(
            x[1].view(N * M, C), target_person.view(N * M)
        )


class LSCE_MBMMD_GROUP(nn.Module):
    def __init__(self, LSCE_GROUP, MMDLoss, weights=[1.0, 0.1]):
        super(LSCE_MBMMD_GROUP, self).__init__()

        assert len(weights) == 2

        self.LSCE_GROUP = LSCE_GROUP
        self.MMD = MMDLoss
        self.weights = weights

    def forward(self, x_tuple, target, target_person):

        N, M, C = x_tuple[-1].size()

        mpmmd = self.MMD(
            x_tuple[-2].view(N * M, C).contiguous(),
            x_tuple[-1].view(N * M, C).contiguous(),
        )

        return (
            self.weights[0]
            * self.LSCE_GROUP((x_tuple[0], x_tuple[1]), target, target_person)
            + self.weights[1] * mpmmd
        )


class InfoGCN_Loss_GROUP(nn.Module):
    def __init__(
        self,
        LabelSmoothingCrossEntropy,
        weights=[1.0, 0.1, 0.0001],
        class_num=26,
        out_channels=256,
        gain=3,
    ):
        super(InfoGCN_Loss_GROUP, self).__init__()

        assert len(weights) == 3

        self.LSCE = LabelSmoothingCrossEntropy
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target, target_person):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(
            x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num
        )
        N, M, C = x_tuple[2].size()

        return (
            self.weights[0]
            * (
                self.LSCE(x_tuple[0], target)
                + self.LSCE(x_tuple[2].view(N * M, C), target_person.view(N * M))
            )
            + self.weights[1] * info_mmd_loss
            + self.weights[2] * l2_z_mean
        )


class InfoGCN_Loss_MBMMD_GROUP(nn.Module):
    def __init__(
        self,
        LabelSmoothingCrossEntropy,
        MMDLoss,
        weights=[1.0, 0.1, 0.0001, 0.1],
        class_num=26,
        out_channels=256,
        gain=3,
    ):
        super(InfoGCN_Loss_MBMMD_GROUP, self).__init__()

        assert len(weights) == 4

        self.LSCE = LabelSmoothingCrossEntropy
        self.MMD = MMDLoss
        self.weights = weights
        self.class_num = class_num
        self.z_prior = torch.empty(class_num, out_channels)
        nn.init.orthogonal_(self.z_prior, gain=gain)

    def forward(self, x_tuple, target, target_person):

        info_mmd_loss, l2_z_mean, z_mean = get_mmd_loss_infogcn(
            x_tuple[1], self.z_prior.to(x_tuple[1].device), target, self.class_num
        )

        N, M, C = x_tuple[-1].size()
        mpmmd = self.MMD(
            x_tuple[-2].view(N * M, C).contiguous(),
            x_tuple[-1].view(N * M, C).contiguous(),
        )

        Np, Mp, Cp = x_tuple[2].size()

        return (
            self.weights[0]
            * (
                self.LSCE(x_tuple[0], target)
                + self.LSCE(x_tuple[2].view(Np * Mp, Cp), target_person.view(Np * Mp))
            )
            + self.weights[1] * info_mmd_loss
            + self.weights[2] * l2_z_mean
            + self.weights[-1] * mpmmd
        )
