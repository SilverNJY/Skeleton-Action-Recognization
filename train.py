#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from model.baseline import TextCLIP
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction
from tools import *
from Text_Prompt import *
from loss import *
from model.infogcn import get_mmd_loss

classes, num_text_aug, text_dict = uav_text_prompt_openai_pasta_pool_4part()
# text_list = text_prompt_openai_random()
text_list = uav_text_prompt_openai_random()


device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = torch.cuda.amp.GradScaler()

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition(".")
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            "Class %s cannot be found (%s)"
            % (class_str, traceback.format_exception(*sys.exc_info()))
        )


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description="Spatial Temporal Graph Convolution Network"
    )
    parser.add_argument(
        "--work-dir",
        default="./work_dir/uav",
        help="the work folder for storing results",
    )

    parser.add_argument("-model_saved_name", default="")
    parser.add_argument(
        "--config",
        default="./config/ctrgcn/k1.yaml",
        help="path to the configuration file",
    )

    # processor
    parser.add_argument("--phase", default="train", help="must be train or test")
    parser.add_argument(
        "--save-score",
        type=str2bool,
        default=False,
        help="if ture, the classification score will be stored",
    )

    # visulize and debug
    parser.add_argument("--seed", type=int, default=1, help="random seed for pytorch")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="the interval for printing messages (#iteration)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="the interval for storing models (#iteration)",
    )
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=30,
        help="the start epoch to save model (#iteration)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="the interval for evaluating models (#iteration)",
    )
    parser.add_argument(
        "--print-log", type=str2bool, default=True, help="print logging or not"
    )
    parser.add_argument(
        "--show-topk",
        type=int,
        default=[1, 5],
        nargs="+",
        help="which Top K accuracy will be shown",
    )

    # feeder
    parser.add_argument(
        "--feeder", default="feeder.feeder", help="data loader will be used"
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=1,
        help="the number of worker for data loader",
    )
    parser.add_argument(
        "--train-feeder-args",
        action=DictAction,
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--test-feeder-args",
        action=DictAction,
        default=dict(),
        help="the arguments of data loader for test",
    )

    # model
    parser.add_argument("--model", default=None, help="the model will be used")
    parser.add_argument(
        "--model-args", action=DictAction, default=dict(), help="the arguments of model"
    )
    parser.add_argument(
        "--weights", default=None, help="the weights for network initialization"
    )
    parser.add_argument("--wrapper", default=None, help="the model will be used")
    parser.add_argument("--wrapper_args", default=dict(), help="the arguments of model")
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )
    parser.add_argument(
        "--cl-mode",
        choices=["ST-Multi-Level"],
        default=None,
        help="mode of Contrastive Learning Loss",
    )
    parser.add_argument(
        "--cl-version",
        choices=["V0", "V1", "V2", "NO FN", "NO FP", "NO FN & FP"],
        default="V0",
        help="different way to calculate the cl loss",
    )
    parser.add_argument(
        "--pred_threshold",
        type=float,
        default=0.0,
        help="threshold to define the confident sample",
    )
    parser.add_argument(
        "--use_p_map",
        type=str2bool,
        default=True,
        help="whether to add (1 - p_{ik}) to constrain the auxiliary item",
    )
    parser.add_argument(
        "--start-cl-epoch", type=int, default=-1, help="epoch to optimize cl loss"
    )
    parser.add_argument(
        "--w-cl-loss", type=float, default=0.1, help="weight of cl loss"
    )
    parser.add_argument(
        "--w-multi-cl-loss",
        type=float,
        default=[0.1, 0.2, 0.5, 1],
        nargs="+",
        help="weight of multi-level cl loss",
    )
    parser.add_argument("--lambda_1", type=float, default=1e-4)
    parser.add_argument("--lambda_2", type=float, default=1e-1)

    # optim
    parser.add_argument(
        "--base-lr", type=float, default=0.001, help="initial learning rate"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=[20, 40, 60],
        nargs="+",
        help="the epoch where optimizer reduce the learning rate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        nargs="+",
        help="the indexes of GPUs for training or testing",
    )
    parser.add_argument("--optimizer", default="SGD", help="type of optimizer")
    parser.add_argument(
        "--nesterov", type=str2bool, default=False, help="use nesterov or not"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="start training from which epoch"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=80, help="stop training in which epoch"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--lr-decay-rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--warm_up_epoch", type=int, default=0)
    parser.add_argument("--loss-alpha", type=float, default=0.8)
    parser.add_argument("--te-lr-ratio", type=float, default=1)
    parser.add_argument("--use_prompt", type=float, default=True)
    parser.add_argument("--use_ab", type=float, default=False)
    parser.add_argument("--loss", default="CrossEntropy", help="the loss will be used")
    parser.add_argument("--loss_args", default=dict(), help="the arguments of loss")
    return parser


class Processor:
    """
    Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == "train":
            if not arg.train_feeder_args["debug"]:
                arg.model_saved_name = os.path.join(arg.work_dir, "runs")
                if os.path.isdir(arg.model_saved_name):
                    print("log_dir: ", arg.model_saved_name, "already exist")
                    shutil.rmtree(arg.model_saved_name)
                    print("Dir removed: ", arg.model_saved_name)
                self.train_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "train"), "train"
                )
                self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "val"), "val"
                )
            else:
                self.train_writer = self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "test"), "test"
                )
        self.global_step = 0
        # pdb.set_trace()
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device,
                )

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                for name in self.arg.model_args["head"]:
                    self.model_text_dict[name] = nn.DataParallel(
                        self.model_text_dict[name],
                        device_ids=self.arg.device,
                        output_device=self.output_device,
                    )

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
            )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
        )

    def load_model(self):
        output_device = (
            self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        )
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        if self.arg.wrapper:
            Wrapper = import_class(self.arg.wrapper)
            self.model = Wrapper(Model(**self.arg.model_args), **self.arg.wrapper_args)
        else:
            self.model = Model(**self.arg.model_args)
        self.loss = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)
        self.CEloss4test = get_loss_func("CrossEntropy", None).cuda(output_device)
        self.klloss = KLLoss().cuda(output_device)
        self.model_text_dict = nn.ModuleDict()
        if self.arg.use_prompt:
            for name in self.arg.model_args["head"]:
                model_, preprocess = clip.load(name, device)
                # model_, preprocess = clip.load('ViT-L/14', device)
                del model_.visual
                model_text = TextCLIP(model_)
                model_text = model_text.cuda(self.output_device)
                self.model_text_dict[name] = model_text

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split("-")[-1])
            self.print_log("Load weights from {}.".format(self.arg.weights))
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "r") as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [
                    [k.split("module.")[-1], v.cuda(output_device)]
                    for k, v in weights.items()
                ]
            )

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log(
                                "Sucessfully Remove Weights: {}.".format(key)
                            )
                        else:
                            self.print_log("Can Not Remove Weights: {}.".format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print("  " + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                [
                    {"params": self.model.parameters(), "lr": self.arg.base_lr},
                    {
                        "params": self.model_text_dict.parameters(),
                        "lr": self.arg.base_lr * self.arg.te_lr_ratio,
                    },
                ],
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError()

        self.print_log("using warm up, epoch: {}".format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == "SGD" or self.arg.optimizer == "Adam":
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step))
                )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + " ] " + str
        print(str)
        if self.arg.print_log:
            with open("{}/log.txt".format(self.arg.work_dir), "a") as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log("Training epoch: {}".format(epoch + 1))
        loader = self.data_loader["train"]
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar("epoch", epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=160, desc=f"Epoch {epoch + 1}", leave=True)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
            timer["dataloader"] += self.split_time()
            self.optimizer.zero_grad()

            # forward
            with torch.cuda.amp.autocast():
                label_g = gen_label(label)
                label = label.long().cuda(self.output_device)
                if self.arg.wrapper:
                    output, feature_dict, logit_scale, part_feature_list, z1, z2 = (
                        self.model(data)
                    )
                elif self.arg.use_ab:
                    output = self.model(data)
                else:
                    output, feature_dict, logit_scale, part_feature_list = self.model(
                        data
                    )
                loss_txt = None
                loss_te_list = []
                if self.arg.use_prompt:
                    for ind in range(num_text_aug):
                        if ind > 0:
                            text_id = np.ones(len(label), dtype=np.int8) * ind
                            texts = torch.stack(
                                [text_dict[j][i, :] for i, j in zip(label, text_id)]
                            )
                            texts = texts.cuda(self.output_device)

                        else:
                            texts = list()
                            for i in range(len(label)):
                                text_len = len(text_list[label[i]])
                                text_id = np.random.randint(text_len, size=1)
                                text_item = text_list[label[i]][text_id.item()]
                                texts.append(text_item)

                            texts = torch.cat(texts).cuda(self.output_device)

                        text_embedding = self.model_text_dict[
                            self.arg.model_args["head"][0]
                        ](texts).float()

                        if ind == 0:
                            logits_per_image, logits_per_text = create_logits(
                                feature_dict[self.arg.model_args["head"][0]],
                                text_embedding,
                                logit_scale[:, 0].mean(),
                            )

                            ground_truth = torch.tensor(
                                label_g,
                                dtype=feature_dict[
                                    self.arg.model_args["head"][0]
                                ].dtype,
                                device=device,
                            )
                        else:
                            logits_per_image, logits_per_text = create_logits(
                                part_feature_list[ind - 1],
                                text_embedding,
                                logit_scale[:, ind].mean(),
                            )

                            ground_truth = torch.tensor(
                                label_g,
                                dtype=part_feature_list[ind - 1].dtype,
                                device=device,
                            )
                        loss_texts = self.klloss(logits_per_text, ground_truth)
                        loss_imgs = self.klloss(logits_per_image, ground_truth)
                        loss_te_list.append((loss_imgs + loss_texts) / 2)
                    loss_txt = (
                        self.arg.loss_alpha * sum(loss_te_list) / len(loss_te_list)
                    )
                    if self.arg.wrapper:
                        loss_ce = self.loss((output, z1, z2), label)
                    else:
                        loss_ce = self.loss(output, label)
                    loss = loss_ce + loss_txt
                elif self.arg.use_ab:
                    loss = sum([self.loss(out, label) for out in output[0]])
                    output[0] = sum(output[0])
                else:
                    loss = self.loss(output, label)
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            timer["model"] += self.split_time()

            value, predict_label = torch.max(output[0].data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar("acc", acc, self.global_step)
            self.train_writer.add_scalar("loss", loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]["lr"]
            self.train_writer.add_scalar("lr", self.lr, self.global_step)
            timer["statistics"] += self.split_time()
            if self.arg.use_prompt:
                process.set_postfix(
                    {
                        "loss": float(loss.data.item()),
                        "loss_txt": float(loss_txt.data.item()),
                        "lr": float(self.lr),
                    },
                    refresh=True,
                )
            else:
                process.set_postfix(
                    {"loss": float(loss.data.item()), "lr": float(self.lr)},
                    refresh=True,
                )
            # statistics of time consumption and loss
        proportion = {
            k: "{:02d}%".format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            "\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.".format(
                np.mean(loss_value), np.mean(acc_value) * 100
            )
        )
        self.print_log(
            "\tTime consumption: [Data]{dataloader}, [Network]{model}".format(
                **proportion
            )
        )

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict(
                [[k.split("module.")[-1], v.cpu()] for k, v in state_dict.items()]
            )

            torch.save(
                weights,
                self.arg.model_saved_name
                + "-"
                + str(epoch + 1)
                + "-"
                + str(int(self.global_step))
                + ".pt",
            )

    def eval(
        self,
        epoch,
        save_score=False,
        loader_name=["test"],
        wrong_file=None,
        result_file=None,
    ):
        if wrong_file is not None:
            f_w = open(wrong_file, "w")
        if result_file is not None:
            f_r = open(result_file, "w")
        self.model.eval()
        self.print_log("Eval epoch: {}".format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(
                self.data_loader[ln], ncols=100, desc=f"Epoch {epoch + 1}", leave=True
            )

            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    # print(data.size())
                    b, _, _, _, _ = data.size()
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output, *_ = self.model(data)
                    loss = self.CEloss4test(output[0], label)

                    score_frag.append(output[0].data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output[0].data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + "," + str(true[i]) + "\n")
                        if x != true[i] and wrong_file is not None:
                            f_w.write(
                                str(index[i]) + "," + str(x) + "," + str(true[i]) + "\n"
                            )
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print("Accuracy: ", accuracy, " model: ", self.arg.model_saved_name)
            if self.arg.phase == "train":
                self.val_writer.add_scalar("loss", loss, self.global_step)
                self.val_writer.add_scalar("acc", accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log("\tMean {} loss: {}.".format(ln, np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log(
                    "\tTop{}: {:.2f}%".format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)
                    )
                )

            if save_score:
                with open(
                    "{}/epoch{}_{}_score.pkl".format(self.arg.work_dir, epoch + 1, ln),
                    "wb",
                ) as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open(
                "{}/epoch{}_{}_each_class_acc.csv".format(
                    self.arg.work_dir, epoch + 1, ln
                ),
                "w",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == "train":
            self.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
            self.global_step = (
                self.arg.start_epoch
                * len(self.data_loader["train"])
                / self.arg.batch_size
            )

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f"# Parameters: {count_parameters(self.model)}")
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (
                    ((epoch + 1) % self.arg.save_interval == 0)
                    or (epoch + 1 == self.arg.num_epoch)
                ) and (epoch + 1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=["test"])

            # test the best model
            # weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights_path = glob.glob(
                os.path.join(
                    self.arg.work_dir, "runs-" + str(self.best_acc_epoch) + "*" + ".pt"
                )
            )[0]
            print(
                "---------------------------------------------------------------------load the best model-",
                weights_path,
            )
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict(
                        [
                            ["module." + k, v.cuda(self.output_device)]
                            for k, v in weights.items()
                        ]
                    )
            self.model.load_state_dict(weights)

            wf = weights_path.replace(".pt", "_wrong.txt")
            rf = weights_path.replace(".pt", "_right.txt")
            self.arg.print_log = False
            self.eval(
                epoch=0,
                save_score=True,
                loader_name=["test"],
                wrong_file=wf,
                result_file=rf,
            )
            self.arg.print_log = True

            num_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.print_log(f"Best accuracy: {self.best_acc}")
            self.print_log(f"Epoch number: {self.best_acc_epoch}")
            self.print_log(f"Model name: {self.arg.work_dir}")
            self.print_log(f"Model total number of params: {num_params}")
            self.print_log(f"Weight decay: {self.arg.weight_decay}")
            self.print_log(f"Base LR: {self.arg.base_lr}")
            self.print_log(f"Batch Size: {self.arg.batch_size}")
            self.print_log(f"Test Batch Size: {self.arg.test_batch_size}")
            self.print_log(f"seed: {self.arg.seed}")

        elif self.arg.phase == "test":
            wf = self.arg.weights.replace(".pt", "_wrong.txt")
            rf = self.arg.weights.replace(".pt", "_right.txt")

            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            self.arg.print_log = False
            self.print_log("Model:   {}.".format(self.arg.model))
            self.print_log("Weights: {}.".format(self.arg.weights))
            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=["test"],
                wrong_file=wf,
                result_file=rf,
            )
            self.print_log("Done.\n")


if __name__ == "__main__":
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
