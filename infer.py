import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import OrderedDict
import yaml
from tqdm import tqdm
import random
import sys
import traceback
from torchlight import DictAction
import argparse
import os
import torch.nn.functional as F

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        default="./work_dir/temp",
        help="the work folder for storing results",
    )
    parser.add_argument(
        '--result_dir',
        default='./results',
        help='the folder for storing results'
    )
    parser.add_argument("-model_saved_name", default="")
    parser.add_argument(
        "--config",
        default="./config/uav-cross-subject/lst_joint_v2.yaml",
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
        "--num-worker", type=int, default=1, help="the number of worker for data loader"
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
        "--weights",
        default="./work_dir/uav/ctrgcn/joint_vel/best.pt",
        help="the weights for network initialization",
    )
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )

    # optim
    parser.add_argument(
        "--base-lr", type=float, default=0.01, help="initial learning rate"
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
        "--num-epoch", type=int, default=80, help="stop training in which epoch"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0005, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--lr-decay-rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--warm_up_epoch", type=int, default=0)

    return parser


def inference(arg):
    # Initialize seed
    init_seed(arg.seed)

    # Load model
    Model = import_class(arg.model)
    model = Model(**arg.model_args)
    output_device = arg.device[0] if type(arg.device) is list else arg.device
    model = model.cuda(output_device)

    # Load model weights
    if arg.weights:
        weights = torch.load(arg.weights)
        if type(arg.device) is list and len(arg.device) > 1:
            weights = OrderedDict(
                [["module." + k, v.cuda(output_device)] for k, v in weights.items()]
            )
        model.load_state_dict(weights)
    else:
        raise ValueError("Please appoint --weights.")

    model.eval()

    # Load data
    Feeder = import_class(arg.feeder)
    data_loader = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.test_feeder_args),
        batch_size=arg.test_batch_size,
        shuffle=False,
        num_workers=arg.num_worker,
        drop_last=False,
        worker_init_fn=init_seed,
    )

    # Inference
    label_list = []
    pred_list = []
    confidence_list = []

    for batch_idx, (data, label, index) in enumerate(tqdm(data_loader, ncols=40)):
        with torch.no_grad():
            data = data.float().cuda(output_device)
            label = label.long().cuda(output_device)
            output, _, _, _ = model(data)
            _, predict_label = torch.max(output.data, 1)
            pred_list.append(predict_label.data.cpu().numpy())
            confidence_scores = F.softmax(output, dim=1)
            # confidence_list.append(confidence_scores.data.cpu().numpy())
            confidence_list.append(output.data.cpu().numpy())
            label_list.append(label.data.cpu().numpy())

    # Concatenate all results
    label_list = np.concatenate(label_list)
    pred_list = np.concatenate(pred_list)
    confidence_list = np.concatenate(confidence_list)

    # Compute Top-1 accuracy
    top1_accuracy = accuracy_score(label_list, pred_list) * 100
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")

    # Save labels and confidences to .npy files
    np.save(os.path.join(arg.result_dir, 'label.npy'), label_list)
    np.save(os.path.join(arg.result_dir, "joint_vel_nsm_B.npy"), confidence_list)

    return top1_accuracy


if __name__ == "__main__":
    # Load arguments from the configuration file
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.safe_load(f.read())
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    top1_accuracy = inference(arg)
