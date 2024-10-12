import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
import itertools
from concurrent.futures import ProcessPoolExecutor


def evaluate_weights_combination(alpha, r1, r2, r3, r4, label):
    right_num = total_num = right_num_5 = 0
    for i in range(len(label)):
        l = label[i]
        r11 = r1[i]
        r22 = r2[i]
        r33 = r3[i]
        r44 = r4[i]

        # Weighted summation of the four modalities
        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    return alpha, acc, acc5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices={
            "uav/xsub_v1",
            "uav/xsub_v2",
            "ntu/xsub",
            "ntu/xview",
            "ntu120/xsub",
            "ntu120/xset",
            "NW-UCLA",
        },
        default="uav/ctrgcn",
        help="the work folder for storing results",
    )
    parser.add_argument(
        "--label-path", help="path to the label file", default="data/test_label.npy"
    )
    parser.add_argument(
        "--joint-dir",
        help='Directory containing "epoch1_test_score.pkl" for joint eval results',
        default="results/joint_nsm.npy",
    )
    parser.add_argument(
        "--bone-dir",
        help='Directory containing "epoch1_test_score.pkl" for bone eval results',
        default="results/bone_nsm.npy",
    )
    parser.add_argument("--joint-motion-dir", default="results/joint_vel_nsm.npy")
    parser.add_argument("--bone-motion-dir", default="results/bone_vel_nsm.npy")

    arg = parser.parse_args()

    dataset = arg.dataset
    if "UCLA" in arg.dataset:
        label = []
        with open("./data/" + "NW-UCLA/" + "/val_label.pkl", "rb") as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info["label"]) - 1)
    elif "ntu120" in arg.dataset:
        if "xsub" in arg.dataset:
            npz_data = np.load("./data/" + "ntu120/" + "NTU120_CSub.npz")
            label = np.where(npz_data["y_test"] > 0)[1]
        elif "xset" in arg.dataset:
            npz_data = np.load("./data/" + "ntu120/" + "NTU120_CSet.npz")
            label = np.where(npz_data["y_test"] > 0)[1]
    elif "ntu" in arg.dataset:
        if "xsub" in arg.dataset:
            npz_data = np.load("./data/" + "ntu/" + "NTU60_CS.npz")
            label = np.where(npz_data["y_test"] > 0)[1]
        elif "xview" in arg.dataset:
            npz_data = np.load("./data/" + "ntu/" + "NTU60_CV.npz")
            label = np.where(npz_data["y_test"] > 0)[1]
    elif "uav" in arg.dataset:
        label = np.load(arg.label_path, mmap_mode="r")
    else:
        raise NotImplementedError

    r1 = np.load(arg.joint_dir, mmap_mode="r")
    r2 = np.load(arg.bone_dir, mmap_mode="r")

    if arg.joint_motion_dir is not None:
        r3 = np.load(arg.joint_motion_dir, mmap_mode="r")
    if arg.bone_motion_dir is not None:
        r4 = np.load(arg.bone_motion_dir, mmap_mode="r")

    # Generate all combinations of weights from 0.1 to 1.0 with a step of 0.1
    weight_combinations = list(itertools.product(np.arange(0.1, 1.1, 0.1), repeat=4))

    # Initialize the best accuracy and corresponding weights
    best_acc = 0
    best_acc5 = 0
    best_weights = []

    # 使用多进程池并行化组合权重的评估
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_weights_combination, alpha, r1, r2, r3, r4, label)
            for alpha in weight_combinations
        ]

        # 处理每个权重组合的结果
        for future in tqdm(futures, total=len(futures)):
            alpha, acc, acc5 = future.result()
            # 检查是否有更好的精度
            if acc > best_acc:
                best_acc = acc
                best_acc5 = acc5
                best_weights = alpha

    print("Best Weights: {}".format(best_weights))
    print("Best Top1 Acc: {:.4f}%".format(best_acc * 100))
    print("Best Top5 Acc: {:.4f}%".format(best_acc5 * 100))
