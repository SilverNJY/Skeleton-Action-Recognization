import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
import itertools
from concurrent.futures import ProcessPoolExecutor


def evaluate_weights_combination(alpha, r1, r2, r3, r4, r55, r66, label):
    right_num = total_num = right_num_5 = 0
    for i in range(len(label)):
        l = label[i]
        r11 = r1[i]
        r22 = r2[i]
        r33 = r3[i]
        r44 = r4[i]
        r55 = r55[i]
        r66 = r66[i]

        # Weighted summation of the four modalities
        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3] * r55 * alpha[4] + r66 * alpha[5]
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
        "--label-path", help="path to the label file", default="data/test_label_A.npy"
    )
    parser.add_argument("--joint-dir", default="results/ctrgcn/k8/A.npy")
    parser.add_argument("--bone-dir", default="results/ctrgcn/k1/A.npy")
    parser.add_argument("--joint-motion-dir", default="results/ctrgcn/k8_vel/A.npy")
    parser.add_argument("--bone-motion-dir", default="results/ctrgcn/k1_vel/A.npy")
    parser.add_argument("--frhead-dir", default="results/ctrgcn/frhead_k8/A.npy")
    parser.add_argument("--infogcn-dir", default="results/infogcn/k8/A.npy")

    arg = parser.parse_args()

    label = np.load(arg.label_path, mmap_mode="r")

    r1 = np.load(arg.joint_dir, mmap_mode="r")
    r2 = np.load(arg.bone_dir, mmap_mode="r")
    r3 = np.load(arg.joint_motion_dir, mmap_mode="r")
    r4 = np.load(arg.bone_motion_dir, mmap_mode="r")
    r5 = np.load(arg.frhead_dir, mmap_mode="r")
    r6 = np.load(arg.infogcn_dir, mmap_mode="r")

    # Generate all combinations of weights from 0.1 to 1.0 with a step of 0.1
    weight_combinations = list(itertools.product(np.arange(0.1, 1.1, 0.1), repeat=6))

    # Initialize the best accuracy and corresponding weights
    best_acc = 0
    best_acc5 = 0
    best_weights = []

    # 使用多进程池并行化组合权重的评估
    max_workers = 1  # 限制最多的线程数

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_weights_combination, alpha, r1, r2, r3, r4, r5, r6, label
            )
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
