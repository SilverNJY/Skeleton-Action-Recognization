import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm
from scipy.special import softmax

# sm_best 0.5 0.4 0.1 0.3 77.1 nsm_best 0.7 0.6 0.1 0.3 76.9
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
    parser.add_argument("--alpha", default=1, help="weighted summation", type=float)
    parser.add_argument(
        "--label-path", help="path to the label file", default="data/test_label.npy"
    )
    parser.add_argument("--joint-dir", default="results/joint_nsm_B.npy")
    parser.add_argument("--bone-dir", default="results/bone_nsm_B.npy")
    parser.add_argument("--joint-motion-dir", default="results/joint_vel_nsm_B.npy")
    parser.add_argument("--bone-motion-dir", default="results/bone_vel_nsm_B.npy")

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

    right_num = total_num = right_num_5 = 0
    pred_results = []  # 用于保存 softmax 后的结果

    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        arg.alpha = [0.7, 0.6, 0.1, 0.3]

        for i in tqdm(range(len(label))):
            l = label[i]
            r11 = r1[i]
            r22 = r2[i]
            r33 = r3[i]
            r44 = r4[i]

            r = (
                r11 * arg.alpha[0]
                + r22 * arg.alpha[1]
                + r33 * arg.alpha[2]
                + r44 * arg.alpha[3]
            )
            r = softmax(r)  # 对组合后的结果进行softmax处理
            pred_results.append(r)  # 将softmax后的结果保存到列表中

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[:, i]
            r11 = r1[i]
            r22 = r2[i]
            r33 = r3[i]

            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            r = softmax(r)  # 对组合后的结果进行softmax处理
            pred_results.append(r)  # 将softmax后的结果保存到列表中

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            r11 = r1[i]
            r22 = r2[i]

            r = r11 + r22 * arg.alpha
            r = softmax(r)  # 对组合后的结果进行softmax处理
            pred_results.append(r)  # 将softmax后的结果保存到列表中

            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print("Top1 Acc: {:.4f}%".format(acc * 100))
    print("Top5 Acc: {:.4f}%".format(acc5 * 100))

    # 将softmax后的预测结果保存到npy文件
    pred_results = np.array(pred_results)
    np.save("results/pred.npy", pred_results)
