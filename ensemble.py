import argparse
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

# 从配置文件中加载模型配置和数据集配置
from ensemble_config import model_config, dataset_config


def load_data(label_path, model_paths):
    label = np.load(label_path, mmap_mode="r")
    results = [np.load(path, mmap_mode="r") for path in model_paths]
    return label, results


def weighted_sum(results, weights):
    result_sum = np.zeros_like(results[0])
    for i, result in enumerate(results):
        result_sum += result * weights[i]
    return result_sum


def evaluate_model(label, results):
    right_num = total_num = right_num_5 = 0
    pred_results = []

    for i in tqdm(range(len(label))):
        l = label[i]

        # 已经是融合后的结果，直接进行softmax
        r = softmax(results[i])
        pred_results.append(r)

        # 获取 Top-5 排名
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)

        # 获取 Top-1 预测
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1

    # 计算准确率
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    return acc, acc5, np.array(pred_results)


def save_predictions(predictions, output_path):
    np.save(output_path, predictions)


def generate_model_paths(model_config, dataset):
    """根据传入的模型配置自动生成路径"""
    paths = []
    for model_name, model_info in model_config.items():
        modalities = model_info["modalities"]
        for modality in modalities:
            paths.append(f"results/{model_name}/{modality}/{dataset}.npy")
    return paths


def main():
    parser = argparse.ArgumentParser()

    # 从配置文件中读取数据集类型和路径
    parser.add_argument(
        "--dataset-type",
        choices=["A", "B"],
        default=dataset_config["dataset_type"],
        help="Choose dataset type A or B",
    )
    parser.add_argument(
        "--output-path",
        default=dataset_config["output_path"],
        help="Path to save predictions",
    )

    args = parser.parse_args()

    # 动态生成 label path
    label_path = f"data/test_label_{args.dataset_type}.npy"

    # 自动生成模型路径
    model_paths = generate_model_paths(model_config, args.dataset_type)

    # 加载数据
    label, model_results = load_data(label_path, model_paths)

    # 每个模型的模态加权求和
    final_results_per_model = []
    model_offset = 0
    for model_name, model_info in model_config.items():
        modalities = model_info["modalities"]
        weights = model_info["weights"]

        model_modal_results = model_results[
            model_offset : model_offset + len(modalities)
        ]
        model_offset += len(modalities)

        weighted_modality_sum = [
            weighted_sum([res[j] for res in model_modal_results], weights)
            for j in range(len(label))
        ]
        final_results_per_model.append(weighted_modality_sum)

    # 模型间加权
    model_weights = [
        model_info["weight"] for model_name, model_info in model_config.items()
    ]
    final_results = [
        weighted_sum([res[i] for res in final_results_per_model], model_weights)
        for i in range(len(label))
    ]

    # 评估模型
    acc, acc5, predictions = evaluate_model(label, final_results)

    # 输出结果
    print(f"Top1 Acc: {acc * 100:.4f}%")
    print(f"Top5 Acc: {acc5 * 100:.4f}%")

    # 保存预测结果
    save_predictions(predictions, args.output_path)


if __name__ == "__main__":
    main()
