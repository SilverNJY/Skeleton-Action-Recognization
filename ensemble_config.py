# 模型配置，包括每个模型的模态、模态权重和模型间权重
model_config = {
    "ctrgcn": {
        "modalities": ["k8", "k1f", "k2f", "k8_velf", "k2_vel", "k1_vel"],
        "weights": [0.6, 0.8, 0.6, 0.6, 0.2, 0.],
        "weight": 0.8,
    },
    "infogcn": {
        "modalities": ["k8f", "k1f", "k2", "k8_vel", "k1_vel"],
        "weights": [0.4, 0.5, 0., 0.2, 0],
        "weight": 0.5,
    },
    "mixformer": {
        "modalities": ["k8", "k1f", "k21", "k8_vel", "k2_velf"],
        "weights": [0.4, 0.5, 0.3, 0.3, 0.5],
        "weight": 0.5,
    },
    "jbf": {
        "modalities": ["ctrgc", "ctrgc-vel"],
        "weights": [0.5, 0.5],
        "weight": 0.5,
    },
}

# 数据集配置
dataset_config = {
    "dataset_type": "A",  # 选择数据集类型 A 或 B
    "output_path": "results/pred.npy",  # 保存预测结果的路径
}
