## 目录结构

```
skeleton-acton-det
│  ensemble.py
│  **environment.yml**
│  loss.py
│  LICENSE
│  **main_multipart_uav.py**
│  README.md
│  run.sh
│  Text_Prompt.py # tokeniz the prompts
│  tools.py
│  tree.txt
│  utils.py
│  
├─**clip**
│  └─ ---text encoder files---
│          
├─config
│   └─ ---train/test yaml files---    
├─data
│
├─feeders
│  └─ ---dataloader files---
├─graph
│  └─ ---adjacency matrix files---
│          
├─**model**
│  └─ ---skeleton encoder files---
│          
├─**text**
│      uav_label_map.txt
│      uav_motion_describe.txt
│      uav_motion_details.txt
│      uav_pasta_openai.txt
│      uav_synonym_openai.txt
│      uav_used_parts-openai.txt
│      
├─torchlight
└─**work_dir**
```


## 代码依赖库
The main environment configuration is as follows,
- python=3.8.13
- torch=1.8.1
- torchvision=0.2.1
- PyYAML, tqdm, tensorboardX
- ...



### 数据集目录
```
- data/
  - train_label.npy
  ...
```

### 训练

- 在比赛数据集上训练模型

```
# Example1: 在比赛数据集上训练joint模态
CUDA_VISIBLE_DEVICES=0 python train.py --config config/ctrgcn/k8.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/custom/k8 --device 0 --log-interval 5 --save-interval 1
```


### 组合模型与模态
```
# Example: 在官方数据集上组合模型与模态进行测试
python ensemble.py
```


### 测试

```
#产生置信度文件pred.npy
python infer.py

```

## 训练日志、权重、置信度文件（国赛）
```
https://drive.google.com/file/d/1VpaRxmR4IazGPNWx3Wojfv-XFBW-CEfx/view?usp=sharing
```

## 训练的最好权重以及日志（省赛）
```
- work_dir/ctrgcn
  - k1
    - k1_best.pt
    - log.txt
  ...
```

## 模型的置信度文件（省赛）
```
- results
  - ctrgcn
    - k1
      - A.npy
      - B.npy
    ...
  ...
```
