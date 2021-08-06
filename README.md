# Classification in PyTorch
⌚️: 2021年5月31日

📚参考

---
<!-- TOC -->
- [Classification in PyTorch](#Classification in PyTorch)
  - [Requirements](#requirements)
  - [Main Features](#main-features)
    - [Models](#models)
    - [Datasets](#datasets)
    - [Losses](#losses)
    - [Learning rate schedulers](#learning-rate-schedulers)
    - [Data augmentation](#data-augmentation)
  - [Training](#training)
  - [Inference](#inference)
  - [Code structure](#code-structure)
  - [Config file format](#config-file-format)
  - [Acknowledgement](#acknowledgement)

<!-- /TOC -->

This repo contains a PyTorch an implementation of different classification models for different datasets.

## 1、环境/Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` and `opencv` for data-preprocessing and `tqdm` for showing the training progress. PyTorch v1.1 is supported (using the new supported tensoboard); can work with ealier versions, but instead of using tensoboard, use tensoboardX.
具体参考[environment_install.md](environment_install.md)

## 2、特性/Main Features

- A clear and easy to navigate structure,
- A `json` config file with a lot of possibilities for parameter tuning,
- Supports various models, losses, Lr schedulers, data augmentations and datasets,

**So, what's available ?**

### 2.1 模型/Models 
- Resnet系列
- Alexnet
- Densenet系列
- efficientnet系列

### 2.2 数据集/Datasets

- 中间区域数据
- 边缘区域数据
- LCD液晶区数据

### 2.3 损失函数/Losses
In addition to the Cross-Entorpy loss, there is also
- Cross-Entorpy

### 2.4 学习率调整策略/Learning rate schedulers
- **StepLR**

### 2.5 数据增强Data augmentation
All of the data augmentations are implemented using OpenCV in `\base\base_dataset.py`, which are: rotation (between -10 and 10 degrees), random croping between 0.5 and 2 of the selected `crop_size`, random h-flip and blurring

## 3. 训练/Training
To train a model, first download the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (the config file is detailed below), then simply run:

```bash
python train.py --configs configs.json
```

The training will automatically be run on the GPUs (if more that one is detected and  multipple GPUs were selected in the config file, `torch.nn.DataParalled` is used for multi-gpu training), if not the CPU is used. The log files will be saved in `saved\runs` and the `.pth` chekpoints in `saved\`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir saved
```


## 4. 推理/Inference

For inference, we need a PyTorch trained model, the images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference_edge/middle/oled.py --configs configs.json --model best_model.pth --images images_folder
```