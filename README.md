# Classification in PyTorch
⌚️: 2021年5月31日

📚参考

---

```
项目结构介绍

-base   基础库包
--__init__.py
--base_dataloader.py
--base_dataset.py
--base_trainer.py

-configs    配置文件夹
--Middle_EfficientNetb0_CEL_SGD.json
--...

-data   数据集制作脚本
--data-copy.py
--data-split.py

-dataloader     数据读取库包
--__init__.py
--edge.py
--middle.py
--oled.py

-models     模型库包
--__init__.py
--alexnet.py
--densenet.py
--EfficientNet.py
--inceptions.py
--resnet.py

-pretrained     预训练模型，需下载创建
--...

-utils
--sync_batchnorm
----...
--__init__.py
--del_onefile.py
--helpers.py
--logger.py
--losses.py
--lr_scheduler.py
--metrics.py
--transforms.py


-.gitignore
-environment_install.md
-inference_middle.py    中间区域推理脚本（把所有文件放到一个list中）
-inference_middle_.py   中间区域推理脚本
-inference_middle_val.py    中间区域推理脚本（输出best_model.pth的混淆矩阵、CAM、预测类别和实际类别）

-train.py   训练代码
-trainer.py 训练过程具体实现
```
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
#### 2.2.1 点灯小图数据
data目录下存放了数据处理的代码。

- data-copy.py 是从最原始模型预测结果中筛选出Edge和Middle中点灯小图，无Ori
  
- data-split.py 是将最终所有图片划分为train和val，生成trainlist.txt、vallist.txt、labels.txt、mean_std.txt。
  
- del-jpg.py 是删除png（热力图）和jpg（增强图），并重新生成
  
- image-check.py 是检查对比 原图、Normalization后的图，Histogram拉伸后的图，先Histogram拉伸然后在Normalization的图，对比它们之间的差异，
  输入到Model是否有影响
  
#### 2.2.2 LCD液晶区数据

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