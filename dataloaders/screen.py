# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import numpy as np
import os
import cv2

from base import BaseDataSet, BaseDataLoader


class ScreenDataset(BaseDataSet):
    def __init__(self, num_class, **kwargs):
        self.num_classes = num_class
        super(ScreenDataset, self).__init__(**kwargs)

    def _set_files(self):
        """
        功能：获取所有文件的文件名和标签
        """
        if self.val:
            list_path = os.path.join(self.root, "vallist.txt")
        else:
            list_path = os.path.join(self.root, "trainlist.txt")

        images, labels = [], []
        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for image_label in images_labels:
                images.append(os.path.join(self.root, image_label.strip().split("____")[0], image_label.strip().split("____")[1]))
                labels.append(image_label.strip().split("____")[2])

        self.files = list(zip(images, labels))

    def _load_data(self, index):
        """
        功能：通过文件名获得，图片和类别
        :param index:
        :return:
        """
        image_path, label = self.files[index]
        if self.in_channels == 1:
            # 修改支持中文路径
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        elif self.in_channels == 3:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img, label, image_path


class ScreenDataLoader(BaseDataLoader):
    def __init__(self, data_dir=None, num_class=33,
                 normalize=False, mean=None, std=None,
                 augment=False, base_size=None, crop_size=None, scale=True,
                 flip=False, rotate=False, blur=False, histogram=False,
                 batch_size=1, num_workers=1, shuffle=True,
                 in_channels=3, val=False
                 ):
        if in_channels == 3:
            self.MEAN = mean
            self.STD = std
        else:
            self.MEAN = mean
            self.STD = std
        kwargs = {
            'root': data_dir,

            'mean': self.MEAN, 'std': self.STD, 'normalize': normalize,     # 是否进行normalize

            'augment': augment,      # 是否进行数据增强
            'base_size': base_size,  # 将图片扩充到一个更大的图片
            'scale': scale,          # 配合base_size使用，扩充时，等比例
            'rotate': rotate,        # 随机旋转图片（-10‘，10‘）
            'crop_size': crop_size,  # 裁剪图片到一定尺寸，resize操作
            'flip': flip,            # 一定概率翻转图片
            'blur': blur,            # 一定概率高斯模糊图片
            'histogram': histogram,  # 是否进行直方图拉伸

            'in_channels': in_channels,     # 图片的通道数

            'val': val      # 是否是验证集
        }

        self.dataset = ScreenDataset(num_class, **kwargs)
        super(ScreenDataLoader, self).__init__(self.dataset, batch_size, shuffle, num_workers)


