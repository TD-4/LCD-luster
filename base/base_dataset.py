# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataSet(Dataset):
    def __init__(self, root=None, mean=None, std=None, normalize=False,
                 augment=False, base_size=None, crop_size=224, scale=False,
                 flip=False, rotate=False, blur=False, histogram=False,
                 val=False, in_channels=1):
        # 图像路径
        self.root = root
        self.split = "val" if val else "train"

        # Normalization
        self.mean = mean
        self.std = std
        self.normalize_flag = normalize

        # 数据增强
        self.augment = augment
        self.crop_size = crop_size
        self.histogram = histogram
        if self.augment:
            self.histogram = histogram
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur

        # 是否是验证集
        self.val = val

        # 所有文件的路径和标签
        self.files = []
        self._set_files()   # 获取所有文件的路径和标签，存放到files中

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        # 输入图像的通道数，是否是灰度图像
        self.in_channels = in_channels

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        if self.crop_size:
            # Scale the smaller side to crop size
            if self.in_channels == 1:
                h, w = image.shape
            else:
                h, w, _ = image.shape
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

            # Center Crop
            h, w = image.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]

        # Histogram
        if self.histogram and self.in_channels == 1:
            # print("val histogram!!")
            rows, cols = image.shape
            flat_gray = image.reshape((cols * rows,)).tolist()
            A = min(flat_gray)
            B = max(flat_gray)
            image = np.uint8(255 / (B - A + 0.1) * (image - A) + 0.5)

        return image, label

    def _augmentation(self, image, label):
        # Scaling, we set the bigger to base size, and the smaller one is rescaled to maintain the same ratio,
        # if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.in_channels == 1:
                h, w = image.shape
            else:
                h, w, _ = image.shape
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            if self.in_channels == 1:
                h, w = image.shape
            else:
                h, w, _ = image.shape
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)  #, borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            if self.in_channels == 1:
                h, w = image.shape
            else:
                h, w, _ = image.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

            # Center Crop
            if self.in_channels == 1:
                h, w = image.shape
            else:
                h, w, _ = image.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)

        # Histogram
        if self.histogram and self.in_channels == 1:
            # print("train histogram!!")
            rows, cols = image.shape
            flat_gray = image.reshape((cols * rows,)).tolist()
            A = min(flat_gray)
            B = max(flat_gray)
            image = np.uint8(255 / (B-A+0.1) * (image - A) + 0.5)

        return image, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_path = self._load_data(index)  # image:PIL, label:int
        if self.val:    # 验证集
            image, label = self._val_augmentation(image, label)
        elif self.augment:  # 训练集
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.normalize_flag:
            return self.normalize(self.to_tensor(image)), label, image_path
        return self.to_tensor(image), label, image_path

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

