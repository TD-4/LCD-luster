# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import shutil
import cv2
import numpy as np
import time
from torchvision import transforms
from utils import transforms as local_transforms


def save_ori(path="", path2="", name =""):  # 原始图
    shutil.copy(path, os.path.join(path2,  name+ "_ori.bmp"))


def save_his(path, path2, name):    # 直方图拉伸
    img_o = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img_gray = np.array(img_o)
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
    cv2.imencode('.png', np.array(img_gray))[1].tofile(os.path.join(path2,name + "_his.png"))


def save_normal(path, path2, name):     # Normalize 后
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img_gray = np.array(img)
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.

    # totensor
    img = transforms.ToTensor()(img_gray)
    img = transforms.Normalize([0.39703156192416594],[0.16976514942964588])(img)

    # restore_transform = transforms.Compose([local_transforms.DeNormalize([0.39703156192416594],[0.16976514942964588]),transforms.ToPILImage()])
    restore_transform = transforms.Compose([transforms.ToPILImage()])
    img_o = restore_transform(img)
    cv2.imencode('.jpg', np.array(img_o))[1].tofile(os.path.join(path2,name + "_his_normal.jpg"))


def save_his_normal(path, path2, name):     # 先his，再Normalize 后
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # totensor
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.39703156192416594],[0.16976514942964588])(img)

    # restore_transform = transforms.Compose([local_transforms.DeNormalize([0.39703156192416594],[0.16976514942964588]),transforms.ToPILImage()])
    restore_transform = transforms.Compose([transforms.ToPILImage()])
    img_o = restore_transform(img)
    cv2.imencode('.jpg', np.array(img_o))[1].tofile(os.path.join(path2,name + "_normal.jpg"))


if __name__ == "__main__":
    path=r"F:\todo2\15_aOK"
    path2 = r"F:\test"
    all_images = [img_p for img_p in os.listdir(os.path.join(path)) if img_p[-4:] == ".bmp"]
    for img in all_images:
        name = time.time().__str__()
        save_ori(os.path.join(path, img), path2, name)
        save_his(os.path.join(path, img), path2, name)
        save_normal(os.path.join(path, img), path2, name)
        save_his_normal(os.path.join(path, img), path2, name)
