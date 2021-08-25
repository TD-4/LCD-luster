# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import random
import cv2
import numpy as np
import functools
import time


def rename_all_images(path=""):
    all_folder = os.listdir(path)
    for folder in all_folder:
        all_images = os.listdir(os.path.join(path, folder))
        for i, img_p in enumerate(all_images):
            os.rename(os.path.join(path, folder, img_p),
                      os.path.join(path, folder, folder+"_"+str(time.time()).split(".")[0]+str(time.time()).split(".")[1]+"_"+str(i)+".bmp"))


def cmp(x,y):
    x,y = x.split("_")[0], y.split("_")[0]
    return int(y) - int(x)


def split_dataset(path="", train_ratio=0.9, shuffle=True, range_=(2000, 2400)):
    all_folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    all_folders.sort(key=functools.cmp_to_key(cmp), reverse=True)
    train_labels = os.path.join(path, "trainlist.txt")
    val_labels = os.path.join(path, "vallist.txt")
    labels = os.path.join(path, "labels.txt")
    data_count = [0,0,0]
    with open(train_labels, "a+") as train_file, open(val_labels, "a+") as val_file, open(labels, "a+") as labels_file:
        for folder in all_folders:  # 处理每个文件夹
            label = folder.split("_")   # 获取标签
            # 写入labels
            labels_file.write("{}:{}\n".format(int(label[0]), label[1]))

            all_images = [img_p for img_p in os.listdir(os.path.join(path, folder)) if img_p[-4:]==".bmp"]
            n_total = min(random.randint(range_[0], range_[1]), len(all_images))  # 控制每类样本数量
            offset = int(n_total * train_ratio)

            train_list = all_images[:offset]  # 训练集图片路径
            val_list = all_images[offset:n_total]  # 验证集图片路径
            print("label {} 类中，训练集--{}, 验证集--{}".format(label[1], len(train_list), len(val_list)))
            tmp = [n_total, len(train_list), len(val_list)]
            data_count = [d[0]+d[1] for d in zip(tmp, data_count)]
            if shuffle:
                random.shuffle(all_images)
            # 写入训练和测试图片路径到train/vallist.txt文件中
            for train_img in train_list:
                train_file.write("{}____{}____{}\n".format(folder, train_img, int(label[0])))
            for val_img in val_list:
                val_file.write("{}____{}____{}\n".format(folder, val_img, int(label[0])))

    print("train {},   val {},  total {}".format(data_count[1], data_count[2], data_count[0]))


def gen_mean_std(root_path=""):
    """
    获得mean & std
    """
    # 获取所有图片
    all_images = open(os.path.join(root_path, "trainlist.txt"), 'rU').readlines()

    # get mean
    gray_channel = 0
    count = 0
    for line in all_images:
        line = os.path.join(root_path, line.strip().split("____")[0], line.strip().split("____")[1])
        img = cv2.imdecode(np.fromfile(line, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        gray_channel += np.sum(img)
        count += 1
        # print("mean count:", str(count))
    gray_channel_mean = gray_channel / (count * 150 * 150)

    # get std
    gray_channel = 0
    count = 0
    for line in all_images:
        line = os.path.join(root_path, line.strip().split("____")[0], line.strip().split("____")[1])
        img = cv2.imdecode(np.fromfile(line, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        gray_channel = gray_channel + np.sum((img - gray_channel_mean)**2)
        count += 1
        # print("std count:", str(count))
    gray_channel_std = np.sqrt(gray_channel / (count * 150 * 150))

    with open(os.path.join(root_path, "mean_std.txt"), "a+") as mean_file:
        mean_file.write("mean:{}\nstd:{}\n".format(gray_channel_mean, gray_channel_std))


if __name__ == "__main__":
    # dirname, filename = os.path.split(os.path.abspath(__file__))
    dirname = r"F:\Data\Screen\trainval_notime_20210820"
    # 1、重命名所有图片文件，将中文改成英文
    rename_all_images(dirname)
    # 2、将数据划分为train和val
    split_dataset(path=dirname, train_ratio=0.9)
    # 3、获得所有train图片的均值与方差
    gen_mean_std(root_path=dirname)

