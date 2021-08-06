import os
import random
import cv2
import numpy as np


def split_dataset(path="", train_ratio=0.9, shuffle=True):
    all_folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    train_labels = os.path.join(path, "trainlist.txt")
    val_labels = os.path.join(path, "vallist.txt")
    test_labels = os.path.join(path, "testlist.txt")
    labels = os.path.join(path, "labels.txt")

    with open(train_labels, "a+") as train_file, open(val_labels, "a+") as val_file, open(labels, "a+") as labels_file:
        for folder in all_folders:  # 处理每个文件夹
            label = folder.split("_")   # 获取标签
            # 写入labels
            labels_file.write("{}:{}\n".format(int(label[0]), label[1]))

            all_images = os.listdir(os.path.join(path, folder))
            n_total = len(all_images)
            offset = int(n_total * train_ratio)
            if shuffle:
                random.shuffle(all_images)
            train_list = all_images[:offset]    # 训练集图片路径
            val_list = all_images[offset:]      # 验证集图片路径
            # 写入训练和测试图片路径到train/vallist.txt文件中
            for train_img in train_list:
                train_file.write("{}____{}____{}\n".format(folder, train_img, int(label[0])))
            for val_img in val_list:
                val_file.write("{}____{}____{}\n".format(folder, val_img, int(label[0])))


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
    dirname = r"F:\Data\lcd\midle\midle"
    split_dataset(path=dirname, train_ratio=0.9)
    gen_mean_std(root_path=dirname)

