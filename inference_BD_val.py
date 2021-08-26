import argparse
import os
import numpy as np
import json
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import models
from torchvision import transforms
import shutil
import matplotlib.pyplot as plt
from torchcam.cams import SmoothGradCAMpp, CAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import transforms
from utils import transforms as local_transforms
import time


def get_images(path=""):
    imgs_path = []
    with open(os.path.join(path, "vallist.txt"), 'r') as file:
        for line in file:
            # 0_BD____20210713_BlackDot_BDEx_228.bmp____0
            img_p = os.path.join(path, line.strip().split("____")[0], line.strip().split("____")[1])
            label = line.strip().split("____")[2]
            imgs_path.append((img_p, label))
    return imgs_path


def img_p2img(img_p=None, config=None, args=None, device=None):
    # 读取一张图片
    # 修改支持中文路径
    img = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # resize
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    # totensor
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(config['train_loader']['args']['mean'], config['train_loader']['args']['std'])(img)
    img = img.to(device=device)
    return img



def main():
    args = parse_arguments()
    config = json.load(open(args.configs))
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

    # 1、-------------------获取数据-----------------
    all_images = get_images(args.images)

    # 2、-------------------获得模型-----------------
    num_classes = 2
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    # Load checkpoint
    checkpoint = torch.load(args.model)
    # load
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    if not os.path.exists(args.outputs):
        os.makedirs(args.outputs)
    if not os.path.exists(args.outputs_ok):
        os.makedirs(args.outputs_ok)

    # 3、--------------------------------预测-----------------------------
    # 获得labels
    labels_ = {}
    with open(args.labels, 'r') as f:
        for line in f:
            labels_[line.strip().split(":")[0]] = line.strip().split(":")[1]

    # 定义混淆矩阵
    confusion_matrix = [[0 for j in range(num_classes)] for i in range(num_classes)]
    pred_target_list = []   # 存放混淆矩阵中所用到的数据
    with torch.no_grad():
        cam_extractor = CAM(model, 'model.layer4.1.conv2')

        mean = config['train_loader']['args']['mean']
        std = config['train_loader']['args']['std']
        restore_transform = transforms.Compose([local_transforms.DeNormalize(mean, std), transforms.ToPILImage()])

        tbar = tqdm(all_images, ncols=100)
        for img_label in tbar:
            img_p = img_label[0]
            img = img_p2img(img_p=img_p, config=config, args=args, device=device)
            input_tensor = img.unsqueeze(0)
            output_tensor = model(input_tensor)
            prediction = output_tensor.squeeze(0).cpu().detach().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

            pred = prediction.item()
            label = img_label[1]

            pred_target_list.append((pred, label))  # 记录预测值和标签值，供计算混淆矩阵使用 (int,str)

            if pred != int(label):  # 预测错误， 输出图片、CAM
                # 1. 拷贝图片
                output_name = "pred-{}__target-{}__{}.bmp".format(labels_[str(pred)], labels_[label], str(time.time()).split('.')[0]+str(time.time()).split('.')[1])
                shutil.copy(os.path.join(img_p), os.path.join(args.outputs, output_name))

                # 2. 拷贝直方图
                output_name = output_name[:-4] + ".jpg"
                img_gray = np.array(restore_transform(input_tensor.squeeze(0)))
                rows, cols = img_gray.shape
                flat_gray = img_gray.reshape((cols * rows,)).tolist()
                A = min(flat_gray)
                B = max(flat_gray)
                img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
                cv2.imencode('.jpg', img_gray)[1].tofile(os.path.join(args.outputs, output_name))

                # CAM图
                output_name = output_name[:-4] + ".png"
                img_cam = restore_transform(input_tensor.squeeze(0))

                activation_map = cam_extractor(output_tensor.cpu().squeeze(0).argmax().item(), output_tensor.cpu())
                result = overlay_mask(img_cam.convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)

                cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
                    os.path.join(args.outputs, output_name))
            else:
                # 1. 拷贝图片
                output_name = "pred-{}__target-{}__{}.bmp".format(labels_[str(pred)], labels_[label],
                                                                  str(time.time()).split('.')[0] +
                                                                  str(time.time()).split('.')[1])
                shutil.copy(os.path.join(img_p), os.path.join(args.outputs_ok, output_name))

                # 2. 拷贝直方图
                output_name = output_name[:-4] + ".jpg"
                img_gray = np.array(restore_transform(input_tensor.squeeze(0)))
                rows, cols = img_gray.shape
                flat_gray = img_gray.reshape((cols * rows,)).tolist()
                A = min(flat_gray)
                B = max(flat_gray)
                img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
                cv2.imencode('.jpg', img_gray)[1].tofile(os.path.join(args.outputs_ok, output_name))

                # CAM图
                output_name = output_name[:-4] + ".png"
                img_cam = restore_transform(input_tensor.squeeze(0))

                activation_map = cam_extractor(output_tensor.cpu().squeeze(0).argmax().item(), output_tensor.cpu())
                result = overlay_mask(img_cam.convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)

                cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
                    os.path.join(args.outputs_ok, output_name))

    # 输出混淆矩阵
    for p, t in pred_target_list:   # (int,str)
        confusion_matrix[int(t)][p] +=1

    # print confusion matrix
    confusion_file = open(os.path.join(args.outputs, "confusion.txt"), 'a+')
    print("{0:10}".format(""), end="")  # 第一行第一个空格
    confusion_file.write("{0:8}".format(""))
    for name in range(num_classes):  # 第一行 所有的类别名称
        print("{0:10}".format(labels_[str(name)]), end="")
        confusion_file.write("{0:8}".format(labels_[str(name)]))
    print("{0:10}".format("Precision"))  # 第一行 每一类的准确率
    confusion_file.write("{0:8}\n".format("Precision"))

    # 第二行以后
    for i in range(num_classes):  # 第N行
        print("{0:10}".format(labels_[str(i)]), end="")   # 第N行，第一列，名称
        confusion_file.write("{0:8}".format(labels_[str(i)]))
        for j in range(num_classes):
            if i == j:
                print("{0:10}".format(str("-" + str(confusion_matrix[i][j])) + "-"), end="")
                confusion_file.write("{0:8}".format(str("-" + str(confusion_matrix[i][j])) + "-"))
            else:
                print("{0:10}".format(str(confusion_matrix[i][j])), end="")
                confusion_file.write("{0:8}".format(str(confusion_matrix[i][j])))
        precision = 0.0 + confusion_matrix[i][i] / sum(confusion_matrix[i])
        print("{0:.4f}".format(precision))
        confusion_file.write("{0:8}\n".format(precision))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--configs', default='configs/BD-Binary_Resnet18_CEL_SGD.json', type=str, help='The configs used to train the model')
    parser.add_argument('-m', '--model', default='pretrained/bd-resnet18.pth', type=str, help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-l', '--labels', default=r'F:\Data\BD_binary\labels.txt', type=str, help='label Path')

    parser.add_argument('-i', '--images', default=r'F:\Data\BD_binary', type=str, help='Path to the vallist.txt')
    parser.add_argument('-o', '--outputs', default=r'F:\BD_error', type=str, help='Output Path')
    parser.add_argument('-oc', '--outputs_ok', default=r'F:\BD_ok', type=str, help='Output Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
