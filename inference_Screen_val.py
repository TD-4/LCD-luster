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


tolerate_class = {
    "00BD": ("02BM", "01WL", "02ZW", "aBHLM", "02BM2"),   # pred:(target) is ok
    "02BM": ("00BD", "01WL", "02ZW", "02BLM", "02BM2", "aBHLM"),
    "01WL": ("00BD", "02BM", "02ZW", "xMark"),
    "02BM2": ("00BD", "01WL", "02ZW", "02BLM", "02BM"),
    "02BLM": ("02BM", "02BM2", "02ZW"),

    "10LD": ("12LM"),
    "12LM": ("10LD", "12LLM", "aBHLM"),
    "12LLM": ("12LM"),

    "hHYB": ("hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYH": ("hHYB", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYP": ("hHYB", "hHYH", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYQ": ("hHYB", "hHYH", "hHYP", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYQ2": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYS": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYT": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYV", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYV": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV2", "hHYW", "hHYX", "hHYO"),
    "hHYV2": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYW", "hHYX", "hHYO"),
    "hHYW": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYX", "hHYO"),
    "hHYX": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYO"),
    "hHYO": ("hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT", "hHYV", "hHYV2", "hHYW", "hHYX")
}
ok_ng_class ={
    "ok": ("01BHH", "01BL", "02CJ", "02DY", "02ZW", "aOK", "hHYB", "hHYH", "hHYP", "hHYQ", "hHYQ2", "hHYS", "hHYT",
           "hHYV", "hHYV2", "hHYW", "hHYX", "xDWF", "XFlag", "xKong", "xMark", "xMark2", "xMoer", "xPao", "xPao2"
           ),
    "ng": ("00BD", "01WL", "02BLM", "02BM", "02BM2", "02DBBM", "10LD", "11LL", "12LLM", "12LM", "aBHLM", "xGZ", "xLYJ")
}


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

    img_gray = np.array(img)
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.

    # totensor
    img = transforms.ToTensor()(img_gray)
    # img = transforms.Normalize(config['train_loader']['args']['mean'], config['train_loader']['args']['std'])(img)
    img = img.to(device=device)
    return img_p, img


def main():
    args = parse_arguments()
    config = json.load(open(args.configs))
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

    # 1、-------------------获取数据-----------------
    all_images = get_images(args.images)

    # 2、-------------------获得模型-----------------
    num_classes = config['train_loader']['args']['num_class']
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
    if not os.path.exists(args.outputs_oo):
        os.makedirs(args.outputs_oo)
    if not os.path.exists(args.outputs_others):
        os.makedirs(args.outputs_others)

    # 3、--------------------------------预测-----------------------------
    # 获得labels
    labels_ = {}
    with open(args.labels, 'r') as f:
        for line in f:
            labels_[line.strip().split(":")[0]] = line.strip().split(":")[1]

    # 定义混淆矩阵
    confusion_matrix = [[0 for j in range(num_classes)] for i in range(num_classes)]
    pred_target_list = []   # 存放混淆矩阵中所用到的数据
    pred_target_list_tolerate_count = {}    # 计算容忍混淆矩阵使用

    with torch.no_grad():
        cam_extractor = CAM(model, target_layer='model._conv_head')

        mean = config['train_loader']['args']['mean']
        std = config['train_loader']['args']['std']
        restore_transform = transforms.Compose([ transforms.ToPILImage()])  # local_transforms.DeNormalize(mean, std),

        tbar = tqdm(all_images, ncols=100)
        for img_label in tbar: # imgs_path, label
            img_p = img_label[0]
            img = img_p2img(img_p=img_p, config=config, args=args, device=device)   # # img_p, img
            input_tensor = img[1].unsqueeze(0)
            output_tensor = model(input_tensor)
            prediction = output_tensor.squeeze(0).cpu().detach().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            scores = F.softmax(output_tensor.squeeze(0), dim=0).cpu().numpy().max()
            pred = prediction.item()
            label = img_label[1]

            # 1、预测值小于阈值
            if scores < 0.3:
                # 1. 拷贝图片
                output_name = "pred-{}__target-{}__{}__score-{}.bmp".format(labels_[str(pred)], labels_[label],
                                                                            str(time.time()).split('.')[0] +
                                                                            str(time.time()).split('.')[1],
                                                                            str(scores))
                shutil.copy(os.path.join(img_p), os.path.join(args.outputs_others, output_name))

                # 2. 拷贝直方图
                output_name = output_name[:-4] + ".jpg"
                img_gray = np.array(restore_transform(input_tensor.squeeze(0)))
                rows, cols = img_gray.shape
                flat_gray = img_gray.reshape((cols * rows,)).tolist()
                A = min(flat_gray)
                B = max(flat_gray)
                img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
                cv2.imencode('.jpg', img_gray)[1].tofile(os.path.join(args.outputs_others, output_name))

                # CAM图
                output_name = output_name[:-4] + ".png"
                img_cam = restore_transform(input_tensor.squeeze(0))

                activation_map = cam_extractor(output_tensor.cpu().squeeze(0).argmax().item(), output_tensor.cpu())
                result = overlay_mask(img_cam.convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)

                cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
                    os.path.join(args.outputs_others, output_name))
                continue

            pred_target_list.append((pred, label))  # 记录预测值和标签值，供计算混淆矩阵使用 (int,str)

            if pred != int(label):  # 预测错误， 输出图片、CAM
                # 2、可容忍的类别分错
                if labels_[str(pred)] in tolerate_class.keys() and labels_[label] in tolerate_class[labels_[str(pred)]]:
                    if (pred, int(label)) in pred_target_list_tolerate_count.keys():
                        pred_target_list_tolerate_count[(pred, int(label))] += 1
                    else:
                        pred_target_list_tolerate_count[(pred, int(label))] = 1
                    # 1. 拷贝图片
                    output_name = "pred-{}__target-{}__{}__score-{}.bmp".format(labels_[str(pred)], labels_[label],
                                                                                str(time.time()).split('.')[0] +
                                                                                str(time.time()).split('.')[1],
                                                                                str(scores))
                    shutil.copy(os.path.join(img_p), os.path.join(args.outputs_oo, output_name))

                    # 2. 拷贝直方图
                    output_name = output_name[:-4] + ".jpg"
                    img_gray = np.array(restore_transform(input_tensor.squeeze(0)))
                    rows, cols = img_gray.shape
                    flat_gray = img_gray.reshape((cols * rows,)).tolist()
                    A = min(flat_gray)
                    B = max(flat_gray)
                    img_gray = np.uint8(
                        255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
                    cv2.imencode('.jpg', img_gray)[1].tofile(os.path.join(args.outputs_oo, output_name))

                    # CAM图
                    output_name = output_name[:-4] + ".png"
                    img_cam = restore_transform(input_tensor.squeeze(0))

                    activation_map = cam_extractor(output_tensor.cpu().squeeze(0).argmax().item(), output_tensor.cpu())
                    result = overlay_mask(img_cam.convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)

                    cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
                        os.path.join(args.outputs_oo, output_name))
                else:   # 3、不可容忍的类别分错
                    # 1. 拷贝图片
                    output_name = "pred-{}__target-{}__{}__score-{}.bmp".format(labels_[str(pred)], labels_[label],
                                                                               str(time.time()).split('.')[0]+str(time.time()).split('.')[1],
                                                                               str(scores))
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
            else:   # 4、预测正确
                # 1. 拷贝图片
                output_name = "pred-{}__target-{}__{}__score-{}.bmp".format(labels_[str(pred)], labels_[label],
                                                                  str(time.time()).split('.')[0] +
                                                                  str(time.time()).split('.')[1],
                                                                            str(scores))
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

    # -------------------------------------输出严格混淆矩阵---------------------------------------------------
    print("输出严格混淆矩阵：")
    for p, t in pred_target_list:   # (int,str)
        confusion_matrix[int(t)][p] += 1

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

    # --------------------------------------输出容忍混淆矩阵--------------------------------------------------
    print("输出可容忍混淆矩阵：")
    for k in list(pred_target_list_tolerate_count.keys()):
        if k in pred_target_list_tolerate_count.keys():
            p, t = k[0], k[1]   # int, int
            confusion_matrix[int(p)][p] += int(pred_target_list_tolerate_count[(p, t)])
            confusion_matrix[int(t)][p] -= int(pred_target_list_tolerate_count[(p, t)])
    confusion_file = open(os.path.join(args.outputs, "confusion_t.txt"), 'a+')
    print("{0:10}".format(""), end="")  # 第一行第一个空格
    confusion_file.write("{0:8}".format(""))
    for name in range(num_classes):  # 第一行 所有的类别名称
        print("{0:10}".format(labels_[str(name)]), end="")
        confusion_file.write("{0:8}".format(labels_[str(name)]))
    print("{0:10}".format("Precision"))  # 第一行 每一类的准确率
    confusion_file.write("{0:8}\n".format("Precision"))

    # 第二行以后
    for i in range(num_classes):  # 第N行
        print("{0:10}".format(labels_[str(i)]), end="")  # 第N行，第一列，名称
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

    # ----------------------------计算过检、漏检------------------------------------------------------------
    total_len = len(pred_target_list)
    gjc = 0 # 过检数量 pred NG -- target OK
    ljc = 0 # 漏检数量 pred OK -- target NG
    for pred_target in pred_target_list:    # pred_target_list.append((pred, label))  # 记录预测值和标签值，供计算混淆矩阵使用 (int,str)
        pred_, target_ = pred_target
        # 为每个预测和target标记OK或者NG
        if labels_[str(pred_)] in ok_ng_class["ok"]:
            pred_flag = "ok"
        else:
            pred_flag = "ng"
        if labels_[str(target_)] in ok_ng_class["ok"]:
            target_flag = "ok"
        else:
            target_flag = "ng"

        if pred_flag == "ng" and target_flag == "ok":
            gjc += 1
        elif pred_flag == "ok" and target_flag == "ng":
            ljc += 1

    gjl = float(gjc) / total_len  # 过检率
    ljl = float(ljc) / total_len    # 漏检率

    print("过检率：{}\n漏检率：{}\n".format(gjl, ljl))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--configs', default='configs/Screen_EfficientNetb0_CEL_SGD.json', type=str, help='The configs used to train the model')
    parser.add_argument('-m', '--model', default='pretrained/b0_best.pth', type=str, help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-l', '--labels', default=r'F:\Data\Screen\trainval\labels.txt', type=str, help='label Path')

    parser.add_argument('-i', '--images', default=r'F:\Data\Screen\trainval', type=str, help='Path to the vallist.txt')
    parser.add_argument('-o', '--outputs', default=r'F:\NG', type=str, help='Output Path')
    parser.add_argument('-oc', '--outputs_ok', default=r'F:\OK', type=str, help='Output Path')
    parser.add_argument('-oo', '--outputs_oo', default=r'F:\tolerate', type=str, help='Output Path')
    parser.add_argument('-ooo', '--outputs_others', default=r'F:\others', type=str, help='Output Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
