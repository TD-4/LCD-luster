# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.23
# @github:https://github.com/felixfu520

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
import re


def get_images(path=""):
    imgs_path = []
    for root, dirs, files in os.walk(path, topdown=True):
        for dir_ in dirs:
            if dir_ == "Edge" or dir_ == "Midle":
                for root_, dirs_, files_ in os.walk(os.path.join(os.path.join(root, dir_)), topdown=True):
                    for name in files_:
                        if name[-4:] == ".bmp" and not re.search("Ori", name) and not re.search("ORI", name):
                            imgs_path.append(os.path.join(path, root_, name))
            else:
                print("skip {} folder".format(os.path.join(root, dir_)))
                continue

    return imgs_path


def get_images2(path=""):
    imgs_path = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            if name[-4:] == ".bmp" and not re.search("Ori", name) and not re.search("ORI", name):
                imgs_path.append(os.path.join(path, root, name))

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
    all_images = get_images2(args.images)

    # 2、-------------------获得模型-----------------
    num_classes = config['arch']['num_class']
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    # Load checkpoint
    checkpoint = torch.load(args.model)
    # load
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    if not os.path.exists(args.outputs):
        os.makedirs(args.outputs)
    if not os.path.exists(os.path.join(args.outputs, "255_other")):
        os.makedirs(os.path.join(args.outputs, "255_other"))
    # 3、--------------------------------预测-----------------------------
    with open(args.labels, 'r') as f:
        for line in f:
            temp_path = os.path.join(args.outputs, line.strip().split(":")[0]+"_"+line.strip().split(":")[1])
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)

    with torch.no_grad():
        cam_extractor = CAM(model, target_layer='model._conv_head')
        # tbar = tqdm(all_images, ncols=100)
        for img_p in all_images:
            try:
                img = img_p2img(img_p=img_p, config=config, args=args, device=device)   # tensor, img_p
            except Exception as e:
                print(e)
                continue
            input = img[1].unsqueeze(0)
            input_p = img[0]
            output = model(input)
            prediction = output.squeeze(0).cpu().detach().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            pred_topk = torch.topk(F.softmax(torch.from_numpy(output.squeeze(0).cpu().detach().numpy()), dim=0), k=2)   # 获得topk 2；values=tensor([0.3845, 0.1654]),indices=tensor([ 9, 20]))
            print("\ntop1:{}  top2:{} ;".format(pred_topk[0][0].item(), pred_topk[0][1].item()))
            # CAM图
            mean = config['train_loader']['args']['mean']
            std = config['train_loader']['args']['std']
            restore_transform = transforms.Compose([local_transforms.DeNormalize(mean, std), transforms.ToPILImage()])
            img_o = restore_transform(img[1])

            activation_map = cam_extractor(output.cpu().squeeze(0).argmax().item(), output.cpu())
            result = overlay_mask(img_o.convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)

            # ---top1置信度<0.3--------------
            if pred_topk[0][0].item() < 0.2:
                print("copy other ...")
                # 1、拷贝原图
                shutil.copy(os.path.join(input_p), os.path.join(args.outputs, "255_other", img_p.split("\\")[-1]))
                # 2. 拷贝直方图
                img_gray = np.array(img_o)
                rows, cols = img_gray.shape
                flat_gray = img_gray.reshape((cols * rows,)).tolist()
                A = min(flat_gray)
                B = max(flat_gray)
                img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
                cv2.imencode('.jpg', img_gray)[1].tofile(
                    os.path.join(args.outputs, "255_other", input_p.split('\\')[-1][:-4] + ".jpg"))
                # 3、拷贝CAM图
                cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
                    os.path.join(args.outputs, "255_other", input_p.split('\\')[-1][:-4] + ".png"))
                continue

            # -top1置信度>0.3 拷贝预测图片、拉伸图、CAM图---------------------
            print("copy ...")
            for path in os.listdir(args.outputs):
                path_id = path.split("_")
                # top1>0.3时
                if prediction.item() == int(path_id[0]):
                    # 1、拷贝原图
                    shutil.copy(os.path.join(input_p), os.path.join(args.outputs, path))
                    # 2. 拷贝直方图
                    img_gray = np.array(img_o)
                    rows, cols = img_gray.shape
                    flat_gray = img_gray.reshape((cols * rows,)).tolist()
                    A = min(flat_gray)
                    B = max(flat_gray)
                    img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
                    cv2.imencode('.jpg', img_gray)[1].tofile(os.path.join(args.outputs, path, input_p.split('\\')[-1][:-4] + ".jpg"))
                    # 3、拷贝CAM图
                    cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(os.path.join(args.outputs, path, input_p.split('\\')[-1][:-4] + ".png"))
                # top2也大于0.3时
                if pred_topk[0][1].item() > 0.48 and pred_topk[1][1].item() == int(path_id[0]):     # top2的置信度判断
                    print("\n{} image 预测为{} 的分数超过0.48，也放到{}中".format(os.path.join(img_p), path, path))
                    # 1、拷贝原图
                    shutil.copy(os.path.join(input_p), os.path.join(args.outputs, path))
                    # 2. 拷贝直方图
                    img_gray = np.array(img_o)
                    rows, cols = img_gray.shape
                    flat_gray = img_gray.reshape((cols * rows,)).tolist()
                    A = min(flat_gray)
                    B = max(flat_gray)
                    img_gray = np.uint8(
                        255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
                    cv2.imencode('.jpg', img_gray)[1].tofile(
                        os.path.join(args.outputs, path, input_p.split('\\')[-1][:-4] + ".jpg"))
                    # 3、拷贝CAM图
                    cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
                        os.path.join(args.outputs, path, input_p.split('\\')[-1][:-4] + ".png"))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--configs', default='configs/Screen_EfficientNetb4_CEL_SGD.json', type=str, help='The configs used to train the model')
    parser.add_argument('-m', '--model', default='pretrained/b4_best.pth', type=str, help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r"F:\Data\Screen\TODO", type=str, help='Path to the images to be segmented')
    parser.add_argument('-o', '--outputs', default=r'F:\TODO', type=str, help='Output Path')
    parser.add_argument('-l', '--labels', default=r'F:\Data\Screen\trainval_notime_20210820\labels.txt', type=str, help='label Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
