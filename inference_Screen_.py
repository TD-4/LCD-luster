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


def get_images(path=""):
    imgs_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name[-4:] == ".bmp":
                imgs_path.append(os.path.join(path, root, name))
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
    return img_p, img



def main():
    args = parse_arguments()
    config = json.load(open(args.configs))
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

    # 1、-------------------获取数据-----------------
    all_images = get_images(args.images)

    # 2、-------------------获得模型-----------------
    num_classes = 21
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    # Load checkpoint
    checkpoint = torch.load(args.model)
    # load
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    if not os.path.exists(args.outputs):
        os.makedirs(args.outputs)
    if not os.path.exists(args.outputs_cam):
        os.makedirs(args.outputs_cam)

    # 3、--------------------------------预测-----------------------------
    with open(args.labels, 'r') as f:
        for line in f:
            temp_path = os.path.join(args.outputs, line.strip().split(":")[0]+"_"+line.strip().split(":")[1])
            temp_path2 = os.path.join(args.outputs_cam, line.strip().split(":")[0]+"_"+line.strip().split(":")[1])
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            if not os.path.exists(temp_path2):
                os.makedirs(temp_path2)

    with torch.no_grad():
        cam_extractor = CAM(model, target_layer='model._conv_head')
        tbar = tqdm(all_images, ncols=100)
        for img_p in tbar:
            img = img_p2img(img_p=img_p, config=config, args=args, device=device)
            input = img[1].unsqueeze(0)
            input_p = img[0]
            output = model(input)
            prediction = output.squeeze(0).cpu().detach().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

            # CAM图
            mean = config['train_loader']['args']['mean']
            std = config['train_loader']['args']['std']
            restore_transform = transforms.Compose([local_transforms.DeNormalize(mean, std), transforms.ToPILImage()])
            img_o = restore_transform(img[1])

            activation_map = cam_extractor(output.cpu().squeeze(0).argmax().item(), output.cpu())
            result = overlay_mask(img_o.convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)

            # 拷贝预测图片
            for path in os.listdir(args.outputs):
                path_id = path.split("_")
                if prediction.item() == int(path_id[0]):
                    shutil.copy(os.path.join(input_p),
                                os.path.join(args.outputs, path))
                    cv2.imencode('.jpg', np.array(result)[:, :, ::-1])[1].tofile(os.path.join(args.outputs, path, input_p.split('\\')[-1][:-4] + ".jpg"))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--configs', default='configs/Middle_EfficientNetb5_CEL_SGD.json', type=str, help='The configs used to train the model')
    parser.add_argument('-m', '--model', default='pretrained/b5_best_model.pth', type=str, help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r"E:\Dataset\点灯数据\全分类Test-Middle", type=str, help='Path to the images to be segmented')
    parser.add_argument('-o', '--outputs', default=r'F:\20210627_aug\middle', type=str, help='Output Path')
    parser.add_argument('-oc', '--outputs_cam', default=r'F:\20210627_aug\middle_cam', type=str, help='Output Path')
    parser.add_argument('-l', '--labels', default=r'F:\Data\lcd\midle\midle\labels.txt', type=str, help='label Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
