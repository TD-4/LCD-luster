# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.8.23
# @github:https://github.com/felixfu520

import argparse
import torch

from PIL import Image
from torchvision.transforms import ToTensor
import json
import numpy as np
import models


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def parse_arguments():
    # exporter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', default='configs/Screen_EfficientNetb4_CEL_SGD.json', type=str,
                        help='Path to the configs file (default: configs.json)')
    parser.add_argument('--model', type=str, default='pretrained/b4_best.pth', help="set model checkpoint path")
    parser.add_argument('--model_out', type=str, default='pretrained/b4_best.onnx')
    parser.add_argument('--image', type=str, default="", help='input image to use')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    config = json.load(open(args.configs))
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    print('running on device ' + str(device))

    # ----------------------------load the image
    if args.image:
        img = Image.open(args.image)
        img_to_tensor = ToTensor()
        input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0]).to(device)
    else:
        pixels = 224
        input = torch.zeros([1, 1, 224, 224], dtype=torch.float32).to(device)
    print("input size is..", input.shape)

    # ----------------------------load the model
    num_classes = 33
    model = get_instance(models, 'arch', config, num_classes).to(device)  # 定义模型
    # model.set_set_swish(memory_efficient=False)

    checkpoint = torch.load(args.model)
    base_weights = checkpoint["state_dict"]   # module.model.conv1.weight格式，而model的权重是model.conv1.weight格式
    print('Loading base network...')

    from collections import OrderedDict  # 导入此模块
    # new_state_dict = OrderedDict()
    # for k, v in base_weights.items():
    #     name = k[7:]  # remove `module.`，即只取module.model.conv1.weights的后面几位
    #     new_state_dict[name] = v

    model.load_state_dict(base_weights)  # 加载权重
    model.eval()
    print("loaded weight")

    # ----------------------------export the model
    input_names = ["input_0"]
    output_names = ["output_0"]

    print('exporting model to ONNX...')
    torch.onnx.export(model, input, args.model_out, verbose=True, input_names=input_names, output_names=output_names, opset_version=9)
    print('model exported to {:s}'.format(args.model_out))
