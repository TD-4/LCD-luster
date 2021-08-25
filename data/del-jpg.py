# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import numpy as np
import cv2


def del_jpg(path=""):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if file[-4:] == ".jpg" or file[-4:] == ".png":
                print("delete {}".format(os.path.join(root, file)))
                os.remove(os.path.join(root, file))


def gen_zengqiang(path=""):
    all_images = os.listdir(path)
    for img_p in all_images:
        img_gray = cv2.imdecode(np.fromfile(os.path.join(path,img_p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        rows, cols = img_gray.shape
        flat_gray = img_gray.reshape((cols * rows,)).tolist()
        A = min(flat_gray)
        B = max(flat_gray)
        img_gray = np.uint8(255 / (B - A + 0.001) * (img_gray - A) + 0.5)  # ((pixel – min) / (max – min))*255.
        cv2.imencode('.jpg', img_gray)[1].tofile(os.path.join(path, img_p[:-4] + ".jpg"))


def rename_(path=""):
    label_path = os.path.join(path, "labels.txt")
    labels = {}
    with open(label_path, 'r') as f:
        for line in f:
            if line is None: continue
            labels[line.strip().split(":")[0]] = line.strip().split(":")[1]

    for k in labels:
        os.rename(os.path.join(path, labels[k]), os.path.join(path, k + "_" + labels[k]))


if __name__ == "__main__":
    # paths = ['00BD', '01BHH', '01BL', '01WL', '02BLM',
    #         '02BLM2', '02BM', '02BM2', '02CJ', '02DBBM',
    #         '02DY', '02ZW', '10LD', '11LL', '12LLM',
    #         '12LM', 'aBHLM', 'aOK', 'hHYB', 'hHYH',
    #         'hHYO', 'hHYP', 'hHYQ', 'hHYQ2', 'hHYS',
    #         'hHYT', 'hHYV', 'hHYV2', 'hHYW', 'hHYX',
    #         'xDWF', 'xFlag', 'xGZ', 'xKong', 'xLYJ',
    #         'xMark', 'xMark2', 'xMoer', 'xPao', 'xPao2',
    #         'xPao3']
    # for p in paths:
    #     path = os.path.join(r"F:\Data\Screen\20210820", p)
    #     del_jpg(path)
    #     gen_zengqiang(path)

    # rename_(path)
    path = r"F:\TODO\24_xDWF"
    del_jpg(path)
    gen_zengqiang(path)