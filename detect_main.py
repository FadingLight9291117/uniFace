from typing import Tuple
import time
import itertools
from pathlib import Path
import shutil

import torch
import cv2
import numpy as np

from retinaface.retinaface import RetinaFace
from retinaface.config import cfg_mnet


def preprocess(img_path: str, roi=None) -> Tuple[torch.Tensor, dict]:
    img = cv2.imread(img_path)

    if roi:
        x1, y1, x2, y2 = roi
        img = img[y1:y2, x1:x2]

    # img = cv2.resize(img, (300, 300))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ! 是否需要
    img = np.float32(img)
    h, w, c = img.shape
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    img_meta = {
        'img_path': img_path,
        'height': h,
        'width': w,
        'channel': c,
    }

    return img, img_meta


def postprocess(bimap, img_meta: dict) -> int:
    thresh = 0.5  # 越大越灵敏
    loc, conf, landms = bimap
    scores: np.array = conf.squeeze(0).data.cpu().numpy()[:, 1]
    cls = 0 if np.all(scores < thresh) else 1

    return cls


def save_result(cls: int, img_path: str):
    img_path = Path(img_path)
    dir_name = img_path.parent.name
    file_name = img_path.name
    if cls == 1:
        shutil.copy(img_path.__str__(),
                    f'result3/1/{dir_name}_{file_name}')
    else:
        shutil.copy(img_path.__str__(),
                    f'result3/0/{dir_name}_{file_name}')


def main():
    weight_path = 'weights/retinaface/mobilenet0.25_Final.pth'
    img_dir = '/home/clz/dataset/FACE_LYG_2021_12_23/'

    img_paths = Path(img_dir).glob('*')

    print('loading model.')
    device = 'cpu'
    net = RetinaFace(cfg=cfg_mnet, phase='test')
    weight = torch.load(weight_path)
    net.load_state_dict(weight)
    net = net.eval().to(device)

    for img_path in img_paths:
        img_path = str(img_path)
        # print('preprocess data.')
        img_tensor, img_meta = preprocess(img_path)
        img_tensor = img_tensor.to(device)
        # print('inference')
        bimap = net(img_tensor)
        # print("postprocess.")
        cls = postprocess(bimap, img_meta)
        print(cls)
        save_result(cls, img_path)


if __name__ == '__main__':
    weight_path = 'weights/retinaface/mobilenet0.25_Final.pth'
    img_dir = '/home/clz/dataset/FACE_LYG_2021_12_23/'

    img_paths = list(Path(img_dir).glob('*'))

    print('loading model.')
    device = 'cpu'
    net = RetinaFace(cfg=cfg_mnet, phase='test')
    weight = torch.load(weight_path)
    net.load_state_dict(weight)
    net = net.eval().to(device)

    img1s = []
    img0s = []
    tb = time.time()
    for img_path in img_paths:
        img_path = str(img_path)
        # print('preprocess data.')
        img_tensor, img_meta = preprocess(img_path)
        img_tensor = img_tensor.to(device)
        # print('inference')
        bimap = net(img_tensor)
        # print("postprocess.")
        cls = postprocess(bimap, img_meta)
        # print(cls)
        # save_result(cls, img_path)
        if cls == 0:
            img0s.append(img_path)
        else:
            img1s.append(img_path)
    te = time.time()
    print(len(img0s), len(img1s))
    print('time:', (te - tb) / len(img_paths))

    # save
    save_path = Path('./results/detect')
    save_path0 = save_path / '0'
    save_path1 = save_path / '1'
    save_path0.mkdir(exist_ok=True, parents=True)
    save_path1.mkdir(exist_ok=True, parents=True)

    for i in img0s:
        shutil.copy(i, save_path0)
    for j in img1s:
        shutil.copy(j, save_path1)
