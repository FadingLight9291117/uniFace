from typing import Tuple
from pathlib import Path
import shutil
import itertools

import torch
import cv2
import numpy as np

from model import FaceClassifier


def preprocess(img_path: str) -> Tuple[torch.Tensor, dict]:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    img_meta = {
        'img_path': img_path,
        'height': h,
        'width': w,
        'channel': c,
    }

    return img, img_meta


def postprocess(bimap: torch.Tensor, img_meta: dict) -> int:
    bimap = bimap.squeeze(0)
    cls = bimap.argmax(0)
    return int(cls)


def save_result(cls: int, img_path: str):
    img_path = Path(img_path)
    dir_name = img_path.parent.name
    file_name = img_path.name
    if cls == 1:
        shutil.copy(img_path.__str__(),
                    f'result/1/{dir_name}_{file_name}')
    else:
        shutil.copy(img_path.__str__(),
                    f'result/0/{dir_name}_{file_name}')


if __name__ == '__main__':
    weight_path = 'weights/mobilnet_FINAL.pth'

    img_paths = ['/home/clz/dataset/人脸2021_12_23/face/1',
                 '/home/clz/dataset/人脸2021_12_23/face/2',
                 '/home/clz/dataset/人脸2021_12_23/face/3']
    img_paths = itertools.chain(*[Path(i).glob('*') for i in img_paths])

    print('loading model.')
    device = 'cpu'
    net = FaceClassifier()
    weight = torch.load(weight_path)
    net.load_state_dict(weight)

    net = net.eval().to(device)

    for img_p in img_paths:
        img_p = str(img_p)
        # print('preprocess data.')
        img_tensor, img_meta = preprocess(img_p)
        img_tensor = img_tensor.to(device)
        # print('inference')
        bimap = net(img_tensor)
        # print("postprocess.")
        cls = postprocess(bimap, img_meta)
        print(cls)
        save_result(cls, img_p)
