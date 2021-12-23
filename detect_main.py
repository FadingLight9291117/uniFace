from typing import Tuple
from pathlib import Path

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
    ...


if __name__ == '__main__':
    weight_path = './models/mobilnet_FINAL.pth'
    img_dir = './pics'

    img_paths = Path(img_dir).glob('*')

    img_path = './pics/000.jpg'

    print('loading model.')

    device = 'cpu'
    net = FaceClassifier()
    weight = torch.load(weight_path)
    net.load_state_dict(weight)

    net.eval().to(device)

    print('preprocess data.')
    # for img_path in img_paths:
    img_path = str(img_path)
    img_tensor, img_meta = preprocess(img_path)
    img_tensor.to(device)

    # print('inference')
    bimap = net(img_tensor)
    # print("postprocess.")

    cls = postprocess(bimap, img_meta)
    print(cls)

    save_result(cls, img_path)
