from typing import Tuple
from pathlib import Path

import torch
import cv2
import numpy as np

from retinaface.retinaface import RetinaFace
from retinaface.config import cfg_mnet


def preprocess(img_path: str) -> Tuple[torch.Tensor, dict]:
    img = cv2.imread(img_path)
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
    thresh = 0.6
    loc, conf, landms = bimap
    scores: np.array = conf.squeeze(0).data.cpu().numpy()[:, 1]
    cls = 0 if np.all(scores < thresh) else 1

    return cls


def save_result(cls: int, img_path: str):
    ...


if __name__ == '__main__':
    weight_path = './models/retinaface/mobilenet0.25_Final.pth'
    img_dir = './pics'

    img_paths = Path(img_dir).glob('*')

    img_path = './pics/FACE_4.jpg'

    print('loading model.')

    device = 'cpu'
    net = RetinaFace(cfg=cfg_mnet, phase='test')
    weight = torch.load(weight_path)
    net.load_state_dict(weight)

    net.eval().to(device)

    print('preprocess data.')
    # for img_path in img_paths:
    img_path = str(img_path)
    img_tensor, img_meta = preprocess(img_path)
    img_tensor.to(device)

    print('inference')
    bimap = net(img_tensor)
    print("postprocess.")

    cls = postprocess(bimap, img_meta)
    print(cls)

    save_result(cls, img_path)
