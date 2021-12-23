from typing import Tuple
import itertools
from pathlib import Path
import shutil

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


if __name__ == '__main__':
    weight_path = 'weights/retinaface/mobilenet0.25_Final.pth'

    img_paths = [
        '/home/clz/dataset/人脸2021_12_23/face/1',
        '/home/clz/dataset/人脸2021_12_23/face/2',
        '/home/clz/dataset/人脸2021_12_23/face/3'
    ]
    img_paths = itertools.chain(*[Path(i).glob('*') for i in img_paths])

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
