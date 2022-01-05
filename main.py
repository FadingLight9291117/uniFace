import os
from pathlib import Path
import time
import shutil
from typing import Tuple

import cv2
import torch
import numpy as np

from retinaface.retinaface import RetinaFace
from retinaface.config import cfg_mnet
from faceclassifier.faceclassifier import FaceClassifier
from utils import json2dict


def preprocess(img_path: str, roi=None, resize=640) -> Tuple[torch.Tensor, dict]:
    img = cv2.imread(img_path)

    if roi:
        x1, y1, x2, y2 = roi
        img = img[y1:y2, x1:x2]

    img = cv2.resize(img, (resize, resize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


# todo: 这里需要输出人脸个数
def detect_postprocess(bimap, img_meta: dict, thresh=0.5) -> int:
    loc, conf, landms = bimap
    scores: np.array = conf.squeeze(0).data.cpu().numpy()[:, 1]
    cls = 0 if np.all(scores < thresh) else 1

    return cls


# todo: 这里还需要修改分类模型输出face_type
# facetype: 0 不是人脸
#           1 是人脸
#           2 口罩头盔人脸
def cls_postprocess(bimap: torch.Tensor, img_meta: dict) -> int:
    bimap = bimap.squeeze(0)
    cls = bimap.argmax(0)
    return int(cls)


def decision_making(face_info: dict, cfg: dict = None) -> int:
    face_num = face_info['face_type']
    face_type = face_info['face_type']

    if face_num == 1:
        return 1
    elif face_num >= 1 and cfg['need_face_num_gt_1']:
        return 1
    elif face_num == 0:
        if face_type == 0:
            return 0
        elif face_type == 1:
            return 1
        elif face_type == 2 and cfg['need_helmet_mask']:
            return 1
        else:
            return 0
    else:
        return 0


def save_result(cls, save_path):
    ...


def main():
    detect_weight_path = 'weights/retinaface/mobilenet0.25_Final.pth'
    cls_weight_path = 'weights/mobilnet_FINAL.pth'

    img_path = '/home/clz/dataset/FACE_LYG_2021_12_23/1.jpg'

    cfg_path = './config.json'
    cfg = json2dict(cfg_path)

    # ================= load models ========================
    print('loading models.')
    device = 'cpu'

    detect_net = RetinaFace(cfg=cfg_mnet, phase='test')
    cls_net = FaceClassifier()

    detect_weight = torch.load(detect_weight_path)
    detect_net.load_state_dict(detect_weight)
    detect_net = detect_net.eval().to(device)

    cls_weight = torch.load(cls_weight_path)
    cls_net.load_state_dict(cls_weight)
    cls_net = cls_net.eval().to(device)

    # ================== preprocess data ====================

    print('preprocess data.')
    img_tensor, img_meta = preprocess(img_path)
    img_tensor = img_tensor.to(device)

    # ============= inference + postprocess ===============

    print('inference')
    # step1
    detect_bimap = detect_net(img_tensor)
    face_num = detect_postprocess(detect_bimap, img_meta)
    if face_num >= 1:
        face_info = {
            'face_num': face_num,
            'face_type': 1,  # 1 代表是人脸
        }
    else:
        # step2
        cls_bimap = cls_net(img_tensor)
        face_type = cls_postprocess(cls_bimap, img_meta)
        face_info = {
            'face_num': 0,
            'face_type': face_type,
        }

    # ================== process info =====================
    print('process info.')
    cls = decision_making(face_info, cfg['decison'])

    # ================== save result ======================
    print('save result.')
    save_result(cls, img_path)
