from dataclasses import dataclass
from pathlib import Path
import os

import torch
import cv2


from utils import json2dict


@dataclass
class Cntlinf:
    initPath: str


@dataclass
class ClsDiaginf:
    path: str
    _option: list
    W: int
    H: int
    C: int
    cls: int  # 0无人脸，1有人脸，2不确定


def uniface_init(cntlinf: Cntlinf):
    config = json2dict(cntlinf.initPath)

    workingDir = config['workingDir']
    models_info = config['recog']
    device = ('cuda'
              if config['device'] == 'gpu' and torch.cuda.is_available()
              else 'cpu')

    net_path = f'{workingDir}/{models_info["net"]}'
    weight_path = f'{workingDir}/{models_info["weight"]}'
    params = models_info['params']
    model: torch.nn.Module = torch.load(net_path)
    model.load_state_dict(weight_path)
    model.eval()
    model = model.to(device)

    model_ = [
        model,
        params,
    ]

    return model_


def uniface_uninitAll(models):
    # TODO: 清理内存
    ...


def recog_diag(diag: ClsDiaginf, model):
    img = cv2.imread(diag)
    diag.H, diag.W, diag.C = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.cuda()
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    pred = model(img).squeeze(0)
    thress = diag._option
    if pred[1] >= thress[1]:
        diag.cls = 1
    elif pred[1] < thress[0]:
        diag.cls = 0
    else:
        diag.cls = 2

    return diag


if __name__ == '__main__':
    # get imgs dir
    config_path = './config.json'
    config = json2dict(config_path)
    img_dir = config['img_dir']

    cntl_inf = Cntlinf(config_path)

    model, params = uniface_init(cntl_inf)

    diags = []

    for img_path in Path(img_dir).glob('*'):
        diag = ClsDiaginf(str(img_path), params)
        diag = recog_diag(diag, model)
        diags.append(diag)

    uniface_uninitAll()
