from typing import Tuple
from pathlib import Path

import torch
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

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


def metrics_fn(preds, targs):
    acc = accuracy_score(targs, preds)
    prec = precision_score(targs, preds)
    recall = recall_score(targs, preds)
    metrics = {
        'acc': float(f'{acc:.3f}'),
        'prec': float(f'{prec:.3f}'),
        'recall': float(f'{recall:.3f}'),
    }
    return metrics


if __name__ == '__main__':
    weight_path = './models/faceCla.weight'
    imgP_dir = "./dataset/p"
    imgN_dir = "./dataset/n"

    print('loading model.')
    device = 'cuda'
    net = FaceClassifier()
    weight = torch.load(weight_path)
    net.load_state_dict(weight)
    net = net.eval().to(device)

    print("prepare dataset")
    imgP_paths = [str(i) for i in Path(imgP_dir).glob('*')]
    imgN_paths = [str(i) for i in Path(imgN_dir).glob('*')]

    img_paths = imgP_paths + imgN_paths
    targs = [1] * len(imgP_paths) + [0] * len(imgN_paths)

    print('begin detect.')
    preds = []
    img_paths = tqdm(img_paths)
    for img_path in img_paths:
        img_tensor, img_meta = preprocess(img_path)
        img_tensor = img_tensor.to(device)

        bimap = net(img_tensor)
        cls = postprocess(bimap, img_meta)
        preds.append(cls)

    print('end detect.')
    preds = np.array(preds)
    targs = np.array(targs)
    metrics = metrics_fn(preds, targs)
    print(metrics)
