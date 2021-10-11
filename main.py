from dataclasses import dataclass

import torch

from utils import json2dict


@dataclass
class Cntlinf():
    initPath: str


def uniface_initAll(cntlinf: Cntlinf):
    config = json2dict(cntlinf.initPath)

    workingDir = config['workingDir']
    models_info = config['models']
    device = ('cuda'
              if config['device'] == 'gpu' and torch.cuda.is_available()
              else 'cpu')

    models = []
    for info in models_info:
        net_path = f'{workingDir}/{info["net"]}'
        weight_path = f'{workingDir}/{info["weight"]}'
        params = info['params']
        model: torch.nn.Module = torch.load(net_path)
        model.load_state_dict(weight_path)
        model.eval()
        model = model.to(device)

        model_ = {
            'model': model,
            'params': params,
        }
        models.append(model_)

    return models


def uniface_uninitAll(models):
    # TODO: 清理内存
    ...


if __name__ == '__main__':
    ...
