# config.py
from easydict import EasyDict as edict

cfg_mnet = edict(
    name='mobilenet0.25',
    return_layers=dict(
        stage1=1,
        stage2=2,
        stage3=3,
    ),
    in_channel=32,
    out_channel=64,
)

cfg_re50 = edict(
    name='Resnet50',
    return_layers=dict(
        layer2=1,
        layer3=2,
        layer4=3,
    ),
    in_channel=256,
    out_channel=256,
)
