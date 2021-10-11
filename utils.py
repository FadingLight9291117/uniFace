import json

from easydict import EasyDict as edict

def json2dict(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d

def json2edict(path):
    return edict(json2dict(path))