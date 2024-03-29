import math

import numpy as np
import torch
from torch import nn


def influence_ecl(model1, model2):
    all = []
    for key in model1.keys():
        l2_norm = torch.dist(model1[key], model2[key], 2)
        val = l2_norm.numpy().min()
        all.append(val)
    return math.fsum(all) / len(all)


def influence_cos(model1, model2, center):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    center = torch.flatten(center)
    p1 = torch.flatten(model1)
    p2 = torch.flatten(model2)
    p1 = torch.subtract(center, p1)
    p2 = torch.subtract(center, p2)
    return cos(p1, p2).numpy().min()


def influence_cos2(model1, model2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    x1 = torch.flatten(model1)
    x2 = torch.flatten(model2)
    return cos(x1, x2).numpy().min()

def normalize(arr, z1=False):
    if z1:
        return (arr - min(arr)) / (max(arr) - min(arr))
    return np.array(arr) / math.fsum(arr)


def smooth(vals, sigma=2):
    from scipy.ndimage import gaussian_filter1d
    return list(gaussian_filter1d(vals, sigma=sigma))
