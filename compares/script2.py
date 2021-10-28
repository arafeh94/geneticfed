import collections
import operator
import pickle

import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools
from src import manifest

[print(v) for v in fl.accuracies.get_saved_accuracy()]

accs = fl.accuracies


def fil(item: str):
    return item.endswith('_e25_b50_r500_s10_mnist_cr02_lr0.1')


filts = [
    fil,
    # lambda x: x.endswith('_e25_b50_r500_s10_mnist_cr01_lr0.1'),
    # lambda x: x == 'warmup_e1_b9999_r1000_s4_cifar10_cr02_lr0.001'
]

all = {}

for fil in filts:
    all = {**all, **accs.get_saved_accuracy(fil)}

accs.show_saved_accuracy_plot_acc(all)
