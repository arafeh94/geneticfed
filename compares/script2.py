import collections
import operator
import pickle

import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools
from src import manifest

accs = fl.AccuracyCompare(manifest.COMPARE_PATH + 'acc.pkl')
acc_cl = fl.AccuracyCompare(manifest.COMPARE_PATH + 'acc_cl.pkl')

[print(v) for v in accs.get_saved_accuracy()]


filts = [
    # lambda item: item.endswith('_e1_b999_r500_s2_mnist_cr01_lr0.001'),
    lambda item: item == 'basic_e1_b999_r500_s10_mnist_cr01_lr0.1',
    lambda item: item == 'genetic_e1_b999_r500_s10_mnist_cr01_lr0.001',
    lambda item: item == 'warmup_e1_b999_r500_s10_mnist_cr01_lr0.001',
]


all = {}

for fil in filts:
    all = {**all, **accs.get_saved_accuracy(fil)}



accs.show_saved_accuracy_plot_acc(all)
