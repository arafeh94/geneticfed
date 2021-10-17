import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools


def fil(item: str):
    return item.endswith('e5_b50_r500_s10_cifar10_cr02_lr0.001')


for k in fl.accuracies.get_saved_accuracy().keys():
    print(k)
fl.accuracies.show_saved_accuracy_plot(fil)
