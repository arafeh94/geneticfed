import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools

fl.ACCURACY_PATH = fl.ACCURACY_PATH + '../cifar200rounds'


def fil(item: str):
    return item.endswith('e1_b9999_r1000_s2_cifar10_cr02_lr0.001')


for k in fl.accuracies.get_saved_accuracy().keys():
    print(k)
fl.accuracies.show_saved_accuracy_plot(fil)
