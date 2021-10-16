import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools


def fil(item: str):
    return item.endswith('e25_b50_r200_s4_cifar10_cr01_lr0.001')


print(fl.accuracies.get_saved_accuracy().keys())
fl.accuracies.show_saved_accuracy_plot(fil)

plt.show()
