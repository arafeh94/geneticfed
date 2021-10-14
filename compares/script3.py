import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools
print(fl.accuracies.get_saved_accuracy())
plt.plot(list(fl.accuracies.get_saved_accuracy()['genetic_e25_b50_r100_s2_cifar10_cr01_lr01']))

plt.show()