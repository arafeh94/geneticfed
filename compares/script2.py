import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools

print(fl.accuracies.get_saved_accuracy())
fl.accuracies.show_saved_accuracy_plot()

plt.show()
