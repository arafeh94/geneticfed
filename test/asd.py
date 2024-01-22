import numpy as np
from matplotlib import pyplot as plt

from src.data.data_loader import mnist

dt = mnist()
for x, y in dt:
    x = np.reshape(x, (28, 28))
    plt.imshow(x, cmap='viridis')
    plt.show()
    print(y)
    break
