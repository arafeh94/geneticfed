import numpy as np
from matplotlib import pyplot as plt

from libs.model.linear.lr import LogisticRegression
from src.apis import utils
from src.apis.extensions import CycleList, TorchModel
from src.data.data_loader import mnist
from src.federated.components.trainers import TorchTrainer

array = {1: 0, 2: 0}
dd = array[-1]
dd += 1
print(dd)
print(array)
