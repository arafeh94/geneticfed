import numpy as np
import torch
from src.data.data_provider import PickleDataProvider

print('loading data')
data = PickleDataProvider('../../datasets/pickles/kdd.pkl').collect()

print(f'data sample size {len(data)}')
labels = np.unique(data.y)
print(f'data labels {len(labels)} , {labels}')
for label in labels:
    print(f'label [{label}] of size [{np.count_nonzero(data.y == label)}]')
