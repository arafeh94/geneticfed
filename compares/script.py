import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools
from src import manifest
from src.apis import utils


def norm(arr, low, high):
    return (np.array(arr) - low) / (high - low)


shards = 2
tags = [
    'genetic_e25_b50_r200_s2_mnist_cr02_lr0.01', 'basic_e25_b50_r200_s2_mnist_cr02_lr0.01',
    'warmup_e25_b50_r200_s2_mnist_cr02_lr0.01'
]

acc = fl.AccuracyCompare(manifest.COMPARE_PATH + "acc.pkl").get_saved_accuracy()
wds = fl.DivergenceCompare(manifest.COMPARE_PATH + "div.pkl").get_saved_divergences()
wds['genetic_e25_b50_r200_s2_mnist_cr01_lr0.01'] = wds['genetic_e25_b50_r200_s2_mnist_cr01_lr0.01']

print(acc.keys())
print(wds.keys())
low = min([min(wds[tag]) for tag in tags])
high = max([max(wds[tag]) for tag in tags])

colors = ['b', '#117733', '#DDCC77']

for index, tag in enumerate(tags):
    plt.plot(utils.smooth(norm(wds[tag], low, high))[:30], '--', color=colors[index], label='ACC')
    plt.plot(utils.smooth(acc[tag])[:30], color=colors[index], label="WD")
    plt.title('Non-IID Ψ=2')
    plt.legend()
    plt.xlabel(tag.split('_')[0].capitalize())
    plt.show()
