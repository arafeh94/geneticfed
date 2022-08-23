import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
from src import manifest
from src.apis import utils


def norm(arr, low, high):
    return (np.array(arr) - low) / (high - low)


shards = 2
tags = [
    'warmup_e1_b999_r500_s2_mnist_cr01_lr0.01', 'basic_e1_b999_r500_s2_mnist_cr01_lr0.01',
    'genetic_e1_b999_r500_s2_mnist_cr01_lr0.01'
]

acc_file = fl.AccuracyCompare(manifest.COMPARE_PATH + "acc.pkl")
wd_file = fl.DivergenceCompare(manifest.COMPARE_PATH + "div.pkl")

acc = acc_file.get_saved_accuracy(lambda c: 'e1' in c and 's2' in c and 'r500' in c and 'lr0.01' in c)
wds = wd_file.get_saved_divergences(lambda c: 'e1' in c)
print(len(acc), acc.keys())
print(len(wds), wds.keys())

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
