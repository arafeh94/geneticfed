import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools


def norm(arr, low, high):
    return (np.array(arr) - low) / (high - low)


def tag_mapper(tag):
    return {
        'basic_sgd10': 'basic_e100_b50_r50_s10_mnist_cr01_lr01',
        'genetic_sgd10': 'genetic_e100_b50_r50_s10_mnist_cr01_lr01',
        'basic_sgd2': 'basic_e100_b50_r50_s2_mnist_cr01_lr01',
        'warmup_sgd2': 'warmup_e100_b50_r50_s2_mnist_cr01_lr01',
        'warmup_sgd10': 'warmup_e100_b50_r50_s10_mnist_cr01_lr01',
        'genetic_sgd2': 'genetic_e100_b50_r50_s2_mnist_cr01_lr01',
    }[tag]


shards = 2
tags = [
    f'basic_sgd{shards}', f'warmup_sgd{shards}', f'genetic_sgd{shards}'
]

acc = fl.accuracies.get_saved_accuracy()
wds = fl.divergences.get_saved_divergences()

low = min([min(wds[tag]) for tag in tags])
high = max([max(wds[tag]) for tag in tags])

colors = ['#DDCC77', '#117733', 'b']

for index, tag in enumerate(tags):
    plt.plot(norm(wds[tag], low, high)[0:10], 'o--', color=colors[index], label=f'wd: {tag.replace("_sgd2", "")}')
for index, tag in enumerate(tags):
    plt.plot(acc[tag_mapper(tag)][0:10], color=colors[index], label=f'acc: {tag.replace("_sgd2", "")}')

plt.ylim(ymin=0)
plt.legend()
plt.title('Non-IID Ψ=2')
plt.show()
