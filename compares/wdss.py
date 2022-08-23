import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB


def norm(arr, low, high):
    return (np.array(arr) - low) / (high - low)


shards = 2

# warmup_e1_b999_r500_s2_mnist_cr01_lr0.01
# basic_e1_b999_r500_s2_mnist_cr01_lr0.01
# genetic_e1_b999_r500_s2_mnist_cr01_lr0.01
# warmup_e25_b50_r500_s10_mnist_cr01_lr0.1
# genetic_e25_b50_r500_s10_mnist_cr01_lr0.1
# basic_e25_b50_r500_s10_mnist_cr01_lr0.1
tags = [
    'warmup_e1_b999_r500_s2_mnist_cr01_lr0.01', 'basic_e1_b999_r500_s2_mnist_cr01_lr0.01',
    'genetic_e1_b999_r500_s2_mnist_cr01_lr0.01'
]

db = FedDB('perf.db')
acc = {
    'warmup_e1_b999_r500_s2_mnist_cr01_lr0.01': db.acc('t30c8c7f98fb508e290766417bcf275ce'),
    'basic_e1_b999_r500_s2_mnist_cr01_lr0.01': db.acc('t0f0a2f2bd283f3b70b000e623b21083f'),
    'genetic_e1_b999_r500_s2_mnist_cr01_lr0.01': db.acc('t6ea0cd46853f4953a775527f47038ec2'),
}
wds = {
    'warmup_e1_b999_r500_s2_mnist_cr01_lr0.01': db.get('t30c8c7f98fb508e290766417bcf275ce', 'wd'),
    'basic_e1_b999_r500_s2_mnist_cr01_lr0.01': db.get('t0f0a2f2bd283f3b70b000e623b21083f', 'wd'),
    'genetic_e1_b999_r500_s2_mnist_cr01_lr0.01': db.get('t6ea0cd46853f4953a775527f47038ec2', 'wd'),
}

gb = Graphs(db)
gb.plot([
    {
        'session_id': 't30c8c7f98fb508e290766417bcf275ce',
        'field': 'wd',
        'config': {'color': 'r', 'label': 'warmup'},
        'transform': utils.smooth
    },
    {
        'session_id': 't0f0a2f2bd283f3b70b000e623b21083f',
        'field': 'wd',
        'config': {'color': 'g', 'label': 'basic'},
        'transform': utils.smooth
    },
    {
        'session_id': 't6ea0cd46853f4953a775527f47038ec2',
        'field': 'wd',
        'config': {'color': 'b', 'label': 'genetic'},
        'transform': utils.smooth
    }
])
exit()
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
