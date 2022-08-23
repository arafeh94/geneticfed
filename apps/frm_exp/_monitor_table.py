import json

import matplotlib.pyplot as plt

from apps.frm_exp import tools
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

plt.rcParams.update({'font.size': 28})

queries = [
    # ['femnist_shard_new', 'r500'],
    # ['femnist_dir_new', 'r500'],
    # ['cifar_dir_new', 'r500'],
    # ['cifar_shard_new', 'r500'],
    # ['mnist', 'logistic', 'dir', 'r1000'],
    # ['mnist', 'logistic', 'shard', 'r1000'],
    # ['mnist', 'logistic', 'unique', 'r1000'],
    ['mnist', 'logistic_e', 'r1000'],

]

for query in queries:
    plts_configs = tools.filter(*query)
    tools.preprocess(plts_configs)
    print(plts_configs)
    for table in plts_configs:
        print(plts_configs[table])
        print('acc', tools.get_info(table, 'acc')[-1], tools.get_info(table, 'loss')[-1])
