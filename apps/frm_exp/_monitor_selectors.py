import json

import matplotlib.pyplot as plt

from apps.frm_exp import tools
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

plt.rcParams.update({'font.size': 28})

queries = [
    ['mnist_selector_', '_2', 'label'],
]

for query in queries:
    tools.pretty(tools.filter(*query))
    plts_configs = tools.filter(*query)
    tools.sett(plts_configs, 't63ec0410948d1ebcbfec98221e10b9a2', 'Random')
    tools.sett(plts_configs, 't6b990442bef5e41f706847ee03e4313d', 'Cluster')
    file_name = "-".join(query)
    tools.plot(tools.plot_builder('acc', plts_configs), f'{file_name}.png', False)
