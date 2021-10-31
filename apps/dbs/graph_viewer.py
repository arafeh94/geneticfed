import matplotlib.pyplot as plt
import numpy as np

from src import manifest
from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


graphs = Graphs(FedDB('../experiments/perf.db'))
print(graphs)
graphs.plot([
    # {
    #     'session_id': 'cifar',
    #     'field': 'acc',
    #     'config': {'color': 'b'},
    #     'transform': utils.smooth
    # },
    {
        'session_id': 'cifar',
        'field': 'wd',
        'config': {'color': 'r'},
        # 'transform': normalize
    },
], animated=True)
