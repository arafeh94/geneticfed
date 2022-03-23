from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('results.db'))

graphs.plot([
    {
        'session_id': 'label_100c_10l_600mn_600mx',
        'field': 'acc',
        'config': {'color': 'b', 'label': "label_100c_10l_600mn_600mx {'iidness': 0}"},
    },
    {
        'session_id': 'label_100c_5l_300mn_300mx',
        'field': 'acc',
        'config': {'color': 'r', 'label': "label_100c_3l_600mn_600mx {'iidness': 8484.848484848484}"},
    },
    {
        'session_id': 'label_100c_10l_10mn_600mx',
        'field': 'acc',
        'config': {'color': 'g', 'label': "label_100c_10l_10mn_600mx {'iidness': 270.7248484848485}"},
    },
    # {
    #     'session_id': 'label_100c_3l_10mn_600mx',
    #     'field': 'acc',
    #     'config': {'color': 'yellow', 'label': "label_100c_3l_10mn_600mx {'iidness': 3345.8772424242425}"},
    # },
    {
        'session_id': 'label_100c_1l_600mn_600mx',
        'field': 'acc',
        'config': {'color': 'cyan', 'label': "label_100c_1l_600mn_600mx {'iidness': 32727.272727272728}"},
    },
    {
        'session_id': 'label_100c_1l_10mn_600mx',
        'field': 'acc',
        'config': {'color': 'black', 'label': "label_100c_1l_10mn_600mx {'iidness': 10312.18997979798}"},
    },
], xlabel='Round', ylabel='Accuracy', plt_func=change)



