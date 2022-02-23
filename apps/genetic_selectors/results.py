from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graph = Graphs(FedDB('res3.db'))
# graph = Graphs(FedDB('res_inverse.db'))

graph.plot([
    {
        'session_id': 'genetic_cluster_test_ga',
        'field': 'acc',
        'config': {'color': 'b', 'label': 'Our Approach'},
    },
    {
        'session_id': 'genetic_cluster_test_clustered',
        'field': 'acc',
        'config': {'color': 'r', 'label': 'Clustering'},
    },
    # {
    #     'session_id': 'genetic_cluster_test_normal',
    #     'field': 'acc',
    #     'config': {'color': 'pink', 'label': 'Normal'},
    # },
])
