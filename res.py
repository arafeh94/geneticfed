from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graph = Graphs(FedDB('res.db'))

graph.plot([
    {
        'session_id': 't1647711234',
        'field': 'acc',
        'config': {'color': 'b', 'label': '1'},
    },
    {
        'session_id': 't1647722525',
        'field': 'acc',
        'config': {'color': 'y', 'label': '2'},
    },
])
