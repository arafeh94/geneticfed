from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

db1 = FedDB('../cached_results.db')
graphs = Graphs(db1)
print(graphs)

# mnist
# graphs.plot([
#     {
#         'session_id': 't1648855787',
#         'field': 'acc',
#         'config': {'color': 'b', 'label': '6.93'},
#     },
#     {
#         'session_id': 't1648865262',
#         'field': 'acc',
#         'config': {'color': 'r', 'label': '6.98'},
#     },
#     {
#         'session_id': 't1648878571',
#         'field': 'acc',
#         'config': {'color': 'g', 'label': '5.87'},
#     },
#     {
#         'session_id': 't1648896767',
#         'field': 'acc',
#         'config': {'color': 'yellow', 'label': '2.86'},
#     },
#     {
#         'session_id': 't1648920298',
#         'field': 'acc',
#         'config': {'color': 'pink', 'label': '3.71'},
#     },
#     {
#         'session_id': 't1648946379',
#         'field': 'acc',
#         'config': {'color': 'gray', 'label': '1.71'},
#     },
# ])

# cifar
graphs.plot([
    {
        'session_id': 't1648856546',
        'field': 'acc',
        'config': {'color': 'b', 'label': '7.14'},
    },
    {
        'session_id': 't1648875801',
        'field': 'acc',
        'config': {'color': 'r', 'label': '7.12'},
    },
    {
        'session_id': 't1648898008',
        'field': 'acc',
        'config': {'color': 'g', 'label': '6.26'},
    },
    {
        'session_id': 't1648923075',
        'field': 'acc',
        'config': {'color': 'yellow', 'label': '2.80'},
    },
    {
        'session_id': 't1648949608',
        'field': 'acc',
        'config': {'color': 'pink', 'label': '3.62'},
    },
    {
        'session_id': 't1648980833',
        'field': 'acc',
        'config': {'color': 'gray', 'label': '1.73'},
    },
])
