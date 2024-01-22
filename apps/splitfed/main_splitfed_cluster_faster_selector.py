import copy
import os
from collections import defaultdict
from typing import Dict

from apps.splitfed.core.clusters import generate_resources
from apps.splitfed.core.splitfed import SplitFed, ClusterExecutor
from apps.splitfed.models import MnistClient, MnistServer
from src.apis import lambdas
from src.data.data_distributor import ClusterDistributor, ShardDistributor
from src.data.data_loader import preload
from src.federated.subscribers.sqlite_logger import SQLiteLogger

logger = SQLiteLogger(os.path.splitext(os.path.basename(__file__))[0], 'logs.db')

# the max number of client in each cluster, use to avoid accuracy problem and cat. for.
cluster_limit = 4
# initialize clients, it should be replaced with weight divergence analysis.
# In this case, we have 5 clusters in each we have x clients
client_cluster_size = 20

# sr=slow_rate: probability of a created clients to be considered as slow with low capabilities
# increase virtual training time
rounds = 50

# init data
client_model = MnistClient(784, 32, 10)
server_model = MnistServer(784, 32, 10)

train_data, test_data = preload('mnist').split(0.9)
test_data = test_data.as_tensor()
inner_distributor = ShardDistributor(10000, 2)
clients_clusters = preload('mnist', ClusterDistributor(inner_distributor, 5, 500, 500), tag='d123')
clusters = generate_resources(clients_clusters, copy.deepcopy(server_model), copy.deepcopy(client_model), 10,
                              [2, 1, 0.5])

resources_based = defaultdict(dict)
split_feds: dict[int, SplitFed] = {}
for key_out, out in clusters.items():
    for key_in, inc in out.items():
        resources_based[key_in][key_out] = inc
for key_rb, rb in resources_based.items():
    split_fed = SplitFed(copy.deepcopy(server_model), copy.deepcopy(client_model), rb, test_data, fed_speed=key_rb)
    split_feds[split_fed.fed_speed] = split_fed

# sf = split_feds[1]
# for i in range(100):
#     sf.one_round()
#     print(sf.infer())

executor = ClusterExecutor([sf for sf in split_feds.values()])

for i in range(50):
    executor.one_round()
    print(executor.infer())
