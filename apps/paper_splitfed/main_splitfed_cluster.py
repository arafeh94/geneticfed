import os

from apps.paper_splitfed.core.splitfed import SplitFed
from apps.paper_splitfed.core import clusters
from src.apis import lambdas
from src.data.data_loader import preload
from apps.main_split import models, dist
from src.federated.subscribers.sqlite_logger import SQLiteLogger

sr = 0.4
# the max number of client in each cluster, use to avoid accuracy problem and cat. for.
cluster_limit = 0
# initialize clients, it should be replaced with weight divergence analysis.
# In this case, we have 5 clusters in each we have x clients
client_cluster_size = 20
rounds = 50
# init tools
logger = SQLiteLogger(os.path.splitext(os.path.basename(__file__))[0], 'logs.db')
# init data
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)
clients_data = preload('mnist', dist.clustered(client_cluster_size, 300), tag=f'cluster{client_cluster_size}p{300}')
train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

# split learning
fast, slow = clusters.generate_speed(train_data, client_model, client_cluster_size, 0, cllr=0.001)
for clf in fast:
    fast[clf].merge(slow[clf])
split_learning = SplitFed(server_model, client_model, fast, test_data, 1, lr=0.001, is_cluster=True)
r = 0
while rounds > 0:
    rounds -= 1
    stats = split_learning.one_round()
    print('acc:', stats['acc'], 'in:', stats['round_exec_time'])
    logger.log(r, acc=stats['acc'], exec_time=stats['round_exec_time'])
    r += 1
