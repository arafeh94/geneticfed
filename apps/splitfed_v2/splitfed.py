import copy
import json
import logging
import random
import sys
from collections import defaultdict

from apps.splitfed_v2._run_configs import global_configs
from apps.splitfed_v2.core import splitlearn
from apps.splitfed_v2.core.clusters import Cluster
from apps.splitfed_v2.core.server import Server
from apps.splitfed_v2.core.splitlearn import ClsIterator, cluster, get_clients
from src.apis import utils
from src.apis.extensions import Dict
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import models
from src.federated.subscribers.sqlite_logger import SQLiteLogger

utils.enable_logging(level=logging.ERROR)
random.seed(42)
configs = Dict(json.loads(sys.argv[1])) if len(sys.argv) > 1 else Dict({'name': 'splitfed'})
logger = SQLiteLogger.new_instance('splitlearn_v2.sqlite', configs)
printer = logging.getLogger('splitfed')
# configs
rounds = configs.get('rounds', global_configs['rounds'])
client_model = global_configs['client_model']
server_model = global_configs['server_model']
mnist = global_configs['train']
test_data = global_configs['test']
cluster_speeds = global_configs['cls_speeds']
out_clusters = global_configs['out_size']
lr_client = global_configs['lr_client']
lr_server = global_configs['lr_server']

double_clustered = get_clients(mnist, out_clusters, cluster_speeds, client_model, lr_client)
one_cluster = Cluster(copy.deepcopy(client_model), splitlearn.clients1d(double_clustered))

server = Server(copy.deepcopy(server_model), copy.deepcopy(client_model), test_data, lr=lr_server)
history = defaultdict(dict)
internal_counter = 0
for round_id in range(rounds):
    stats = splitlearn.one_round_resource([one_cluster], server, is_parallel=True, is_selection=False)
    stats['speed'] = 1
    stats['round_num'] = round_id
    stats['iter_num'] = round_id
    logger.log_all(internal_counter, stats)
    history[round_id][round_id] = stats
    internal_counter += 1
    printer.error("speed_idx - {}: {}".format(round_id, stats))
