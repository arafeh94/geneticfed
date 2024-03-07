import copy
import json
import logging
import random
import sys
from collections import defaultdict

import numpy as np

from apps.splitfed_v2._run_configs import global_configs
from apps.splitfed_v2.core import splitlearn
from apps.splitfed_v2.core.server import Server
from apps.splitfed_v2.core.splitlearn import ClsIterator, cluster, get_clients
from src.apis import utils
from src.apis.extensions import Dict
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from apps.donotuse.split_learning import models
from src.federated.subscribers.sqlite_logger import SQLiteLogger

np.seterr(divide='ignore', invalid='ignore')

utils.enable_logging(level=logging.ERROR)
random.seed(42)
configs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {'name': '2layers_selection'}
logger = SQLiteLogger.new_instance('splitlearn_v2.sqlite', configs)
printer = logging.getLogger('2layers_selection')
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
server = Server(copy.deepcopy(server_model), copy.deepcopy(client_model), test_data, lr_server)
history = defaultdict(dict)
internal_counter = 0
for round_id in range(rounds):
    itero = ClsIterator(cluster_speeds)
    stats = None
    while True:
        cross = itero.next()
        if not cross[2]:
            break
        if len(cross) > 3 and cross[3]:
            continue
        speed_idx = itero.val()
        run_clusters = []
        for rs_key, outer in double_clustered.items():
            run_clusters.append(outer[speed_idx])
        stats = splitlearn.one_round_resource(run_clusters, server, is_parallel=True, is_selection=True)
        stats['speed'] = speed_idx
        stats['round_num'] = round_id
        stats['iter_num'] = itero.counter()
        logger.log_all(internal_counter, stats)
        history[round_id][itero.counter()] = stats
        internal_counter += 1
        printer.error("speed_idx - {}: {}".format(speed_idx, stats))
        itero.append_time(stats['round_time'])
    printer.error('round_end: {}'.format(round_id))
    printer.error('------------------------------------')
