import copy
import logging
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data import data_loader

from src.data.data_container import DataContainer
from src.federated import subscribers, fedruns
from src.federated.components import params, client_selectors
from src.federated.components.aggregators import AVGAggregator
from src.federated.components.metrics import AccLoss
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import TorchTrainer, TorchChunkTrainer
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = data_loader.mnist_2shards_100c_600min_600max()
logger.info('Generating Data --Ended')

config = {
    'batch_size': 50,
    'epochs': 15,
    'clients_per_round': 0.2,
    'num_rounds': 50,
    'desired_accuracy': 0.99,
    'nb_clusters': 10,
    'model': lambda: LogisticRegression(28 * 28, 10),

    'clustering_num_rounds': 10,
    'linkage': 'complete',
    'clustering_measuring_algo': 'cosine',
    'update_data_ration': 0.1,
    'clustering_data_ratio': 0.1,
}


def partition(per):
    part_data = {}
    for client_id, data in client_data.items():
        part, rest = data.split(per)
        part_data[client_id] = part
        client_data[client_id] = rest
    return part_data


cluster_data = partition(config['clustering_data_ratio'])
update_data = partition(config['update_data_ration'])

federated_learning = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=TrainerParams(trainer_class=TorchChunkTrainer, batch_size=config['batch_size'], epochs=3,
                                 criterion='cel', optimizer='sgd', lr=0.1),
    num_rounds=config['clustering_num_rounds'],
    client_selector=client_selectors.All(),
    desired_accuracy=0.99,
    train_ratio=0.8,
    metrics=AccLoss(config['batch_size'], 'cel'),
    aggregator=AVGAggregator(),
    initial_model=config['model'],
    trainers_data_dict=cluster_data
)

federated_learning.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))

global_model = federated_learning.start()
w = global_model.state_dict()
clients_diff_weights = {}
for client_id, data in update_data.items():
    model_copy = copy.deepcopy(global_model)
    tools.train(model_copy, data.batch(config['batch_size']), epochs=3, lr=0.1)
    wc = model_copy.state_dict()
    delta = tools.flatten_weights(w) - tools.flatten_weights(wc)
    clients_diff_weights[client_id] = delta

cluster = AgglomerativeClustering(n_clusters=config['nb_clusters'], affinity=config['clustering_measuring_algo'],
                                  linkage=config['linkage'])
cluster_dist = cluster.fit_predict(list(clients_diff_weights.values()))
client_ids = list(clients_diff_weights.keys())
clustered_clients = defaultdict(list)
for cluster_id, client_id in zip(cluster_dist, client_ids):
    clustered_clients[cluster_id].append(client_id)
logger.info(cluster_dist)

clustered_federated = {}
for cluster, client_ids in clustered_clients.items():
    logger.info(f'cluster:{cluster}-clients:{client_ids}')
    if len(client_ids) < 3:
        logger.info("can't run federated learning with less than 3 clients")
        continue
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=TrainerParams(trainer_class=TorchChunkTrainer, batch_size=50, epochs=10, criterion='cel',
                                     optimizer='sgd', lr=0.1),
        num_rounds=config['num_rounds'],
        client_selector=client_selectors.Random(0.55),
        desired_accuracy=0.99,
        train_ratio=0.8,
        metrics=AccLoss(config['batch_size'], 'cel'),
        aggregator=AVGAggregator(),
        initial_model=lambda: global_model,
        trainers_data_dict=tools.dict_select(client_ids, client_data)
    )

    federated.add_subscriber(subscribers.FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_FED_END]))
    federated.init()
    clustered_federated[cluster] = federated

finished_tasks = [False] * len(clustered_federated)
while not all(finished_tasks):
    for index, (cluster_id, federated) in enumerate(clustered_federated.items()):
        logger.info(f'{cluster_id}')
        finished_tasks[index] = federated.one_round()

runs = fedruns.FedRuns(clustered_federated)
runs.plot_avg()
