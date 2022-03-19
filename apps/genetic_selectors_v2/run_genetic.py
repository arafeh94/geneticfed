import logging
import sys
from datetime import datetime
from os.path import dirname
from pathlib import Path

from libs.model.cv.cnn import Cnn1D
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.data.data_loader import preload
from src.data.data_tools import iidness

sys.path.append(dirname(__file__) + '../')
from torch import nn
from libs.model.linear.lr import LogisticRegression
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.components.trainers import CPUTrainer, TorchTrainer
from src.federated.protocols import TrainerParams
from src.federated.components import metrics, aggregators
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.subscribers.analysis import ClientSelectionCounter, ShowDataDistribution

from apps.genetic_selectors_v2 import distributor
from apps.genetic_selectors_v2.algo import initializer
from apps.genetic_selectors_v2.algo.selector import GeneticSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
db_name = './results.db'
experiment_name = f'test_2'
logger.info('Generating Data --Started')
client_data = preload('fall_ar_by_client').map(lambdas.as_list).map(lambda cid, dt: dt.filter(lambda x, y: y != 0)) \
    .map(lambda cid, dt: dt.map(lambda x, y: (x, y - 1)))
print(iidness(client_data, 13, by_label=True))
ShowDataDistribution.plot(client_data.map(lambdas.as_list), 13)
exit(1)
logger.info('Generating Data --Ended')

configs = {
    # federated learning configs
    'batch_size': 80_000,
    'epochs': 5,
    'clients_per_round': 4,
    'num_rounds': 10,
    'desired_accuracy': 0.99,
    'model': lambda: Cnn1D(15),
    # genetic_configs
    'max_iter': 10,
    'r_cross': 0.05,
    'r_mut': 0.1,
    'c_size': 8,
    'p_size': 200,
    'nb_clusters': 1,
    'desired_fitness': 0,
}

initiator, initial_model = initializer.ga_module_creator(
    client_data, configs['model'], configs, epoch=50,
    batch=configs['batch_size'], lr=0.01)
client_selector = GeneticSelector(initiator, configs['clients_per_round'], configs)

trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=configs['epochs'],
                               batch_size=configs['batch_size'], criterion='cel', lr=0.01)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=configs['batch_size'], criterion=nn.CrossEntropyLoss()),
    client_selector=client_selector,
    trainers_data_dict=client_data,
    initial_model=initial_model,
    num_rounds=configs['num_rounds'],
    desired_accuracy=configs['desired_accuracy'],
)

FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]).attach(federated)
SQLiteLogger(f'{experiment_name}', db_name).attach(federated)
ClientSelectionCounter(save_dir='plots/').attach(federated)
client_selector.attach(federated)
logger.info("----------------------")
logger.info(f"start federated genetics")
logger.info("----------------------")
federated.start()
