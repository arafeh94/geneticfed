import logging
import sys
from os.path import dirname

sys.path.append(dirname(__file__) + '../')
from torch import nn
from libs.model.linear.lr import LogisticRegression
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.components.trainers import CPUTrainer
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

db_name = './dbs/res1.db'
logger.info('Generating Data --Started')
client_data = distributor.get_distributed_data()
logger.info('Generating Data --Ended')

configs = {
    # federated learning configs
    'batch_size': 50,
    'epochs': 1,
    'clients_per_round': 10,
    'num_rounds': 1,
    'desired_accuracy': 0.99,
    'model': lambda: LogisticRegression(28 * 28, 10),
    # genetic_configs
    'max_iter': 10,
    'r_cross': 0.05,
    'r_mut': 0.1,
    'c_size': 10,
    'p_size': 200,
    'nb_clusters': 5,
    'desired_fitness': 0.45,
}

initiator, initial_model = initializer.ga_module_creator(client_data, configs['model'], configs, epoch=50)
client_selector = GeneticSelector(initiator, configs['clients_per_round'], configs)

trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=CPUTrainer, optimizer='sgd', epochs=configs['epochs'],
                               batch_size=configs['batch_size'], criterion='cel', lr=0.1)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=configs['batch_size'], criterion=nn.CrossEntropyLoss(), device='cpu'),
    client_selector=client_selector,
    trainers_data_dict=client_data,
    initial_model=initial_model,
    num_rounds=configs['num_rounds'],
    desired_accuracy=configs['desired_accuracy'],
)

FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]).attach(federated)
SQLiteLogger(f'genetic_cluster_test_genetics', db_name).attach(federated)
ClientSelectionCounter(save_dir='plots/').attach(federated)
ShowDataDistribution(10, save_dir='plots/').attach(federated)
client_selector.attach(federated)
logger.info("----------------------")
logger.info(f"start federated genetics")
logger.info("----------------------")
federated.start()
