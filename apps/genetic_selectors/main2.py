# mpiexec -n 4 python main_mpi.py
import logging
import sys
from os.path import dirname

from torch import nn

from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from libs.model.cv.cnn import CNN
from src import manifest
from src.apis import files
from src.federated.subscribers import Timer

sys.path.append(dirname(__file__) + '../')

from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import TorchTrainer
from src.federated.protocols import TrainerParams
from apps.genetic_selectors.algo import initializer
from src.federated.components import metrics, client_selectors, aggregators
from src.federated import subscribers, fedruns
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import SeqTrainerManager
from src.data import data_generator, data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = data_loader.mnist_2shards_100c_600min_600max().select(range(30))
logger.info('Generating Data --Ended')

config = {
    'batch_size': 50,
    'epochs': 15,
    'clients_per_round': 0.2,
    'num_rounds': 1,
    'desired_accuracy': 0.99,
    'nb_clusters': 10,
    'model': lambda: LogisticRegression(28 * 28, 10),

    'ga_max_iter': 10,
    'ga_r_cross': 0.05,
    'ga_r_mut': 0.1,
    'ga_c_size': 30,
    'ga_p_size': 200,
    'ga_min_fitness': 0.45,
}

initial_model = initializer.ga_module_creator(
    client_data, config['model'], max_iter=config['ga_max_iter'],
    r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
    c_size=config['ga_c_size'], p_size=config['ga_p_size'],
    clusters=config['nb_clusters'],
    desired_fitness=config['ga_min_fitness']
)

# initial_model = config['model']

trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=config['epochs'],
                               batch_size=config['batch_size'], criterion='cel', lr=0.1)
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion='cel'),
    client_selector=client_selectors.Random(config['clients_per_round']),
    trainers_data_dict=client_data,
    initial_model=initial_model,
    num_rounds=config['num_rounds'],
    desired_accuracy=config['desired_accuracy'],
)

federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(subscribers.FedPlot())
federated.add_subscriber(subscribers.FedSave('genetic'))
logger.info("----------------------")
logger.info(f"start federated 1")
logger.info("----------------------")
federated.start()

all_acc = federated.context.history.reduce(
    lambda first, key, val: [val['acc']] if first is None else first.append(val['acc'])
)

files.append(all_acc, 'genetic', manifest.DEFAULT_ACC_PATH)
