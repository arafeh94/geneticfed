# mpiexec -n 4 python main_mpi.py
import logging
import sys
from os.path import dirname

from torch import nn

from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from apps.paper_experiments import federated_args
from src import manifest
from src.apis import files
from src.data.data_loader import preload
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

args = federated_args.FederatedArgs({
    'epoch': 10,
    'batch': 50,
    'round': 3,
    'shard': 2,
    'dataset': 'mnist',
    'clients_ratio': 0.1,
    'learn_rate': 0.1,
    'tag': 'genetic',
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(f'mnist_{args.shard}shards_100c_600min_600max', args.dataset,
                      lambda dg: dg.distribute_shards(100, args.shard, 600, 600))
logger.info('Generating Data --Ended')

config = {
    'batch_size': args.batch,
    'epochs': args.epoch,
    'clients_per_round': args.clients_ratio,
    'num_rounds': args.round,
    'desired_accuracy': 0.99,
    'nb_clusters': 10,
    'model': lambda: LogisticRegression(28 * 28, 10),

    'ga_max_iter': 20,
    'ga_r_cross': 0.05,
    'ga_r_mut': 0.1,
    'ga_c_size': 50,
    'ga_p_size': 200,
    'ga_min_fitness': 0.45,

    'save_dir': 'pics',
}

initial_model = initializer.ga_module_creator(
    client_data, config['model'], max_iter=config['ga_max_iter'],
    r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
    c_size=config['ga_c_size'], p_size=config['ga_p_size'],
    clusters=config['nb_clusters'],
    desired_fitness=config['ga_min_fitness'], epoch=200, batch=50,
    saved_models=f'./saved_models_{args.shard}'
)

trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=config['epochs'],
                               batch_size=config['batch_size'], criterion='cel', lr=args.learn_rate)
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

# federated.add_subscriber(subscribers.WandbLogger(config))
# federated.add_subscriber(subscribers.ShowDataDistribution(per_round=True, label_count=62, save_dir=config['save_dir']))
# federated.add_subscriber(subscribers.ShowWeightDivergence(save_dir=config['save_dir'], plot_type='linear'))
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND, Timer.AGGREGATION, Timer.TRAINING]))
# federated.add_subscriber(subscribers.FedPlot())
federated.add_subscriber(subscribers.FedSave(args.tag))
logger.info("----------------------")
logger.info(f"start federated 1")
logger.info("----------------------")
federated.start()
files.accuracies.save_accuracy(federated, args.tag)
