# mpiexec -n 4 python main_mpi.py
import logging
import sys
sys.path.append("../../")

from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.timer import Timer


from torch import nn

from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from libs.model.cv.cnn import Cifar10Model
from src import manifest, tools
from src.apis import files, lambdas, federated_args
from src.apis.extensions import TorchModel
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import TorchTrainer
from src.federated.protocols import TrainerParams
from apps.genetic_selectors.algo import initializer
from src.federated.components import metrics, client_selectors, aggregators
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import SeqTrainerManager

args = federated_args.FederatedArgs({
    'epoch': 25,
    'batch': 50,
    'round': 300,
    'shard': 2,
    'dataset': 'cifar10',
    'clients_ratio': 0.1,
    'learn_rate': 0.001,
    'tag': 'genetic',
    'min': 600,
    'max': 600,
    'clients': 100,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
dist = LabelDistributor(args.clients, args.shard, args.min, args.max)
client_data = preload(args.dataset, dist)
print(client_data)
logger.info('Generating Data --Ended')

if args.dataset == 'mnist':
    c_model = LogisticRegression(28 * 28, 10)
elif args.dataset == 'cifar10':
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    c_model = Cifar10Model()
else:
    c_model = LogisticRegression(28 * 28, 10)

config = {
    'batch_size': args.batch,
    'epochs': 100,
    'clients_per_round': args.clients_ratio,
    'num_rounds': args.round,
    'desired_accuracy': 0.99,
    'nb_clusters': 20,
    'model': lambda: c_model,

    'ga_max_iter': 20,
    'ga_r_cross': 0.05,
    'ga_r_mut': 0.1,
    'ga_c_size': 30,
    'ga_p_size': 300,
    'ga_min_fitness': 0,
    'ga_epochs': 500,

    'save_dir': 'pics',
}

initial_model = initializer.ga_module_creator(
    client_data, config['model'], max_iter=config['ga_max_iter'],
    r_cross=config['ga_r_cross'], r_mut=config['ga_r_mut'],
    c_size=config['ga_c_size'], p_size=config['ga_p_size'],
    clusters=config['nb_clusters'],
    desired_fitness=config['ga_min_fitness'], epoch=config['ga_epochs'], batch=50,
    saved_models=f'./saved_models_{args.shard}_{config["ga_epochs"]}_{args.dataset}_{args.learn_rate}',
    lr=args.learn_rate
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
    accepted_accuracy_margin=0.01
)

federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(Resumable(str(args)))

# federated.add_subscriber(subscribers.WandbLogger(config))
# federated.add_subscriber(subscribers.ShowDataDistribution(per_round=True, label_count=62, save_dir=config['save_dir']))
# federated.add_subscriber(subscribers.ShowWeightDivergence(save_dir=config['save_dir'], plot_type='linear'))
# federated.add_subscriber(subscribers.FedPlot())
# federated.add_subscriber(subscribers.FedSave(args.tag))
# federated.add_subscriber(ShowWeightDivergence(save_dir="./pct", plot_type='linear', divergence_tag=f'genetic_sgd{args.shard}'))
# federated.add_subscriber(subscribers.ShowAvgWeightDivergence(plot_each_round=False, save_dir="./pct",
#                                                              divergence_tag='wd_' + str(args)))

logger.info("----------------------")
logger.info(f"start federated 1")
logger.info("----------------------")
federated.start()
files.accuracies.save_accuracy(federated, str(args))
