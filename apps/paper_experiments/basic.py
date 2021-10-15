import logging
import sys

from libs.model.cv.cnn import Cifar10Model
from src.data.data_distributor import ShardDistributor, LabelDistributor

sys.path.append('../../')
from libs.model.cv.resnet import ResNet, resnet56
from apps.paper_experiments import federated_args
from src.data.data_loader import preload
from typing import Callable
from torch import nn
from src.apis import lambdas, files
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager, SharedTrainerProvider
from src.federated.subscribers import Timer, ShowWeightDivergence

args = federated_args.FederatedArgs({
    'epoch': 25,
    'batch': 50,
    'round': 200,
    'shard': 2,
    'dataset': 'mnist',
    'clients_ratio': 0.1,
    'learn_rate': 0.1,
    'tag': 'basic',
    'min': 600,
    'max': 600,
    'clients': 100,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(f'{args.dataset}_{args.shard}shards_{args.clients}c_{args.min}min_{args.max}max', args.dataset,
                      LabelDistributor(args.clients, args.shard, args.min, args.max))
print(client_data)
logger.info('Generating Data --Ended')

if args.dataset == 'mnist':
    initial_model = LogisticRegression(28 * 28, 10)
elif args.dataset == 'cifar10':
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    initial_model = Cifar10Model()
else:
    initial_model = LogisticRegression(28 * 28, 10)

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=args.batch, epochs=args.epoch,
                               optimizer='sgd', criterion='cel', lr=args.learn_rate)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=args.batch, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(args.clients_ratio),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=args.round,
    desired_accuracy=0.99,
    # accepted_accuracy_margin=0.02
)
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
# federated.add_subscriber(subscribers.FedPlot(show_loss=False, plot_each_round=True))

# federated.add_subscriber(subscribers.FedSave(args.tag))
# federated.add_subscriber(ShowWeightDivergence(save_dir="./pct", plot_type='linear', divergence_tag=f'basic_sgd{args.shard}'))
federated.add_subscriber(subscribers.ShowAvgWeightDivergence(plot_each_round=False, save_dir="./pct",
                                                             divergence_tag='div_' + str(args)))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
files.accuracies.save_accuracy(federated, str(args))
