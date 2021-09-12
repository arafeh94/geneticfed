import logging
import sys

from apps.paper_experiments import federated_args
from src.data.data_loader import preload

sys.path.append('../../')

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
from src.federated.subscribers import Timer

args = federated_args.FederatedArgs({
    'epoch': 10,
    'batch': 50,
    'round': 3,
    'shard': 2,
    'dataset': 'mnist',
    'clients_ratio': 0.1,
    'learn_rate': 0.1,
    'tag': 'basic',
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(f'mnist_{args.shard}shards_100c_600min_600max', args.dataset,
                      lambda dg: dg.distribute_shards(100, args.shard, 600, 600))
logger.info('Generating Data --Ended')

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
    num_rounds=3,
    desired_accuracy=0.99,
)
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(subscribers.FedSave(args.tag))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
files.accuracies.save_accuracy(federated, args.tag)
