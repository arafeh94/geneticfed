import logging
import sys
import sys
import time

from apps.frm_exp import distributors
from src.apis.rw import IODict
from src.federated.subscribers.fed_plots import EMDWeightDivergence
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.subscribers.trackers import BandwidthTracker

sys.path.append("../../")

from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.timer import Timer

from libs.model.cv.cnn import Cifar10Model, CNN_OriginalFedAvg
from src.data.data_distributor import ShardDistributor, LabelDistributor, DirichletDistributor, UniqueDistributor

from libs.model.cv.resnet import ResNet, resnet56
from src.data.data_loader import preload
from typing import Callable
from torch import nn
from src.apis import lambdas, files, federated_args, utils
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager, SharedTrainerProvider

args = federated_args.FederatedArgs({
    'epoch': 25,
    'batch': 50,
    'round': 200,
    'distributor': 2,
    'dataset': 'mnist',
    'clients_ratio': 0.2,
    'learn_rate': 0.01,
    'tag': 'mnist',
    'min': 600,
    'max': 600,
    'clients': 100,
})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(args.dataset, distributors.load(args)[args.distributor])
print(client_data)
logger.info('Generating Data --Ended')

initial_model = CNN_OriginalFedAvg(True)

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
    desired_accuracy=1,
)

federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(SQLiteLogger(utils.hash_string(str(args)), db_path='./res.db', config=args))
federated.add_subscriber(BandwidthTracker())

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
