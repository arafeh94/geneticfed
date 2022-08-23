import logging
import sys

from apps.dbs import distributors
from src.federated.subscribers.fed_plots import EMDWeightDivergence
from src.federated.subscribers.sqlite_logger import SQLiteLogger

sys.path.append('../../')

from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.timer import Timer
from libs.model.cv.cnn import Cifar10Model
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from torch import nn
from src.apis import lambdas, files, federated_args, utils
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

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
logger.info('Generating Data --Ended')

if args.dataset == 'mnist':
    initial_model = TorchModel(LogisticRegression(28 * 28, 10))
elif args.dataset == 'cifar10':
    client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
    initial_model = TorchModel(Cifar10Model())
else:
    initial_model = TorchModel(LogisticRegression(28 * 28, 10))

warmup_client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.2)[0]).reduce(lambdas.dict2dc).as_tensor()
task_client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.2)[1]).map(lambdas.as_tensor)

initial_model.train(warmup_client_data.batch(50), epochs=300)
initial_model = initial_model.extract()

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=args.batch, epochs=args.epoch,
                               optimizer='sgd', criterion='cel', lr=args.learn_rate)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(args.clients_ratio),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=args.round,
    desired_accuracy=0.99,
    # accepted_accuracy_margin=0.02
)


federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(EMDWeightDivergence(show_plot=False))
federated.add_subscriber(SQLiteLogger(utils.hash_string(str(args)), db_path='res.db', config=args))


logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
