import logging
import random
import sys

from apps.main_split.models import MnistNet
from apps.sequential_fed.s_core.modules import SequentialSelector
from src.apis import lambdas
from src.data.data_container import DataContainer

sys.path.append('../../../')
from libs.model.linear.lr import LogisticRegression
from src.federated.components.client_scanners import DefaultScanner
from src.federated.events import Events
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import ShardDistributor, LabelDistributor, UniqueDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = preload('mnist', UniqueDistributor())

# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=50, epochs=1, optimizer='sgd',
    criterion='cel', lr=0.1)

random_selector = client_selectors.Random(0.1).select
sequential_list = SequentialSelector.continuous(len(client_data.keys()), 100)
# fl parameters
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
    client_scanner=DefaultScanner(client_data),
    client_selector=SequentialSelector(sequential_list, random_selector),
    trainers_data_dict=client_data,
    initial_model=lambda: MnistNet(28 * 28, 32, 10),
    num_rounds=100,
    desired_accuracy=0.99
)

# (subscribers)
federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))

logger.info("------------------------")
logger.info("start federated learning")
logger.info("------------------------")
federated.start()
