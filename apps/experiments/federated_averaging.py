import logging
import random
import sys

from src.apis import lambdas
from src.data.data_container import DataContainer

sys.path.append('../../')
from libs.model.linear.lr import LogisticRegression
from src.federated.components.client_scanners import DefaultScanner
from src.federated.events import Events
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import ShardDistributor, LabelDistributor
from src.data.data_loader import preload
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

mnist_test: DataContainer = None

mnist_train, mnist_test = preload('mnist').split(0.8)
client_data = LabelDistributor(80, 5, 60, 300).distribute(mnist_train)


def poison(dc: DataContainer, rate):
    total_size = len(dc)
    poison_size = total_size * rate
    labels = dc.labels()
    while poison_size > 0:
        dc.y[random.randint(0, total_size - 1)] = random.choice(labels)
        poison_size -= 1


poison(mnist_test, 0.8)
# trainers configuration
trainer_params = TrainerParams(
    trainer_class=trainers.TorchTrainer,
    batch_size=50, epochs=10, optimizer='sgd',
    criterion='cel', lr=0.1)

# fl parameters
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
    client_scanner=DefaultScanner(client_data),
    client_selector=client_selectors.Random(0.1),
    trainers_data_dict=client_data,
    test_data=mnist_test,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=50,
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
