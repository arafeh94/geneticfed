import logging

from apps.main_split.models import MnistNet
from src.apis import utils, lambdas
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload
from src.federated.components import trainers, aggregators, metrics, client_selectors
from src.federated.components.client_scanners import DefaultScanner
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.fed_plots import RoundAccuracy, FinalAccuracyPlot
from src.federated.subscribers.logger import TqdmLogger, FederatedLogger
from src.federated.subscribers.timer import Timer

utils.enable_logging()
logger = logging.getLogger('main')
train, test = preload('mnist').split(.8)
distributor = ShardDistributor(300, 2)
clients = distributor.distribute(train).map(lambdas.as_tensor)
test = test.as_tensor()
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
    client_scanner=DefaultScanner(clients),
    # client_selector=client_selectors.All(),
    client_selector=client_selectors.Random(5),
    trainers_data_dict=clients,
    test_data=test,
    initial_model=lambda: MnistNet(28 * 28, 32, 10),
    num_rounds=2,
    desired_accuracy=0.99
)

# (subscribers)
federated.add_subscriber(TqdmLogger())
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(FinalAccuracyPlot())

logger.info("------------------------")
logger.info("start federated learning")
logger.info("------------------------")
federated.start()
