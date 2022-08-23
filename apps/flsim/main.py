import logging

from torch import nn

from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from src.data import data_loader
from src.federated.components import metrics, aggregators, trainers
from libs.model.linear.lr import LogisticRegression
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.timer import Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

data_file = '../../datasets/pickles/mnist_10shards_100c_400mn_400mx.pkl'

logger.info('Generating Data --Started')
client_data = data_loader.mnist_10shards_100c_400min_400max()
logger.info('Generating Data --Ended')

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=20, optimizer='sgd',
                               criterion='cel', lr=0.1)

initial_model = LogisticRegression(28 * 28, 10)
booted_model, client_director, w_first = rl_module_creator(client_data, initial_model)
client_selector = RLSelector(10, client_director, w_first)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selector,
    trainers_data_dict=client_data,
    # initial_model=lambda: LogisticRegression(28 * 28, 10),
    initial_model=booted_model,
    num_rounds=1000,
    desired_accuracy=0.99
)

federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
