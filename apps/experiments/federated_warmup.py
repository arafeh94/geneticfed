import logging
import sys

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
from src.federated.subscribers import Timer, ShowWeightDivergence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(f'mnist_10shards_100c_600min_600max', 'mnist', lambda dg: dg.distribute_shards(100, 10, 600, 600))
# client_data = preload('mnist_2shards_100c_600min_600max', 'mnist', lambda dg: dg.distribute_shards(100, 2, 600, 600))
# warmup_client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.05)[0]).reduce(lambdas.dict2dc).as_tensor()
# task_client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.05)[1]).map(lambdas.as_tensor)
logger.info('Generating Data --Ended')

# initial_model = TorchModel(LogisticRegression(28 * 28, 10))
# initial_model.train(warmup_client_data.batch(50), epochs=500)
# initial_model = initial_model.extract()
initial_model = LogisticRegression(28 * 28, 10)

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=1, optimizer='sgd',
                               criterion='cel', lr=0.1)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(0.1),
    trainers_data_dict=client_data,
    initial_model=lambda: initial_model,
    num_rounds=50,
    desired_accuracy=0.99,
)
# federated.add_subscriber(subscribers.ShowDataDistribution(10, per_round=True, save_dir='./pct'))
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
# federated.add_subscriber(subscribers.FedSave('basic'))
# federated.add_subscriber(ShowWeightDivergence(save_dir="./pct", plot_type='linear', divergence_tag='warmup'))
federated.add_subscriber(ShowWeightDivergence(save_dir="./pct", plot_type='linear'))
logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
