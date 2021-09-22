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
from src.federated.subscribers import Timer, ShowWeightDivergence

args = federated_args.FederatedArgs({
    'epoch': 1,
    'batch': 9999,
    'round': 50,
    'shard': 2,
    'dataset': 'mnist',
    'clients_ratio': 0.1,
    'learn_rate': 0.1,
    'tag': 'warmup',
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
client_data = preload(f'mnist_{args.shard}shards_100c_600min_600max', args.dataset,
                      lambda dg: dg.distribute_shards(100, args.shard, 600, 600))
warmup_client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.05)[0]).reduce(lambdas.dict2dc).as_tensor()
task_client_data = client_data.map(lambda ci, dc: dc.shuffle(42).split(0.05)[1]).map(lambdas.as_tensor)
logger.info('Generating Data --Ended')

initial_model = TorchModel(LogisticRegression(28 * 28, 10))
initial_model.train(warmup_client_data.batch(50), epochs=300)
initial_model = initial_model.extract()

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=args.batch, epochs=args.epoch,
                               optimizer='sgd',
                               criterion='cel', lr=args.learn_rate)

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
)
# federated.add_subscriber(subscribers.ShowDataDistribution(10, per_round=True, save_dir='./pct'))
federated.add_subscriber(subscribers.FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(subscribers.FedSave(args.tag))
federated.add_subscriber(ShowWeightDivergence(save_dir="./pct", plot_type='linear', divergence_tag='warmup_sgd2'))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
files.accuracies.save_accuracy(federated, args.tag)
