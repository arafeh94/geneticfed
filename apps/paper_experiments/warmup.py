import logging
import sys

from apps.paper_experiments import federated_args
from libs.model.cv.cnn import Cifar10Model
from src import tools
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload

sys.path.append('../../')

from torch import nn
from src.apis import lambdas, files
from src.apis.extensions import TorchModel
from libs.model.linear.lr import LogisticRegression
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated import subscribers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager, SharedTrainerProvider
from src.federated.subscribers import Timer, ShowWeightDivergence, Resumable, FederatedLogger

args = federated_args.FederatedArgs({
    'epoch': 25,
    'batch': 50,
    'round': 1000,
    'shard': 2,
    'dataset': 'cifar10',
    'clients_ratio': 0.1,
    'learn_rate': 0.1,
    'tag': 'warmup',
    'min': 600,
    'max': 600,
    'clients': 100,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

logger.info('Generating Data --Started')
dist = LabelDistributor(args.clients, args.shard, args.min, args.max)
client_data = preload(args.dataset, dist)
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

federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
federated.add_subscriber(Resumable(federated, dist, 'warmup_02cr'))

# federated.add_subscriber(subscribers.ShowDataDistribution(10, per_round=True, save_dir='./pct'))
# federated.add_subscriber(subscribers.FedSave(str(args)))
# federated.add_subscriber(
#     ShowWeightDivergence(save_dir="./pct", plot_type='linear', divergence_tag=f'warmup_sgd{args.shard}'))

logger.info("----------------------")
logger.info("start federated 1")
logger.info("----------------------")
federated.start()
files.accuracies.save_accuracy(federated, str(args))
