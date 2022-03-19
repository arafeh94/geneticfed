import calendar
import logging
import time

from libs.model.cv.cnn import Cnn1D
from libs.model.linear.lr import LogisticRegression
from src.apis import lambdas
from src.data.data_loader import preload
from src.federated.components import aggregators, metrics, client_selectors
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.components.trainers import TorchTrainer, CPUTrainer
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

client_data = preload('fall_ar_by_client').map(lambdas.as_tensor)
logger.info(client_data)

config = {
    'epochs': 1,
    'batch_size': 40000,
    'clients_per_round': 5,
    'initial_model': lambda: Cnn1D(15),
    'num_rounds': 500,
}

print(client_data)
trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=TorchTrainer, optimizer='sgd', epochs=config['epochs'],
                               batch_size=config['batch_size'],
                               criterion='cel', lr=0.01)
federated = FederatedLearning(
    trainer_manager=trainer_manager,
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=config['batch_size'], criterion='cel'),
    client_selector=client_selectors.Random(config['clients_per_round']),
    trainers_data_dict=client_data,
    initial_model=config['initial_model'],
    num_rounds=config['num_rounds'],
)
federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(SQLiteLogger(str(calendar.timegm(time.gmtime())), 'res.db', config))
logger.info("----------------------")
logger.info(f"start federated")
logger.info("----------------------")
federated.start()
