import copy

from src.federated.components import trainers, aggregators, metrics, client_selectors
from src.federated.components.client_scanners import DefaultScanner
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers.logger import TqdmLogger, FederatedLogger
from src.federated.subscribers.timer import Timer
from src.federated.subscribers.wandb_logger import WandbLogger


def create_fl(client_data, test, model, config, id=None):
    # trainers configuration
    trainer_params = TrainerParams(
        trainer_class=trainers.TorchTrainer,
        batch_size=500, epochs=config['fe_epochs'], optimizer='sgd',
        criterion='cel', lr=0.1)

    # fl parameters
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
        client_scanner=DefaultScanner(client_data),
        client_selector=client_selectors.Random(config['cr']),
        trainers_data_dict=client_data,
        test_data=test,
        initial_model=lambda: copy.deepcopy(model),
        num_rounds=config['fe_rounds'],
        desired_accuracy=0.99
    )

    # (subscribers)
    federated.add_subscriber(TqdmLogger())
    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    if 'wandb' not in config or ('wandb' in config and config['wandb']):
        federated.add_subscriber(WandbLogger(config=config, id=id))
    return federated
