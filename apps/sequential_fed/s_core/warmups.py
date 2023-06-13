import copy
import logging
from typing import Union

from src.apis import federated_tools, lambdas
from src.apis.extensions import TorchModel
from src.data.data_container import DataContainer

logger = logging.getLogger('warmups')


def sequential_warmup(model, rounds, train_clients, test_dataset: DataContainer, epochs, lr):
    trainer = TorchModel(copy.deepcopy(model))
    last_round_trainers = []
    for k in range(rounds):
        logger.info(f'round {k}')
        trainer_old: Union[TorchModel, None] = None
        for i, vi in train_clients.items():
            trainer.train(train_clients[i].batch(), lr=lr, epochs=epochs, verbose=0)
            if k != rounds - 1 and trainer_old is not None:
                trainer.dilute(trainer_old, 10)
            trainer_old = trainer.copy()
            if k == rounds - 1:
                last_round_trainers.append(trainer.copy())
        logger.info(f"avg: {trainer.infer(test_dataset.batch(500))}")
    models_states = {}
    models_sizes = {}
    for i in range(len(last_round_trainers)):
        models_states[i] = last_round_trainers[i].model.state_dict()
        models_sizes[i] = 1
    weights = federated_tools.aggregate(models_states, models_sizes)
    trainer.load(weights)
    logger.info(f"final avg: {trainer.infer(test_dataset.batch(), verbose=0)}")
    return weights


def original_warmup(warmup_ratio, train_data, model, epochs=500):
    warmup_data = train_data.map(lambda ci, dc: dc.shuffle(42).split(warmup_ratio)[0]).reduce(
        lambdas.dict2dc).as_tensor()
    train_data = train_data.map(lambda cid, dt: (cid, dt.concat(warmup_data)))
    initial_model = TorchModel(copy.deepcopy(model))
    initial_model.train(warmup_data.batch(), epochs=epochs, verbose=1)
    return train_data, initial_model.extract().state_dict()
