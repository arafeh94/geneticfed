import copy
import math
import statistics
from typing import Union

from apps.main_split.models import MnistNet
from libs.model.linear.lr import LogisticRegression
from src.apis import lambdas, federated_tools
from src.apis.extensions import TorchModel
from src.apis.federated_tools import aggregate
from src.data.data_distributor import PipeDistributor
from src.data.data_loader import preload
from src.federated.components.trainers import TorchTrainer


def fullf():
    dataset = preload('mnist')
    dist = PipeDistributor([
        PipeDistributor.pick_by_label_id([0], 700),
        PipeDistributor.pick_by_label_id([1], 700),
        PipeDistributor.pick_by_label_id([2], 700),
        PipeDistributor.pick_by_label_id([3], 700),
        PipeDistributor.pick_by_label_id([4], 700),
        PipeDistributor.pick_by_label_id([5], 700),
        PipeDistributor.pick_by_label_id([6], 700),
        PipeDistributor.pick_by_label_id([7], 700),
    ])
    clients = dist.distribute(dataset)
    train_clients = clients.map(lambda cid, dt: dt.split(.8)[0]).map(lambdas.as_tensor)
    test_clients = clients.map(lambda cid, dt: dt.split(.8)[1]).map(lambdas.as_tensor)

    # model = LogisticRegression(28 * 28, 10)
    model = MnistNet(28 * 28, 32, 10)

    trainer = TorchModel(copy.deepcopy(model))
    last_round_trainers = []
    for k in range(10):
        total = []
        trainer_old: Union[TorchModel, None] = None
        for i in range(8):
            trainer.train(train_clients[i].batch(), lr=0.01, epochs=100)
            if k != 9 and trainer_old is not None:
                trainer.dilute(trainer_old, 10)
            trainer_old = trainer.copy()
            if k == 9:
                last_round_trainers.append(trainer.copy())
            for j in range(8):
                acc, loss = trainer.infer(test_clients[j].batch())
                print(f'{i} test on {j} {acc},{loss}')
                total.append(acc)
            print("avg: ", statistics.mean(total))
    weights = federated_tools.aggregate({
        0: last_round_trainers[0].model.state_dict(),
        1: last_round_trainers[1].model.state_dict(),
        2: last_round_trainers[2].model.state_dict(),
        3: last_round_trainers[3].model.state_dict(),
        4: last_round_trainers[4].model.state_dict(),
        5: last_round_trainers[5].model.state_dict(),
        6: last_round_trainers[6].model.state_dict(),
        7: last_round_trainers[7].model.state_dict(),
    }, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})
    trainer.load(weights)
    total = []
    for j in range(8):
        acc, loss = trainer.infer(test_clients[j].batch())
        print(f'final test on {j} {acc},{loss}')
        total.append(acc)
    print("avg: ", statistics.mean(total))
    return statistics.mean(total)


fft = []
for i in range(10):
    fft.append(fullf())
print('arafeh avg:', statistics.mean(fft))
