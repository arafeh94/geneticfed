import copy
import math
import statistics
from typing import Union

from apps.main_split.models import MnistNet
from libs.model.cv.cnn import Cnn1D
from libs.model.linear.lr import LogisticRegression
from libs.model.linear.lr_kdd import KDD_LR
from src.apis import lambdas, federated_tools
from src.apis.extensions import TorchModel
from src.apis.federated_tools import aggregate
from src.data.data_distributor import PipeDistributor, UniqueDistributor
from src.data.data_loader import preload
from src.federated.components.trainers import TorchTrainer


def fullf():
    dataset = preload('fekdd_train', UniqueDistributor())
    train_clients = dataset.map(lambda cid, dt: dt.split(0.8)[0])
    train_clients = train_clients.map(lambda cid, dt: dt.split(0.05)[0]).filter(lambdas.empty)
    test_clients = dataset.map(lambda cid, dt: dt.split(0.8)[1]).reduce(lambdas.dict2dc).as_tensor()
    # separate warmup dataset
    warmup_data = train_clients.map(lambda cid, dt: dt.split(.999)[0]).reduce(lambdas.dict2dc).as_tensor()
    train_clients = train_clients.map(lambda cid, dt: dt.split(.999)[1]).map(lambdas.as_tensor)

    # model = LogisticRegression(41, 23)
    # model = MnistNet(41, 32, 23)
    model = KDD_LR(41, 23)

    trainer = TorchModel(copy.deepcopy(model))
    trainer.train(warmup_data.batch(500), epochs=10, lr=0.01, momentum=0.9)
    print(trainer.infer(test_clients.batch(500)))


#     last_round_trainers = []
#     for k in range(10):
#         print('')
#         print('')
#         print('')
#         print('')
#         print(f'round {k}')
#         total = []
#         trainer_old: Union[TorchModel, None] = None
#         for i in range(8):
#             trainer.train(train_clients[i].batch(), lr=0.01, epochs=100)
#             if k != 9 and trainer_old is not None:
#                 trainer.dilute(trainer_old, 10)
#             trainer_old = trainer.copy()
#             if k == 9:
#                 last_round_trainers.append(trainer.copy())
#             for j in range(8):
#                 acc, loss = trainer.infer(test_clients[j].batch())
#                 print(f'{i} test on {j} {acc},{loss}')
#                 total.append(acc)
#             print("avg: ", statistics.mean(total))
#     weights = federated_tools.aggregate({
#         0: last_round_trainers[0].model.state_dict(),
#         1: last_round_trainers[1].model.state_dict(),
#         2: last_round_trainers[2].model.state_dict(),
#         3: last_round_trainers[3].model.state_dict(),
#         4: last_round_trainers[4].model.state_dict(),
#         5: last_round_trainers[5].model.state_dict(),
#         6: last_round_trainers[6].model.state_dict(),
#         7: last_round_trainers[7].model.state_dict(),
#     }, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})
#     trainer.load(weights)
#     total = []
#     for j in range(8):
#         acc, loss = trainer.infer(test_clients[j].batch())
#         print(f'final test on {j} {acc},{loss}')
#         total.append(acc)
#     print("avg: ", statistics.mean(total))
#     return statistics.mean(total)
#
#
fullf()
