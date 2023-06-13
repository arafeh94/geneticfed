import copy
import math
import statistics
from typing import Union

from apps.main_split.dist import kdd_clustered
from apps.main_split.models import MnistNet
from libs.model.linear.lr import LogisticRegression
from libs.model.linear.lr_kdd import KDD_LR
from src.apis import lambdas, federated_tools
from src.apis.extensions import TorchModel
from src.apis.federated_tools import aggregate
from src.data.data_distributor import PipeDistributor, UniqueDistributor
from src.data.data_loader import preload
from src.federated.components.trainers import TorchTrainer

# dataset = preload('fekdd_train')
# clients = UniqueDistributor(0, 100, 0.1).distribute(dataset)

dataset = preload('fekdd_train', UniqueDistributor())
train_clients = dataset.map(lambda cid, dt: dt.split(0.8)[0])
train_clients = train_clients.map(lambda cid, dt: dt.split(0.05)[0]).filter(lambdas.empty)
test_clients = dataset.map(lambda cid, dt: dt.split(0.8)[1])

# model = LogisticRegression(28 * 28, 10)
# model = MnistNet(28 * 28, 32, 10)
model = KDD_LR(41, 23)

trainer = TorchModel(copy.deepcopy(model))
last_round_trainers = []
rounds = 30
for k in range(rounds):
    print('=================')
    print(f'round {k}')
    print('=================')
    total = []
    trainer_old: Union[TorchModel, None] = None
    for i, vi in train_clients.items():
        trainer.train(train_clients[i].batch(), lr=0.01, epochs=100, verbose=0)
        if k != rounds - 1 and trainer_old is not None:
            trainer.dilute(trainer_old, 10)
        trainer_old = trainer.copy()
        if k == rounds - 1:
            last_round_trainers.append(trainer.copy())
        for j, vj in train_clients.items():
            acc, loss = trainer.infer(test_clients[j].batch(), verbose=0)
            print(f'{i} test on {j} {acc},{loss}')
            total.append(acc)
        print("avg: ", statistics.mean(total))

models_states = {}
models_sizes = {}
for i in range(len(last_round_trainers)):
    models_states[i] = last_round_trainers[i].model.state_dict()
    models_sizes[i] = 1
weights = federated_tools.aggregate(models_states, models_sizes)
trainer.load(weights)
total = []
for j, jv in test_clients.items():
    acc, loss = trainer.infer(test_clients[j].batch())
    print(f'final test on {j} {acc},{loss}')
    total.append(acc)
print("avg: ", statistics.mean(total))
