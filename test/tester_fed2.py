import copy
import statistics

from apps.main_split.models import MnistNet
from libs.model.linear.lr import LogisticRegression
from src.apis import lambdas, federated_tools
from src.apis.extensions import TorchModel
from src.apis.federated_tools import aggregate
from src.data.data_distributor import PipeDistributor
from src.data.data_loader import preload
from src.federated.components.trainers import TorchTrainer

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

trainers = []
for i in range(8):
    trainers.append(TorchModel(copy.deepcopy(model)))

for k in range(10):
    print('')
    print('')
    print('')
    print('')
    print(f'round {k}')
    for i in range(8):
        trainers[i].train(train_clients[i].batch(), lr=0.01, epochs=100)
        for j in range(8):
            print(f'{i} test on {j}', trainers[i].infer(test_clients[j].batch()))

    weights = aggregate({
        0: trainers[0].model.state_dict(),
        1: trainers[1].model.state_dict(),
        2: trainers[2].model.state_dict(),
        3: trainers[3].model.state_dict(),
        4: trainers[4].model.state_dict(),
        5: trainers[5].model.state_dict(),
        6: trainers[6].model.state_dict(),
        7: trainers[7].model.state_dict(),
    },
        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})
    modl = copy.deepcopy(model)
    modl.load_state_dict(weights)
    total = []
    for j in range(8):
        acc, loss = federated_tools.infer(modl, test_clients[j].batch())
        print(f'test on {j}: {acc}')
        total.append(acc)
    print('avg:', statistics.mean(total))
# print(f'agg test on 0', federated_tools.infer(modl, test_clients[0].batch()))
# print(f'agg test on 1', federated_tools.infer(modl, test_clients[1].batch()))
