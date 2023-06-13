from apps.main_split.models import MnistNet
from src.apis import lambdas, utils
from src.apis.extensions import TorchModel
from src.data.data_distributor import PipeDistributor
from src.data.data_loader import preload
from src.federated.components.trainers import TorchTrainer
utils.enable_logging()

dataset = preload('mnist')
dist = PipeDistributor([
    PipeDistributor.pick_by_label_id([0], 700),
    PipeDistributor.pick_by_label_id([1], 700),
])
clients = dist.distribute(dataset)
train_clients = clients.map(lambda cid, dt: dt.split(.8)[0]).map(lambdas.as_tensor)
test_clients = clients.map(lambda cid, dt: dt.split(.8)[1]).map(lambdas.as_tensor)

model = MnistNet(28 * 28, 32, 10)
trainer = TorchModel(model)

for i in range(2):
    print('')
    print('')
    print('')
    print(f'round {i}')
    print('training on 0')
    trainer.train(train_clients[0].batch(), lr=0.01, epochs=1000)
    print(f'test on {0}', trainer.infer(test_clients[0].batch()))
    print(f'test on {1}', trainer.infer(test_clients[1].batch()))
    trainer_old = trainer.copy()
    print('training on 1')
    trainer.train(train_clients[1].batch(), lr=0.01, epochs=1000)
    print(f'test on {0}', trainer.infer(test_clients[0].batch()))
    print(f'test on {1}', trainer.infer(test_clients[1].batch()))

    print('after asyncgragation')
    trainer.dilute(trainer_old, 10)
    print(f'test on {0}', trainer.infer(test_clients[0].batch()))
    print(f'test on {1}', trainer.infer(test_clients[1].batch()))
