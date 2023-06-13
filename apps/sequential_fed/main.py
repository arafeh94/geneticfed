import copy
from sys import argv

from apps.main_split.models import MnistNet
from apps.sequential_fed.s_core import warmups, pfed
from src.apis import lambdas
from src.apis.extensions import TorchModel
from src.app.prebuilt import FastFed
from src.data.data_distributor import UniqueDistributor
from src.data.data_loader import preload
from configs import configs

train, test = preload("mnist").as_tensor().split(0.8)
train_clients = UniqueDistributor().distribute(train)
base_model = MnistNet(28 * 28, 32, 10)

for config in configs:
    model = copy.deepcopy(base_model)
    method = config['method']
    if method == "seq":
        initial_weights = warmups.sequential_warmup(
            model, config['wp_rounds'], train_clients, test, config['wp_epochs'], config['lr'])
    elif method == "warmup":
        train_data, initial_weights = warmups.original_warmup(
            config['wp_ratio'], train_clients, model, config['wp_epochs'])
    else:
        initial_weights = model.state_dict()

    model.load_state_dict(initial_weights)
    federate = pfed.create_fl(train_clients, model, config)
    federate.start()
