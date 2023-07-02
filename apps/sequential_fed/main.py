import copy
from sys import argv

import wandb
from apps.main_split.models import MnistNet
from apps.sequential_fed import toos
from apps.sequential_fed.s_core import warmups, pfed
from libs.model.linear.lr_kdd import KDD_LR
from src.apis import lambdas, utils
from src.data.data_loader import preload
from configs import configs
from configs import distributor
from src.federated.events import Events

utils.enable_logging()
# train, test = preload("fekdd_train").filter(lambda x, y: y not in [21, 22, 23]).as_tensor().split(0.8)
train, test = preload("mnist").as_tensor().split(0.8)
train_clients = distributor.distribute(train).map(lambdas.as_tensor)
# base_model = KDD_LR(41, 23)
base_model = MnistNet(28 * 28, 32, 10)

for config in configs:
    run_id = config['id'] if 'id' in config else None
    model = copy.deepcopy(base_model)
    method = config['method']
    acc = []
    if method == "seq":
        initial_weights, acc = warmups.sequential_warmup(
            model, config['wp_rounds'], train_clients, test, config['wp_epochs'], config['lr'])
    elif method == "warmup":
        train_data, initial_weights, acc = warmups.original_warmup(
            config['wp_ratio'], train_clients, model, config['wp_epochs'])
    else:
        initial_weights = model.state_dict()

    model.load_state_dict(initial_weights)
    config['pre_acc'] = acc
    federate = pfed.create_fl(train_clients, test, model, config, run_id)
    federate.start()
