import logging

import torch

from apps.sequential_fed.s_core import warmups, pfed
from apps.splitfed.models import MnistNet
from src.apis import lambdas, utils
from src.data.data_loader import preload
from configs import distributor, fed_config

logger = logging.getLogger('seqfed')
utils.enable_logging()

train, test = preload("mnist").split(0.8)
train_clients = distributor.distribute(train).map(lambdas.as_tensor)
test = test.as_tensor()
base_model = MnistNet(28 * 28, 32, 10)

model_id = 'seqop_mnist_all_32_20_0.01.cp'
base_model.load_state_dict(torch.load(f'./weights/{model_id}'))
federate = pfed.create_fl(train_clients, test, base_model, fed_config, model_id)
federate.start()
