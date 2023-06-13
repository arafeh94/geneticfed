import copy
import random
import time

import torch
from torch.optim import Adam

from src.data.data_container import DataContainer
from src.federated.components.trainers import TorchTrainer

t_optimizer = None


class Trainer(TorchTrainer):
    def __init__(self, initial_model, speed=1, lr=0.0001):
        super().__init__()
        self.model = initial_model
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.device = torch.device('cpu')
        self.model_output = None
        self.is_trained = False
        self.speed = speed
        self.id = random.randint(0, 999999999)

    def split_train(self, data: DataContainer):
        inputs, labels = (data.x, data.y) if isinstance(data, DataContainer) else (data[0], data[1])
        inputs = inputs.to(self.device)
        labels = labels.clone().detach().long().to(self.device)
        self.model.to(self.device)
        self.optimizer.zero_grad()
        self.model_output = self.model(inputs)
        out = self.model_output.clone().detach().requires_grad_(True)
        self.is_trained = True
        return out, labels

    def model_copy(self):
        return copy.deepcopy(self.model)

    def backward(self, grad):
        self.model_output.backward(grad)
        self.optimizer.step()
        self.is_trained = False


class Client(Trainer):
    def __init__(self, data, initial_model, speed=1, lr=0.0001, cid=None):
        super().__init__(initial_model, speed, lr)
        self.data = data
        self.cid = cid

    def local_train(self):
        return super(Client, self).split_train(self.data)
