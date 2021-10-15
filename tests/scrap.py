from numpy.random import random

from src.data import data_loader
from src.data.data_distributor import LabelDistributor, UniqueDistributor, ShardDistributor

clients_data = data_loader.preload('111_' + str(random()), 'mnist', ShardDistributor(300, 2))
print(clients_data)
