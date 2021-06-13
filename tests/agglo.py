import copy
import json
import logging
import random
from collections import namedtuple
from random import randint

import numpy as np
import torch
from sklearn import decomposition
from sklearn.cluster import KMeans
from torch import optim, nn
from sklearn.cluster import AgglomerativeClustering

from libs.model.dqn import DeepQNetwork
from libs.model.linear.lr import LogisticRegression
from src import tools
from src.data.data_generator import DataGenerator
from src.data.data_provider import PickleDataProvider


def compress(weights):
    weights = weights.reshape(10, -1)
    pca = decomposition.PCA(n_components=4)
    pca.fit(weights)
    weights = pca.transform(weights)
    return weights.flatten()


clients_data = DataGenerator(PickleDataProvider('../../datasets/pickles/mnist10k.pkl')) \
    .distribute_shards(30, 1, 100, 100)
tools.detail(clients_data, display=lambda x: print(x))
client_weights = {}
client_weights_c = {}
model = LogisticRegression(28 * 28, 10)
for client_id, data in clients_data.items():
    mm = copy.deepcopy(model)
    tools.train(mm, data.batch(99999), 1)
    weights = mm.state_dict()
    weights = tools.flatten_weights(weights)
    client_weights[client_id] = weights
    weights = compress(weights)
    client_weights_c[client_id] = weights

cluster = AgglomerativeClustering(n_clusters=10, affinity='cosine', linkage='complete')
predicted = cluster.fit_predict(list(client_weights.values()))
print(predicted.reshape(-1, 10))
