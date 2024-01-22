import copy
import random
from collections import defaultdict
from typing import List, Dict

from apps.splitfed.core.client import Client
from src.apis.utils import UniqueSelector


class Speed:
    FAST = 1
    SLOW = 0

    def __init__(self, slow_rate=0.4):
        self.total_fast = 0
        self.total_slow = 0
        self.speeders = 0.55
        self.type = -1
        self.value = -1

    def get_uniform(self):
        if random.random() > self.speeders:
            return self.get_fast()
        return self.get_slow()

    def get_fast(self):
        self.type = 1
        self.value = random.uniform(0.03, 0.04)
        return self

    def get_slow(self):
        self.type = 0
        self.value = random.uniform(0.008, 0.01)
        return self


class Cluster:

    def __init__(self, initial_model, clients: List[Client]):
        self.model = initial_model
        self.clients = clients

    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())
        for client in self.clients:
            client.model.load_state_dict(new_model.state_dict())

    def merge(self, other: 'Cluster'):
        for c in other.clients:
            self.clients.append(c)
        self.update_model(self.model)

    def __repr__(self):
        if len(self.clients) > 0:
            return f'Cluster: {self.clients[0].__repr__()}'
        else:
            return "Empty Cluster"


def generate(clients_data, initial_model, client_cluster_size, cluster_limit=1) -> Dict[int, Cluster]:
    clusters = {}
    cluster_id = 0
    cluster_client = []
    for i in range(0, len(clients_data), client_cluster_size):
        cluster_data = clients_data.select(range(i, i + client_cluster_size))
        for index, dt in cluster_data.items():
            cluster_client.append(Client(dt.as_tensor(), initial_model, 1, cid=index))
            if cluster_limit != 0 and len(cluster_client) >= cluster_limit:
                clusters[cluster_id] = Cluster(initial_model, cluster_client)
                cluster_id += 1
                cluster_client = []
        if len(cluster_client) > 0:
            clusters[cluster_id] = Cluster(initial_model, cluster_client)
            cluster_id += 1
            cluster_client = []
    return clusters


def generate_speed(clients_data, initial_model, client_cluster_size, cluster_limit=1, slow_rate=0.4, cllr=0.0001):
    clusters_fast = {}
    clusters_slow = {}
    cluster_id_fast = 0
    cluster_id_slow = 0
    cluster_client_fast = []
    cluster_client_slow = []
    if not client_cluster_size:
        client_cluster_size = len(clients_data)
    for i in range(0, len(clients_data), client_cluster_size):
        cluster_data = clients_data.select(range(i, i + client_cluster_size))
        for index, dt in cluster_data.items():
            speed = Speed(slow_rate).get_uniform()
            if speed.type == speed.FAST:
                cluster_client_fast.append(Client(dt, initial_model, speed.value, cllr))
                if cluster_limit and len(cluster_client_fast) >= cluster_limit:
                    clusters_fast[cluster_id_fast] = Cluster(initial_model, cluster_client_fast)
                    cluster_id_fast += 1
                    cluster_client_fast = []
            else:
                cluster_client_slow.append(Client(dt, initial_model, speed.value, cllr))
                if cluster_limit and len(cluster_client_slow) >= cluster_limit:
                    clusters_slow[cluster_id_slow] = Cluster(initial_model, cluster_client_slow)
                    cluster_id_slow += 1
                    cluster_client_slow = []
        if len(cluster_client_fast) > 0:
            clusters_fast[cluster_id_fast] = Cluster(initial_model, cluster_client_fast)
            cluster_id_fast += 1
            cluster_client_fast = []
        if len(cluster_client_slow) > 0:
            clusters_slow[cluster_id_slow] = Cluster(initial_model, cluster_client_slow)
            cluster_id_slow += 1
            cluster_client_slow = []
    return clusters_fast, clusters_slow


def generate_resources(clusters, server_model, client_model, in_cluster_size, speeds=[1, 1, 1], lr=0.0001):
    client_out = {}
    for i in range(len(clusters)):
        client_in = {}
        client_out[i] = client_in
        selectables = UniqueSelector(clusters[i])
        for speed_index, speed in enumerate(speeds):
            ics = in_cluster_size(speed_index) if callable(in_cluster_size) else in_cluster_size
            similar_r_c = []
            for j in range(ics):
                similar_r_c.append(Client.generate(selectables.peek(), copy.deepcopy(client_model), speed, lr))
            client_in[speed] = Cluster(copy.deepcopy(server_model), similar_r_c)
    return client_out


def shuffle(clusters):
    as_list = list(clusters.items())
    random.shuffle(as_list)
    return dict(as_list)


def cluster_speed(clusters: Dict[int, Cluster]):
    pass
