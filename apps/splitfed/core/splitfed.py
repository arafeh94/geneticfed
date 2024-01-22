import copy
import time
from collections import defaultdict
from typing import List

from apps.donotuse.main_split import funcs
from apps.splitfed.core import clusters
from apps.splitfed.core.server import Server
from src.apis.extensions import LinkedList, Node
from src.apis.federated_tools import aggregate, asyncgregate


class SplitFed:
    def __init__(self, server_model, client_model, client_clusters, test_data, epochs=10, lr=0.1, **configs):
        self.fed_server = Server(server_model, lr)
        self.client_model = copy.deepcopy(client_model)
        self.client_clusters = client_clusters
        self.acc = []
        self.round_exec_times = []
        self.test_data = test_data
        self.epochs = epochs
        self.round_nb = 0
        self.measure = lambda x: max(x) if 'is_cluster' in configs and configs['is_cluster'] else sum(x)
        self.fed_speed = configs['fed_speed'] if 'fed_speed' in configs else -1
        self.history = []

    def one_round(self):
        client_handlers = []
        cluster_exec_times = {}
        client_clusters = clusters.shuffle(self.client_clusters)
        for cluster_index, client_cluster in client_clusters.items():
            client_exec_time = {}
            client_speed = {}
            for client in client_cluster.clients:
                start_time = time.time()
                handler = Server(self.fed_server.model_copy())
                for e in range(self.epochs):
                    out, labels = client.local_train()
                    grad = handler.train(out, labels)
                    client.backward(grad)
                client_handlers.append(handler)
                spent_time = time.time() - start_time
                exec_time = 1 / client.speed * spent_time
                client_exec_time[client.id] = exec_time
                client_speed[client.id] = client.speed
            cluster_exec_times[cluster_index] = self.measure(client_exec_time.values())
            # cluster_exec_times[cluster_index] = max(client_exec_time.values())
            weights_clients = funcs.as_dict([c.model_copy().state_dict() for c in client_cluster.clients])
            weights_servers = funcs.as_dict([s.model_copy().state_dict() for s in client_handlers])
            self._aggregate(weights_clients, weights_servers)
            client_handlers.clear()
        self.round_exec_times.append(sum(cluster_exec_times.values()))
        self.acc.append(self.infer())
        self.round_nb += 1
        stats = {'acc': self.acc[-1], 'round_exec_time': self.round_exec_times[-1],
                 'cluster_exec_times': cluster_exec_times}
        self.history.append(stats)
        return stats

    def infer(self):
        return self.fed_server.infer(self.client_model, self.test_data)

    def _aggregate(self, weights_clients, weights_servers):
        avg_weights_clients = aggregate(weights_clients, {})
        avg_weights_server = aggregate(weights_servers, {})
        self.client_model.load_state_dict(avg_weights_clients)
        self.fed_server.model.load_state_dict(avg_weights_server)

    def crossgregate(self, weights_client, weights_server, staled_round):
        staleness = self.round_nb - staled_round
        s_weights = self.fed_server.model_copy().state_dict()
        c_weights = copy.deepcopy(self.client_model).state_dict()
        ac_s_weights = asyncgregate(s_weights, weights_server, staleness)
        ac_c_weights = asyncgregate(c_weights, weights_client, staleness)
        self.fed_server.model.load_state_dict(ac_s_weights)
        self.client_model.load_state_dict(ac_c_weights)

    def crossgregate2(self, split: 'SplitFed'):
        if self.round_nb < split.round_nb:
            raise Exception('crossgregate with future split not possible')
        self.crossgregate(copy.deepcopy(split.client_model).state_dict(),
                          split.fed_server.model_copy().state_dict(), self.round_nb - split.round_nb)
        # new version check if working correctly
        split.fed_server.model.load_state_dict(copy.deepcopy(self.fed_server.model).state_dict())
        split.client_model.load_state_dict(copy.deepcopy(self.client_model).state_dict())


class ClusterExecutor:
    def __init__(self, splitfeds: List[SplitFed]):
        self.splitfeds = sorted(splitfeds, key=lambda x: x.fed_speed)
        self.linked = LinkedList.create(self.splitfeds)
        self.current: Node = self.linked.head
        self.breadcrumbs = defaultdict(list)
        self.round_id = -1

    def one_round(self):
        self.round_id += 1
        while True:
            # print('next: {}'.format(self.current.value.fed_speed))
            self.current.value.one_round()
            self.breadcrumbs[self.round_id].append(self.current.value.fed_speed)
            if not self.next_fed():
                break

    def next_fed(self):
        if self.current.has_next():
            self.current = self.current.next_node()
        while self.is_time_exceeded():
            # print('before aggregate acc:', self.infer_all())
            self.current.value.crossgregate2(self.current.prev_node().value)
            # print("agg between {},{}".format(self.current.value.fed_speed, self.current.prev_node().value.fed_speed))
            # print('after aggregate acc:', self.infer_all())
            self.current = self.current.prev_node()
            if not self.current.has_prev():
                return False
        return True

    def infer(self):
        return max(self.infer_all().values())

    def infer_all(self):
        res = {}
        for sf in self.splitfeds:
            res[sf.fed_speed] = sf.infer()
        return res

    def is_time_exceeded(self):
        key = 'round_exec_time'
        current_history = self.current.value.history
        prev_history = self.current.prev_node().value.history
        total_current = sum([d[key] for d in current_history])
        total_prev = sum([d[key] for d in prev_history])
        return total_current > total_prev
