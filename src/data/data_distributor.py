import logging
import random
from abc import abstractmethod
from collections import defaultdict
import numpy as np
import typing
from libs.data_distribute import distribute
from src.apis import lambdas, utils
from src.apis.extensions import Dict, CycleList
from src.data.data_container import DataContainer


class Distributor:
    def __init__(self):
        self.logger = logging.getLogger('datasets')

    def log(self, msg, level=logging.INFO):
        self.logger.log(level, msg)

    @abstractmethod
    def id(self):
        pass

    @abstractmethod
    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        pass

    def __repr__(self):
        return self.id()


# noinspection PyTypeChecker
class ClusterDistributor(Distributor):

    def __init__(self, cluster_distributor, cluster_client_num: int, min_size, max_size):
        super().__init__()
        self.cluster_client_num = cluster_client_num
        self.cluster_distributor = cluster_distributor
        self.min_size = min_size
        self.max_size = max_size

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_numpy()
        clusters = {}
        clusters_data = self.cluster_distributor.distribute(data)
        for cluster_index in range(len(clusters_data)):
            cluster_data: DataContainer = clusters_data[cluster_index].as_numpy()
            label_distributor = LabelDistributor(self.cluster_client_num, len(cluster_data.labels()),
                                                 self.min_size, self.max_size)
            clusters[cluster_index] = label_distributor.distribute(cluster_data)
        return Dict(clusters)

    def id(self):
        return "clsdist_" + str(self.cluster_distributor) + "_" + str(self.min_size) + "_" + str(
            self.max_size) + "_" + str(self.cluster_client_num)


class DirichletDistributor(Distributor):

    def __init__(self, num_clients, num_labels, skewness=0.5):
        super().__init__()
        self.num_clients = num_clients
        self.num_labels = num_labels
        self.skewness = skewness

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_numpy()
        client_rows = distribute(data.y, self.num_clients, self.num_labels, self.skewness)
        clients_data = {}
        for client in client_rows:
            client_x = []
            client_y = []
            for pos in client_rows[client]:
                client_x.append(data.x[pos])
                client_y.append(data.y[pos])
            clients_data[client] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)

    def id(self):
        return f'dirichlet_{self.num_clients}c_{self.num_labels}l_{self.skewness}s'


class PercentageDistributor(Distributor):
    def __init__(self, num_clients, min_size, max_size, percentage):
        super().__init__()
        self.num_clients = num_clients
        self.min_size = min_size
        self.max_size = max_size
        self.percentage = percentage

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_list()
        clients_data = {}
        xs = data.x
        ys = data.y
        unique_labels = np.unique(ys)
        for i in range(self.num_clients):
            client_data_size = random.randint(self.min_size, self.max_size)
            selected_label = unique_labels[random.randint(0, len(unique_labels) - 1)]
            client_x = []
            client_y = []
            while len(client_y) / client_data_size < self.percentage:
                for index, item in enumerate(ys):
                    if item == selected_label:
                        client_x.append(xs.pop(index))
                        client_y.append(ys.pop(index))
                        break
            while len(client_y) < client_data_size:
                for index, item in enumerate(ys):
                    if item != selected_label:
                        client_x.append(xs.pop(index))
                        client_y.append(ys.pop(index))
                        break
            clients_data[i] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)

    def id(self):
        return f'percentage_{self.num_clients}c_{self.percentage}p_{self.min_size}mn_{self.max_size}mx'


class LabelDistributor(Distributor):

    def __init__(self, num_clients, label_per_client, min_size, max_size,
                 is_random_label_size=False, allow_duplicate=True):
        super().__init__()
        self.num_clients = num_clients
        self.label_per_client = label_per_client
        self.min_size = min_size
        self.max_size = max_size
        self.is_random_label_size = is_random_label_size
        self.allow_duplicate = allow_duplicate

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_numpy()
        self.log(f'distributing {data}', level=0)
        clients_data = {}
        available_labels = CycleList(data.labels())
        data_selector = DataContainerLabelSelector(data, not self.allow_duplicate)
        for client_id in range(self.num_clients):
            client_data_size = random.randint(self.min_size, self.max_size)
            label_per_client = random.randint(1, self.label_per_client) if self.is_random_label_size \
                else self.label_per_client
            selected_labels = available_labels.peek(label_per_client)
            self.log(f'generating data for {client_id}-{selected_labels}')
            clients_data[client_id] = DataContainer([], [])
            for label in selected_labels:
                label_size = int(client_data_size / len(selected_labels)) or 1
                clients_data[client_id] = clients_data[client_id].concat(data_selector.peek(label, label_size))
        for cid, cdt in clients_data.items():
            clients_data[cid] = cdt.as_tensor()
        return Dict(clients_data)

    class Grouper:
        def __init__(self, x, y):
            self.grouped = defaultdict(list)
            self.selected = defaultdict(lambda: 0)
            self.label_cursor = 0
            for label, data in zip(y, x):
                self.grouped[label].append(data)
            self.all_labels = list(self.grouped.keys())

        def groups(self, count=None):
            if count is None:
                return self.all_labels
            selected_labels = []
            for i in range(count):
                selected_labels.append(self.next())
            return selected_labels

        def next(self):
            if len(self.all_labels) == 0:
                raise Exception('no more data available to distribute')

            temp = 0 if self.label_cursor >= len(self.all_labels) else self.label_cursor
            self.label_cursor = (self.label_cursor + 1) % len(self.all_labels)
            return self.all_labels[temp]

        def clean(self):
            for label, records in self.grouped.items():
                if label in self.selected and self.selected[label] >= len(records):
                    print('cleaning the good way')
                    del self.all_labels[self.all_labels.index(label)]

        def get(self, label, size=0):
            if size == 0:
                size = len(self.grouped[label])
            x = self.grouped[label][self.selected[label]:self.selected[label] + size]
            y = [label] * len(x)
            self.selected[label] += size
            if len(x) == 0 and label in self.all_labels:
                print('cleaning the wrong way')
                del self.all_labels[self.all_labels.index(label)]
            return x, y

    def id(self):
        r = '_r' if self.is_random_label_size else ''
        return f'label_{self.num_clients}c_{self.label_per_client}l_{self.min_size}mn_{self.max_size}mx' + r


class DataContainerLabelSelector:
    def __init__(self, data_container: DataContainer, raise_exception=True):
        self.data_containers = data_container.group_by(lambda x, y: y)
        self.select_cursor = defaultdict(int)
        self.raise_exception = raise_exception

    def peek(self, label, size) -> DataContainer:
        start_index = self.select_cursor[label]
        end_index = self.select_cursor[label] + size
        if end_index > len(self.data_containers[label]):
            return self._handle_out_range(label, start_index, end_index)
        self.select_cursor[label] = end_index
        return self.data_containers[label][start_index:end_index]

    def _handle_out_range(self, label, start, end):
        if self.raise_exception:
            raise Exception('data of label "{}" is out of range: {}:{}'.format(label, start, end))
        else:
            size = end - start
            max_size = len(self.data_containers[label])
            dt = self.data_containers[label][start:max_size]
            rest_size = size - len(dt)
            while rest_size > 0:
                rest_container = self.data_containers[label][0:rest_size]
                self.select_cursor[label] = rest_size
                rest_size -= len(rest_container)
                dt = dt.concat(rest_container)
            return dt


class SizeDistributor(Distributor):
    def __init__(self, num_clients, min_size, max_size):
        super().__init__()
        self.num_clients = num_clients
        self.min_size = min_size
        self.max_size = max_size

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_list()
        clients_data = Dict()
        xs = data.x
        ys = data.y
        data_pos = 0
        for i in range(self.num_clients):
            client_data_size = random.randint(self.min_size, self.max_size)
            client_x = xs[data_pos:data_pos + client_data_size]
            client_y = ys[data_pos:data_pos + client_data_size]
            data_pos += len(client_x)
            clients_data[i] = DataContainer(client_x, client_y).as_tensor()
        return Dict(clients_data)

    def id(self):
        return f'size_{self.num_clients}c_{self.min_size}mn_{self.max_size}mx'


class UniqueDistributor(Distributor):
    def __init__(self, num_clients=0, min_size=0, max_size=0):
        super().__init__()
        self.num_clients = num_clients
        self.min_size = min_size
        self.max_size = max_size

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_list()
        clients_data = Dict()
        xs = data.x
        ys = data.y
        group = {}
        for index in range(len(xs)):
            if ys[index] not in group:
                group[ys[index]] = []
            group[ys[index]].append(xs[index])
        if not self.num_clients:
            self.num_clients = len(group.keys())
        for i in range(self.num_clients):
            mins, maxs = self._sizes_values(len(group[i]))
            client_data_size = random.randint(mins, maxs)
            client_data_size = client_data_size if client_data_size else len(group[i])
            client_x = group[i][0:client_data_size]
            client_y = [i for _ in range(len(client_x))]
            clients_data[i] = DataContainer(client_x, client_y).as_tensor()
        return clients_data

    def _rsizes(self, total, size):
        if 0 < size <= 1:
            return size * total
        return size

    def _sizes_values(self, group_len):
        mins = self._rsizes(group_len, self.min_size)
        maxs = self._rsizes(group_len, self.max_size)
        if mins > maxs:
            mins, maxs = utils.swap(mins, maxs)
        return int(mins), int(maxs)

    def id(self):
        return f'unique_{self.num_clients}c_{self.min_size}mn_{self.max_size}mx'


class ShardDistributor(Distributor):
    def __init__(self, shard_size, shards_per_client):
        super().__init__()
        self.shard_size = shard_size
        self.shards_per_client = shards_per_client

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_list()
        data = self.as_shards(data)
        clients_data = {}
        index = 0
        while len(data) > 0:
            labels = data.keys()
            if len(labels) < self.shards_per_client:
                break
            selected_labels = random.sample(labels, self.shards_per_client)
            client_x = []
            client_y = []
            for label in selected_labels:
                x = self._pop(label, data).data
                y = [label] * len(x)
                client_x.extend(x)
                client_y.extend(y)
            clients_data[index] = DataContainer(client_x, client_y).as_tensor()
            index += 1
        return Dict(clients_data)

    def as_shards(self, data: DataContainer) -> typing.Dict[int, typing.List['ShardDistributor.Shard']]:
        shards = defaultdict(lambda: [self.Shard(y, shard_size)])
        shard_size = self.shard_size
        for x, y in zip(data.x, data.y):
            if shards[y][-1].is_full():
                shards[y].append(self.Shard(y, shard_size))
            shards[y][-1].append(x)
        return shards

    def _pop(self, label, shards: typing.Dict[int, typing.List['ShardDistributor.Shard']]):
        shard = shards[label].pop(0)
        if len(shards[label]) == 0:
            del shards[label]
        return shard

    class Shard:
        def __init__(self, label, max_size):
            self.label = label
            self.data = []
            self.max_size = max_size

        def is_full(self) -> bool:
            return len(self.data) >= self.max_size

        def append(self, x):
            self.data.append(x)

    def id(self):
        return f'shard_{self.shards_per_client}spp_{self.shard_size}ss'


class ManualDistributor(Distributor):
    def __init__(self, size_dict: Dict[int, int]):
        """
        distribute data to client manually based on dictionary
        Args:
            size_dict: a dictionary contains key:how much data, value:size of clients. example {10:100, 2:40} means 10
            clients will receive 100 records and 2 clients will receive 40 records
        """
        super().__init__()
        self.size_dict = size_dict

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        data = data.as_list()
        clients_data = Dict()
        data_pos = 0
        xs = data.x
        ys = data.y
        for i in self.size_dict:
            client_data_size = self.size_dict[i]
            client_x = xs[data_pos:data_pos + client_data_size]
            client_y = ys[data_pos:data_pos + client_data_size]
            data_pos += len(client_x)
            clients_data[i] = DataContainer(client_x, client_y).as_tensor()
        return clients_data

    def id(self):
        st = []
        for key, val in self.size_dict.items():
            st.append(key + ':' + val)
        st = '_'.join(st)
        return f'manual_{st}'


class PipeDistributor(Distributor):
    @staticmethod
    def pick_by_label_id(label_ids: list, size: int, repeat: int = 1) -> object:
        """
        pick from dataset records by labels
        @param label_ids: the list of label ids
        @param size: the total amount of records that should be picked
        @param repeat: how many instance this picker create. for instance, if repeat is 3 we create 3 clients
         of the same distribution
        @return: pipe function used by the distributor
        """

        def picker(grouper: LabelDistributor.Grouper):
            per_label_size = int(size / len(label_ids))
            clients = []
            for r in range(repeat):
                client_x = []
                client_y = []
                for label in label_ids:
                    rx, ry = grouper.get(label, per_label_size)
                    if len(rx) != 0:
                        client_x = rx if len(client_x) == 0 else np.concatenate((client_x, rx))
                        client_y = ry if len(client_y) == 0 else np.concatenate((client_y, ry))
                clients.append(DataContainer(client_x, client_y))
            return clients

        return picker

    def __init__(self, pipes: list, tag=''):
        super().__init__()
        self.pipes = pipes
        self.tag = tag

    def id(self):
        return f'pipes({len(self.pipes)})({self.tag})'

    def distribute(self, data: DataContainer) -> Dict[int, DataContainer]:
        all_clients = []
        grouper = LabelDistributor.Grouper(data.x, data.y)
        for pipe in self.pipes:
            clients = pipe(grouper)
            clients = [c for c in clients if len(c) > 0]
            all_clients.extend(clients)
        client_dict = {}
        for i in range(len(all_clients)):
            client_dict[i] = all_clients[i]
        return Dict(client_dict)
