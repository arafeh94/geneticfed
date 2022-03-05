import typing
from abc import ABC, abstractmethod
from typing import List

from apps.genetic_selectors.algo.cluster_selector import ClusterSelector
from apps.genetic_selectors_v2.algo.context import SelectorContext, InitiatorContext
from apps.genetic_selectors.algo import genetic
from apps.genetic_selectors.algo.context import Context
from src.apis.extensions import Dict
from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning
from src.federated.protocols import ClientSelector, ModelBasedClientSelector


class GeneticSelector(ModelBasedClientSelector):

    def __init__(self, initiator_context: 'InitiatorContext', nb_client, genetic_config):
        super().__init__()
        self.initiator_context = initiator_context
        self.nb_client = nb_client
        self.genetic_config = genetic_config

    def model_based_select(self, client_ids, clients_weights: Dict[int, any], sample_sizes: Dict[int, int],
                           context: 'FederatedLearning.Context'):
        self.genetic_config['c_size'] = self.nb_client
        clients_weights = clients_weights or self.initiator_context.model_stats
        sample_sizes = sample_sizes or self.initiator_context.sample_dict
        return genetic_selector(clients_weights, sample_sizes, self.genetic_config)


def genetic_selector(weight_dict, sample_dict, config) -> typing.Callable:
    context = SelectorContext(weight_dict, sample_dict)
    best, all_solutions = genetic.ga(fitness=context.ecl, genes=context.clustered_models,
                                     desired=config['desired_fitness'],
                                     max_iter=2, r_cross=config['r_cross'],
                                     r_mut=config['r_mut'], c_size=config['c_size'], p_size=config['p_size'])
    return best


def cluster_selector(clients_data, init_model, clusters=10, c_size=1):
    context = Context(clients_data, init_model)
    context.train(data_ratio=0.1)
    clustered = ClusterSelector(context.cluster(clusters))
    selected_idx = []
    while len(selected_idx) < c_size:
        available = clustered.list()
        selected_idx.append(clustered.select(available[0]))
    global_model = context.aggregate_clients(selected_idx)
    return lambda: global_model
