import os
import pickle
import typing

import numpy as np
from matplotlib import pyplot as plt

from src import manifest
from src.apis.extensions import Serializable


class AccuracyCompare(Serializable):
    def __init__(self):
        super().__init__(manifest.DEFAULT_ACC_PATH)
        self.accuracies = {}
        self.load()

    def append(self, tag, val):
        self.accuracies[tag] = val
        self.save()

    def save_accuracy(self, federated_learning, tag):
        def reducer(first, key, val):
            return [val['acc']] if first is None else np.append(first, val['acc'])

        all_acc = federated_learning.context.history.reduce(reducer)
        self.append(tag, all_acc)

    def get_saved_accuracy(self):
        self.load()
        return self.accuracies

    def show_saved_accuracy_plot(self, filter: typing.Callable[[str], bool] = None):
        accs = self.get_saved_accuracy()
        for tag, vals in accs.items():
            is_not_filtered = False if filter is None else filter(tag)
            if vals is not None and is_not_filtered:
                plt.plot(vals, label=tag)
                plt.legend()
        plt.show()


accuracies = AccuracyCompare()
