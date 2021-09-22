import collections
from collections import defaultdict

from src.apis import files, test_cases
from src.federated.subscribers import FedSave

files.divergences.show_saved_divergences_plot(lambda x: 'sgd2' in x, title='FedSGD Ψ=2')
