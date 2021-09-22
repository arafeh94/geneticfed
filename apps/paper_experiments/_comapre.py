import collections
from collections import defaultdict

from src.apis import files, test_cases
from src.federated.subscribers import FedSave


client_ratio = ['01', '02', '05', '10']
labels = ['2']
eps = [1]
batch = [9999]

for lbl in labels:
    for ep in eps:
        for cr in client_ratio:
            for bt in batch:
                files.accuracies.show_saved_accuracy_plot(
                    lambda tag: tag.endswith(f'_e{ep}_b{bt}_r50_s{lbl}_mnist_cr{cr}_lr01 ')
                )




























#
# class Dett:
#     def __init__(self, tag: str):
#         self.tag = tag.split('_')
#         self.f = self.tag[0]
#         self.e = self.tag[1].replace('e', '')
#         self.b = self.tag[2].replace('b', '')
#         self.r = self.tag[3].replace('r', '')
#         self.s = self.tag[4].replace('s', '')
#         self.cr = self.tag[6].replace('cr', '')
#
#
# sgd_tests = test_cases.build({
#     'epoch': [1],
#     'batch': [9999],
#     'round': [50, 100],
#     'client_ratio': [0.1, 0.2, 0.5, 1],
#     'dataset': ['mnist'],
#     'tag': ['basic', 'cluster', 'warmup', 'genetic'],
#     'shard': [2, 4, 10]
# })
#
# all_tests = test_cases.build({
#     'epoch': [1],
#     'batch': [9999],
#     'round': [50],
#     'client_ratio': [0.1, 0.2, 0.5, 1],
#     'dataset': ['mnist'],
#     'tag': ['basic', 'cluster', 'warmup', 'genetic'],
#     'shard': [2]
# })
# a = len(all_tests) + len(sgd_tests)
# print(a, len(files.accuracies.get_saved_accuracy()))
#
# # good : _e100_b50_r100_s2_mnist_cr01_lr01
# filters = [
#     lambda tag: tag.endswith('cluster_e50_b50_r50_s2_mnist_cr05_lr01 '),
# ]
#
# acc = files.accuracies.get_saved_accuracy()
# info = defaultdict(list)