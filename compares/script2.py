import copy

from matplotlib import pyplot as plt

import src.apis.files as fl
from src import manifest

#
# path = manifest.COMPARE_PATH + './cifar_all.pkl'
# accss = fl.AccuracyCompare(path)
#
cluster = fl.AccuracyCompare(manifest.COMPARE_PATH + './faircluster/acc.pkl')
# accss = fl.AccuracyCompare(manifest.COMPARE_PATH + './cifar_all.pkl')
accss = fl.AccuracyCompare(manifest.COMPARE_PATH + './acc.pkl')

print(accss.get_saved_accuracy().keys())
print(cluster.get_saved_accuracy().keys())

plt.rcParams.update({'font.size': 28})
plt.grid()
plt.legend(loc='best')


# [print(v) for v in accs.get_saved_accuracy()]


def contains(st: str, *items):
    for it in items:
        if it not in st:
            return False
    return True


filts = [
    # # 0,0
    # [
    #     lambda item: contains(item, 'basic', 'e1', 'r500', 's2', 'cr01', 'lr0.01'),
    #     lambda item: contains(item, 'genetic', 'e1', 'r500', 's2', 'cr01', 'lr0.01'),
    #     lambda item: contains(item, 'warmup', 'e1', 'r500', 's2', 'cr01', 'lr0.01'),
    # ],
    # # 0,1
    # [
    #     lambda item: contains(item, 'genetic', 'e1', 'r500', 's4'),
    #     lambda item: contains(item, 'warmup', 'e1', 'r500', 's4'),
    #     lambda item: contains(item, 'basic', 'mnist', 'r500', 's2', 'cr01', 'lr0.001', 'e25'),
    # ],
    # # 0,2
    # [
    #     lambda item: contains(item, 'genetic', 'e1', 'r500', 's10'),
    #     lambda item: contains(item, 'warmup', 'e1', 'r500', 's10'),
    #     lambda item: contains(item, 'basic', 'mnist', 'r500', 's2', 'cr01', 'lr0.001', 'e25'),
    # ],
    # # 1,0
    # [
    #     lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's2', 'basic', ),
    #     lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's2', 'genetic', ),
    #     lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's2', 'warmup', ),
    # ],
    # 1,1
    # [
    #     lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's4', 'basic', ),
    #     lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's4', 'genetic', ),
    #     lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's4', 'warmup', ),
    # ],
    # # 1,2
    [
        lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's10', 'cr01', 'basic', ),
        lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's10', 'cr01', 'genetic', ),
        lambda item: contains(item, 'lr0.1', 'e25', 'r500', 's10', 'cr01', 'warmup', ),
    ],
    # [
    #     lambda item: contains(item, 'cifar', 's2'),
    # ],
    # [
    #     lambda item: contains(item, 'cifar', 's4'),
    #     lambda item: contains(item, 'cifar', 's2', 'warmup'),
    # ],
    # [
    #     lambda item: contains(item, 'cifar', 's10'),
    # ],
    # [
    #     lambda item: contains(item, 'cifar', 'e5', 'genetic', 's10', 'cr01'),
    #     lambda item: contains(item, 'cifar', 'e5', 'warmup', 's10', 'cr01'),
    #     lambda item: contains(item, 'cifar', 'e5', 'basic', 's10', 'cr01'),
    # ],
    # [
    #     lambda item: contains(item, 'cifar', 's4'),
    #     lambda item: contains(item, 'cifar', 's2', 'warmup'),
    # ],
    # [
    #     lambda item: contains(item, 'cifar', 's10'),
    # ]
]

for ft in filts:
    all = {}
    for fil in ft:
        all = {**all, **accss.get_saved_accuracy(fil)}

    # cluster_acc = cluster.get_saved_accuracy(
    #     lambda tag: 'cluster' in tag and 's10' in tag and 'e25' in tag and 'lr0.001' in tag
    # )
    # all = {**all, **cluster_acc}
    print(all.keys())

    accss.show_saved_accuracy_plot_acc(all)
