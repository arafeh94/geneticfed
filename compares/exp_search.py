import pickle
import src.apis.files as fl

import dill

from src import manifest


def load(file):
    res = {}
    with open(file, 'rb') as fop:
        for key, item in dill.load(fop).items():
            res[key] = item
    return res['accuracies']


files = [
    'acc.pkl',
    'acc1.pkl',
    'acc2.pkl',
    'acc3.pkl',
    'acc4.pkl',
    'acc5.pkl',
    'acc_old.pkl',
    '2cluster/acc.pkl',
    'cifar500rounds/acc.pkl',
    'cifar500roundssgdd/acc.pkl',
    'div/acc.pkl',
    'faircluster/acc.pkl',
    'latest/acc.pkl',
    'mnist/acc.pkl',
    'mnist500rounds/acc.pkl',
    'noiid/acc.pkl',
]


def contains(st: str, *items):
    for it in items:
        if it not in st:
            return False
    return True


for f in files:
    acc = fl.AccuracyCompare(manifest.COMPARE_PATH + f).get_saved_accuracy(
        lambda c: 'basic' in c and 'r200' in c)
    if len(acc) > 0:
        print(f)
