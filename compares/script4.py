import collections
import src.apis.files as fl
from src import manifest

acc1 = fl.AccuracyCompare(manifest.COMPARE_PATH + 'cifar500rounds/acc.pkl')
acc2 = fl.AccuracyCompare(manifest.COMPARE_PATH + 'cifar500roundssgdd/acc.pkl')
acc3 = fl.AccuracyCompare(manifest.COMPARE_PATH + 'faircluster/acc.pkl')
acc4 = fl.AccuracyCompare(manifest.COMPARE_PATH + 'mnist500rounds/acc.pkl')
# acc5 = fl.AccuracyCompare(manifest.COMPARE_PATH + 'acc_later.pkl')
acc5 = fl.AccuracyCompare(manifest.COMPARE_PATH + 'acc.pkl')


def fil(item: str):
    return (item.startswith('warmup') or item.startswith('genetic') or item.startswith('basic')) and 'r5_' not in item


in1 = acc1.get_saved_accuracy(fil)
in2 = acc2.get_saved_accuracy(fil)
in3 = acc3.get_saved_accuracy()
in4 = acc4.get_saved_accuracy()
in5 = acc5.get_saved_accuracy()

merged = {**in1, **in2, **in3, **in4, **in5}
merged = dict(collections.OrderedDict(sorted(merged.items())))
for key, val in merged.items():
    fl.accuracies.append(key, val)
fl.accuracies.save()
