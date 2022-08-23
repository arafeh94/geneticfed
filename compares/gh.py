from apps.paper_experiments.tag_builder import build_tag
from src.apis import files, test_cases

cs = test_cases.build({
    'epoch': [5],
    'batch': [50],
    'round': [500],
    'client_ratio': [0.1],
    'dataset': ['cifar10'],
    'learn_rate': [0.001],
    'tag': ['basic', 'warmup', 'genetic', 'cluster'],
    'shard': [2, 4, 10]
})

accs = files.accuracies.get_saved_accuracy()
no = []
for c in cs:
    tag = build_tag(c)
    if tag not in accs:
        no.append(tag)
print(no)