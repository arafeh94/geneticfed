import os

from tqdm import tqdm

from src.apis import test_cases

sgd_tests = test_cases.build({
    'epoch': [1],
    'batch': [9999],
    'round': [50, 100],
    'client_ratio': [0.1, 0.2, 0.5, 1],
    'dataset': ['mnist'],
    'tag': ['basic', 'cluster', 'warmup', 'genetic'],
    'shard': [2, 4, 10]
})

for test in tqdm(sgd_tests):
    print(test)
    os.system(f"py {test['tag']}.py  -e {test['epoch']} -b {test['batch']} -r {test['round']} "
              f"-s {test['shard']} -d {test['dataset']} -cr {test['client_ratio']} -lr 0.1 -t {test['tag']}")

all_tests = test_cases.build({
    'epoch': [50, 100],
    'batch': [50],
    'round': [50, 100],
    'client_ratio': [0.1, 0.2, 0.5, 1],
    'dataset': ['mnist'],
    'tag': ['basic', 'cluster', 'warmup', 'genetic'],
    'shard': [2, 4, 10]
})

for test in tqdm(all_tests):
    print(test)
    os.system(f"py {test['tag']}.py  -e {test['epoch']} -b {test['batch']} -r {test['round']} "
              f"-s {test['shard']} -d {test['dataset']} -cr {test['client_ratio']} -lr 0.1 -t {test['tag']}")
