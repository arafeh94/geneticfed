from src.apis import files, test_cases

cs = test_cases.build({
    'epoch': [25],
    'batch': [50],
    'round': [200],
    'client_ratio': [0.1, 0.2],
    'dataset': ['cifar10'],
    'learn_rate': [0.001, 0.01],
    'tag': ['basic', 'warmup', 'genetic', 'cluster'],
    # 'tag': ['genetic'],
    'shard': [2, 4]
})
print(len(cs))
print(len(files.accuracies.get_saved_accuracy().keys()))
