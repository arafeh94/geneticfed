import multiprocessing
import os

from tqdm import tqdm

from src.apis import test_cases, files


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
# for test in tqdm(sgd_tests):
#     print(test)
#     os.system(f"py {test['tag']}.py  -e {test['epoch']} -b {test['batch']} -r {test['round']} "
#               f"-s {test['shard']} -d {test['dataset']} -cr {test['client_ratio']} -lr 0.1 -t {test['tag']}")

def build_tag(args):
    return f'{args["tag"]}_e{args["epoch"]}_b{args["batch"]}_r{args["round"]}_s{args["shard"]}' \
           f'_{args["dataset"]}_cr{str(args["client_ratio"]).replace(".", "")}' \
           f'_lr{str(args["learn_rate"])}'.replace('cr1', 'cr10')


a = files.accuracies.get_saved_accuracy()


def run_command(execution, force=False):
    save_tag = build_tag(execution)
    if force or save_tag not in files.accuracies.get_saved_accuracy():
        print(f'Executing: {save_tag}')
        os.system(f"py {execution['tag']}.py  -e {execution['epoch']} -b {execution['batch']} -r {execution['round']} "
                  f"-s {execution['shard']} -d {execution['dataset']} -cr {execution['client_ratio']} -lr {execution['learn_rate']} "
                  f"-t {execution['tag']}")
    else:
        print("Experiments already executed, skip.")


all_tests = test_cases.build({
    'epoch': [25],
    'batch': [50],
    'round': [10],
    'client_ratio': [0.1],
    'dataset': ['cifar10'],
    'learn_rate': [0.001],
    'tag': ['basic'],
    # 'tag': ['basic', 'warmup', 'genetic', 'cluster'],
    'shard': [2]
})

if __name__ == '__main__':
    # pool = multiprocessing.Pool(2)
    # pool.map(run_command, all_tests)
    for t in all_tests:
        run_command(t, True)
