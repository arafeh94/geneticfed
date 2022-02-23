import multiprocessing
import os
import sys

sys.path.append("../../")
from src.apis import test_cases, files


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


key_tests = {
    'mnist': test_cases.build({
        'epoch': [500],
        'batch': [50],
        'round': [500],
        'client_ratio': [0.001],
        'dataset': ['cifar10'],
        'learn_rate': [0.001],
        'tag': ['genetic'],
        'shard': [10]
    }),
}

if __name__ == '__main__':
    # pool = multiprocessing.Pool(2)
    # pool.map(run_command, all_tests)
    for kt in key_tests:
        print(f"starting {kt} tests: {len(key_tests[kt])}")
        for t in key_tests[kt]:
            run_command(t)
