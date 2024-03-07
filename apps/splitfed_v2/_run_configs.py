import logging
import subprocess

from apps.donotuse.split_learning import models
from src.data.data_distributor import ShardDistributor
from src.data.data_loader import preload

global_configs = {
    'rounds': 50,
    'lr_client': 0.001,
    'lr_server': 0.00001,
    'client_model': models.MnistClient(784, 1024, 10),
    'server_model': models.MnistServer(784, 1024, 10),
    'train': preload('mnist', ShardDistributor(300, 2)),
    'test': preload('mnist10k').as_tensor(),
    'cls_speeds': [.1, .25, 1],
    'out_size': 5,
}

if __name__ == '__main__':
    runs = [
        './split.py',
        './splitfed.py',
        './splitfed1layer.py',
        './splitfed2layers_selection.py',
        './splitfed2layers_selection_v1.py',
        './splitfed2layers_standard.py',
        './splitfed2layers_standard_v1.py',
    ]
    logger = logging.getLogger('_run')
    for path in runs:
        logger.error('--------------Starting {} Execution--------------'.format(path))
        subprocess.run(["C:/Users/mhara/OneDrive/Documents/Projects/geneticfed/venv/Scripts/python.exe", path])
        logger.error('--------------{} Finished Execution--------------'.format(path))
