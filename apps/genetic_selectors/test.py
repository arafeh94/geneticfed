# mpiexec -n 4 python main_mpi.py
import logging
import sys
from os.path import dirname

from torch import nn

from apps.flsim.src.client_selector import RLSelector
from apps.flsim.src.initializer import rl_module_creator
from libs.model.cv.cnn import CNN, CNN_OriginalFedAvg
from src import manifest
from src.apis import files
from src.data.data_loader import preload
from src.federated.subscribers import Timer

sys.path.append(dirname(__file__) + '../')

from libs.model.linear.lr import LogisticRegression
from src.federated.components.trainers import TorchTrainer
from src.federated.protocols import TrainerParams
from apps.genetic_selectors.algo import initializer
from src.federated.components import metrics, client_selectors, aggregators
from src.federated import subscribers, fedruns
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.components.trainer_manager import SeqTrainerManager
from src.data import data_generator, data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

dt = preload('cifar10_10shards_100c_400min_400max', 'cifar10', lambda dg: dg.distribute_shards(100, 10, 400, 400))
print(dt)