from src.app.cache import Cache
from src.app.federated_app import FederatedApp
from src.app.settings import Settings
from src.data.data_distributor import UniqueDistributor

settings = Settings({
    'cache': {
        'class': 'src.app.cache.Cache',
        'path': './samira.log',
    },
    'dataset': 'mnist',
    'distributor': {
        'class': 'src.data.data_distributor.UniqueDistributor',
        'num_clients': 10,
        'min_size': 600,
        'max_size': 600
    },
    'model': {
        'class': 'libs.model.linear.lr.LogisticRegression',
        'input_dim': 28 * 28,
        'output_dim': 10
    },
    'trainer_class': {
        'class_ref': 'src.federated.components.trainers.CPUTrainer'
    },
    'epochs': 50,
    'lr': 0.1,
    'rounds': 10,
    'batch_size': 50,
    'client_ratio': 0.2,
    'device': 'cpu'
})

app = FederatedApp(settings)
app.start()
