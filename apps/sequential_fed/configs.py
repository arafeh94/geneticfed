from src.data.data_distributor import ShardDistributor

distributor = ShardDistributor(400, 1)
configs = [
        {
            'id': 'MNIST_FedSeq_E1_CR10',
            'method': 'seq',
            'cr': 10,
            'lr': 0.01,
            'wp_rounds': 30,
            'fe_rounds': 100,
            'wp_epochs': 5,
            'fe_epochs': 1,
            'distributor': str(distributor),
        },
        {
            'id': 'MNIST_Warmup_E1_CR10',
            'method': 'warmup',
            'cr': 10,
            'lr': 0.01,
            'fe_rounds': 100,
            'wp_epochs': 500,
            'fe_epochs': 1,
            'wp_ratio': 0.05,
            'distributor': str(distributor),
        },
]
#
# configs = [
#     {
#         'id': 'MNIST_FedSeq_E1_CR10',
#         'method': 'seq',
#         'cr': 10,
#         'lr': 0.01,
#         'wp_rounds': 30,
#         'fe_rounds': 100,
#         'wp_epochs': 5,
#         'fe_epochs': 1,
#         'distributor': str(distributor),
#     },
#     {
#         'id': 'MNIST_Warmup_E1_CR10',
#         'method': 'warmup',
#         'cr': 10,
#         'lr': 0.01,
#         'fe_rounds': 100,
#         'wp_epochs': 500,
#         'fe_epochs': 1,
#         'wp_ratio': 0.05,
#         'distributor': str(distributor),
#     },
#     {
#         'id': 'MNIST_Basic_E1_CR10',
#         'method': 'basic',
#         'cr': 10,
#         'lr': 0.01,
#         'fe_rounds': 100,
#         'fe_epochs': 1,
#         'distributor': str(distributor),
#     },
#     # Exp with 5E
#     {
#         'id': 'MNIST_FedSeq_E5_CR10',
#         'method': 'seq',
#         'cr': 10,
#         'lr': 0.01,
#         'wp_rounds': 30,
#         'fe_rounds': 100,
#         'wp_epochs': 5,
#         'fe_epochs': 5,
#         'distributor': str(distributor),
#     },
#     {
#         'id': 'MNIST_Warmup_E5_CR10',
#         'method': 'warmup',
#         'cr': 10,
#         'lr': 0.01,
#         'fe_rounds': 100,
#         'wp_epochs': 500,
#         'fe_epochs': 5,
#         'wp_ratio': 0.05,
#         'distributor': str(distributor),
#     },
#     {
#         'id': 'MNIST_Basic_E5_CR10',
#         'method': 'basic',
#         'cr': 10,
#         'lr': 0.01,
#         'fe_rounds': 100,
#         'fe_epochs': 5,
#         'distributor': str(distributor),
#     }
# ]
