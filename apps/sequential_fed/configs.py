from src.data.data_distributor import ShardDistributor, DirichletDistributor, PipeDistributor
from easydict import EasyDict as edict

# distributor = PipeDistributor(
#     [
#         PipeDistributor.pick_by_label_id([1, 2, 4], 1000, 6),
#         PipeDistributor.pick_by_label_id([1, 2, 8], 1000, 6),
#         PipeDistributor.pick_by_label_id([4, 5, 8], 1000, 6),
#         PipeDistributor.pick_by_label_id([5, 6, 7, 8], 4000, 4),
#         PipeDistributor.pick_by_label_id([6, 7], 2000, 5),
#         PipeDistributor.pick_by_label_id([7, 8], 2000, 5),
#         PipeDistributor.pick_by_label_id([0], 200, 15),
#         PipeDistributor.pick_by_label_id([1], 200, 15),
#         PipeDistributor.pick_by_label_id([2], 200, 15),
#         PipeDistributor.pick_by_label_id([3], 200, 15),
#         PipeDistributor.pick_by_label_id([4], 200, 15),
#         PipeDistributor.pick_by_label_id([5], 200, 15),
#         PipeDistributor.pick_by_label_id([6], 200, 15),
#         PipeDistributor.pick_by_label_id([7], 200, 15),
#         PipeDistributor.pick_by_label_id([8], 200, 15),
#         PipeDistributor.pick_by_label_id([9], 200, 15),
#     ], tag='t1'
# )
# distributor = DirichletDistributor(150, 10, 0.1)
# distributor = ShardDistributor(400, 1)

parameters = {
    'selector': ['rand', 'ga', 'all'],
    'warmup': ['data_ratio', 'epochs', 'lr'],
    'seqop': ['selector_id', 'rounds', 'epochs', 'lr', 'cr'],
    'ewc': ['rounds', 'epochs', 'lr', 'weight', 'selector', 'cr']
}

cr = 10
wlr = 0.01
we = 5
wr = 32
dt = 'kdd'
distributor = ShardDistributor(400, 1)
# distributor = DirichletDistributor(120, 26, 0.1)

fed_config = {
    'rounds': 300,
    'lr': 0.01,
    'epochs': 25,
    'cr': 10
}

configs = edict({
    'warmup': {
        'id': f'warmup_{dt}_all_{500}_{500}_{wlr}',
        'method': 'warmup',
        'distributor': str(distributor),
        'wmp': {
            'data_ratio': 0.1,
            'lr': wlr,
            'epochs': 5,
        },
        'fed': fed_config,
    },
    'seq1': {
        'id': f'seqop_{dt}_all_{wr}_{we}_{wlr}',
        'method': 'seqop_all',
        'distributor': str(distributor),
        'wmp': {
            'selector': 'all',
            'rounds': wr,
            'epochs': we,
            'lr': wlr,
        },
        'fed': fed_config,
    },
    'seq2': {
        'id': f'seqop_{dt}_ga{cr}_{wr}_{we}_{wlr}',
        'method': 'seqop_ga',
        'distributor': str(distributor),
        'wmp': {
            'selector': 'ga',
            'rounds': wr,
            'epochs': we,
            'lr': wlr,
            'cr': cr,
        },
        'fed': fed_config,
    },
    'seq3': {
        'id': f'seqop_{dt}_rn{cr}_{wr}_{we}_{wlr}',
        'method': 'seqop_rn',
        'distributor': str(distributor),
        'wmp': {
            'selector': 'rn',
            'rounds': wr,
            'epochs': we,
            'lr': wlr,
            'cr': cr,
        },
        'fed': fed_config,
    },
    'ewc1': {
        'id': f'ewc_{dt}_all_{wr}_{we}_{wlr}',
        'method': 'ewc_all',
        'distributor': str(distributor),
        'wmp': {
            'selector': 'all',
            'rounds': wr,
            'epochs': we,
            'lr': wlr,
            'weight': 0.1,
        },
        'fed': fed_config,
    },
    'ewc2': {
        'id': f'ewc_{dt}_ga{cr}_{wr}_{we}_{wlr}',
        'method': 'ewc_ga',
        'distributor': str(distributor),
        'wmp': {
            'selector': 'ga',
            'rounds': wr,
            'epochs': we,
            'lr': wlr,
            'weight': 0.1,
            'cr': cr,
        },
        'fed': fed_config,
    },
    'ewc3': {
        'id': f'ewc_{dt}_rn{cr}_{wr}_{we}_{wlr}',
        'method': 'ewc_rn',
        'distributor': str(distributor),
        'wmp': {
            'selector': 'rn',
            'rounds': wr,
            'epochs': we,
            'lr': wlr,
            'weight': 0.1,
            'cr': cr,
        },
        'fed': fed_config,
    },
})
# 'seq': {
#     'id': 'seq',
#     'method': 'seqop',
#     'distributor': str(distributor),
#     'wmp': {
#         'selector': 'all',
#         'epochs': 2,
#         'lr': 0.01,
#         'rounds': 10,
#         'cr': 10,
#     },
#     'fed': fed_config,
# },
# 'warmup': {
#     'id': 'warmup',
#     'method': 'warmup',
#     'distributor': str(distributor),
#     'wmp': {
#         'data_ratio': 0.1,
#         'lr': 0.01,
#         'epochs': 10,
#     },
#     'fed': fed_config,
# }
# 'ewc': {
#     'id': 'ewc',
#     'method': 'ewc',
#     'distributor': str(distributor),
#     'wmp': {
#         'rounds': 32,
#         'epochs': 1,
#         'lr': 0.0001,
#         'weight': 0.1,
#         'selector': 'all',
#         'cr': 10,
#     },
#     'fed': fed_config,
# }

# configs = edict({
#     '1': {
#         'id': '{dt}_EWC_E1_CR10',
#         'method': 'seqop_rand',
#         'distributor': str(distributor),
#         'wmp': {
#             'selector_id': 'rand',
#             'epochs': 25,
#             'lr': 0.01,
#             'rounds': 32,
#             'cr': 10,
#         },
#         'fed': fed_config,
#     },
#     '2': {
#         'id': '{dt}_EWC_E1_CR10',
#         'method': 'seqop_ga',
#         'distributor': str(distributor),
#         'wmp': {
#             'selector_id': 'rand',
#             'epochs': 25,
#             'lr': 0.01,
#             'rounds': 32,
#             'cr': 10,
#         },
#         'fed': fed_config,
#     },
# })
# {
#     'id': '{dt}_EWC_E1_CR10',
#     'method': 'seq',
#     'lr': 0.01,
#     'wp_epochs': 5,
#     'wp_rounds': 30,
#     'distributor': str(distributor),
# },

#
# configs = [
#     {
#         'id': '{dt}_FedSeq_E1_CR10',
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
#         'id': '{dt}_Warmup_E1_CR10',
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
#         'id': '{dt}_Basic_E1_CR10',
#         'method': 'basic',
#         'cr': 10,
#         'lr': 0.01,
#         'fe_rounds': 100,
#         'fe_epochs': 1,
#         'distributor': str(distributor),
#     },
#     # Exp with 5E
#     {
#         'id': '{dt}_FedSeq_E5_CR10',
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
#         'id': '{dt}_Warmup_E5_CR10',
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
#         'id': '{dt}_Basic_E5_CR10',
#         'method': 'basic',
#         'cr': 10,
#         'lr': 0.01,
#         'fe_rounds': 100,
#         'fe_epochs': 5,
#         'distributor': str(distributor),
#     }
# ]
