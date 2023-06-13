configs = [
    {
        'method': 'seq',
        'lr': 0.01,
        'wp_rounds': 50,
        'fe_rounds': 50,
        'wp_epochs': 10,
        'fe_epochs': 1
    },
    {
        'method': 'warmup',
        'lr': 0.01,
        'fe_rounds': 50,
        'wp_epochs': 500,
        'fe_epochs': 1,
        'wp_ratio': 0.05
    },
    {
        'method': 'basic',
        'lr': 0.01,
        'fe_rounds': 50,
        'fe_epochs': 1
    }
]
