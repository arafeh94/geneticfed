import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
from src import manifest

dv = fl.DivergenceCompare(manifest.COMPARE_PATH + 'div/div.pkl')
for k, v in dv.get_saved_divergences().items():
    print(k, list(v[:10]))
dv.show_saved_divergences_plot(lambda c: 'e25' in c)
