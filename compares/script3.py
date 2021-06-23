import matplotlib.pyplot as plt
import numpy as np

import src.apis.files as fl
import src.tools as tools
from src import manifest

dv = fl.DivergenceCompare(manifest.COMPARE_PATH + 'div/div.pkl')
for k, v in dv.get_saved_divergences().items():
    print(k, list(v[:10]))
