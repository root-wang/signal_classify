import numpy as np


def base_signal(nb):
    data = [1 if x > 0.5 else 0 for x in np.random.randn(1, nb)[0]]
    return np.array(data)

