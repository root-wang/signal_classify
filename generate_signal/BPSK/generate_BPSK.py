
from generate_signal.utils.generate_base import base_signal

import numpy as np

def BPSK(nb, delta_T, T, fc=None):
    data = base_signal(nb)
