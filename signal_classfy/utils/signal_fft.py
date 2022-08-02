import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

mpl.rcParams['axes.unicode_minus'] = True


def signal_fft_(signal):
    N = len(signal)
    y = signal

    fft_y = fft(y)
    x = np.arange(-N/2, N/2, 1)

    abs_y = np.abs(fft_y)
    angle_y = np.angle(fft_y)
    normalization_y = abs_y / N

    plt.figure()
    plt.plot(x, normalization_y, 'g')
    plt.title('双边振幅谱(归一化)', fontsize=9, color='blue')
    plt.show()
