import numpy as np
import random
from signal_classfy.utils.awgn import awgn
from sklearn.model_selection import train_test_split
from signal_classfy.utils.signal_fft import signal_fft_

SNR = 5


def data_init():
    labels = np.loadtxt('.././dataset/label.txt')
    labels = labels.reshape(16000, 4)
    signals = np.loadtxt('.././dataset/signal.txt')
    signals = signals.reshape(16000, 1024, 2)
    index = [i for i in range(len(signals))]
    random.shuffle(index)
    signals = signals[index]
    labels = labels[index]

    signals_noise = awgn(signals, SNR)

    # signal_sample = signals[2345, 1:1024, 1]
    # signals_noise_sample = signals_noise[2345, 1:1024, 1]

    # Split arrays for two 3 parts (we take only third part of dataset of labeled signals because of the memory)
    part = 1
    signals_noise = signals_noise[::part, :, :]
    labels = labels[::part]

    x_train, x_test, y_train, y_test = train_test_split(signals_noise, labels, train_size=0.6)

    # Train|validation|test = 64|16|20
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)

    return x_train, y_train, x_val, y_val, x_test, y_test


def data_test_init():
    test_list = []

    from generate_signal.QPSK.generate_QPSK import QPSK

    data = QPSK(256, 1 / 8, 1, 10)
    for i in range(100):
        data = np.vstack([data, QPSK(256, 1 / 8, 1, 10)])
    return data
