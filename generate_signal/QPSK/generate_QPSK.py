from generate_signal.utils.generate_base import base_signal
from signal_classfy.utils.awgn import awgn

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from signal_classfy.utils.signal_fft import signal_fft_


def QPSK(nb, fs, T, fc):
    data = base_signal(nb)
    data_nrz = np.multiply(2, data) - 1

    # spilt 2 I/Q
    data_i = data_nrz[0:(nb - 1):2]
    data_q = data_nrz[1:nb:2]

    ich = []
    qch = []

    # 上采样 1/delta_T倍
    for i in range(int(nb / 2)):
        ich += [data_i[i]] * int(T * fs * 2)
        qch += [data_q[i]] * int(T * fs * 2)

    cos_carry = []
    sin_carry = []

    # 总采样点数
    t = np.arange(0, nb * T * 2, 1 / fs)
    N = len(t)

    # 载波生成
    for i in range(int(N / 2)):
        cos_carry.append(np.math.cos(2 * np.pi * fc * t[i]))
        sin_carry.append(np.math.sin(2 * np.pi * fc * t[i]))

    # IQ两路调制
    data_i_1 = np.array(ich) * np.array(cos_carry)
    data_q_1 = np.array(qch) * np.array(sin_carry)

    # s = np.array([np.vstack([np.array([data_i_1]), np.array([data_q_1])]).T])
    # 将IQ两路信号加
    s = data_i_1 - data_q_1
    # plt.figure(figsize=(30,10))
    # plt.plot(s)
    # plt.axis([0, 1500, -2, 2])
    # plt.show()
    # 通过高斯噪声
    s = awgn(s, -8)
    signal_fft_(s)
    # 解调

    demodulate_I = np.multiply(s, cos_carry)
    demodulate_Q = np.multiply(s, np.multiply(-1, sin_carry))

    # return demodulate_I, demodulate_Q

    b, a = signal.butter(8, 0.8, 'lowpass')

    demodulate_I = signal.filtfilt(b, a, demodulate_I)
    demodulate_Q = signal.filtfilt(b, a, demodulate_Q)

    recover_I = []
    recover_Q = []
    for i in range(int(nb / 2)):
        output_I = demodulate_I[i * (int(T * fs * 2)):(i + 1) * (int(T * fs * 2))]
        if np.sum(output_I) > 0:
            recover_I.append(1)
        else:
            recover_I.append(0)
        output_Q = demodulate_Q[i * (int(T * fs * 2)):(i + 1) * (int(T * fs * 2))]
        if np.sum(output_Q) > 0:
            recover_Q.append(1)
        else:
            recover_Q.append(0)
    out_I = np.array(recover_I) * 2 - 1 - np.array(data_i)
    out_Q = np.array(recover_Q) * 2 - 1 - np.array(data_q)
    print(np.sum(out_I))
    print(np.sum(out_Q))
    # recover_i = s*
    #
    # plt.figure()
    # plt.scatter(data_i_1, data_q_1)
    # plt.show()


# 载波频率
fc = 20
# 码元速率
T = 1
# 采样频率
fs = 4 * fc
# 码元数
N = 100000

QPSK(N, fs, T, fc)

# from signal_classfy.utils.signal_fft import signal_fft_
#
# signal_fft_(demodulate_I)
# signal_fft_(demodulate_Q)
