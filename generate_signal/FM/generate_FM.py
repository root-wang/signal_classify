import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from signal_classfy.utils.signal_fft import signal_fft_


def FM(fm, fc, kf, fs, nb, T=1.0):
    """
    :param fm: 基带信号余弦波频率
    :param fc: 调制载波频率
    :param kf: 调频灵敏度
    :param fs: 采样频率
    :param nb: 基带信号码元个数
    :param T : 码元周期
    :return:
    """
    # 采样间隔
    ts = 1 / fs

    # 总采样点数
    t = np.arange(0, nb) * ts

    # 基带信号产生
    cos_m = np.cos(2 * np.pi * fm * t)

    # Calculate integration
    integration = np.cumsum(cos_m) * ts

    # 载波
    st = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integration)

    plt.Figure()

    # plt.subplot(211)
    plt.plot(t[1000:], cos_m[1000:], 'b')

    # 相干解调
    # 锁相环解调
    vco_phase = np.zeros(nb)
    rt = np.zeros(nb)
    et = np.zeros(nb)
    vt = np.zeros(nb)
    Av = 1  # VCO输出幅度
    Kv = 40000  # VCO频率灵敏度
    Km = 1  # PD增益
    K0 = 1  # LF增益

    rt[0] = np.cos(vco_phase[0])
    et[0] = Km * st[0] * rt[0]

    b0 = 0.07295965726826667  # Fs = 40000，fcut = 1000的1阶巴特沃斯低通滤波器系数，由FDA生成
    b1 = 0.07295965726826667
    a1 = -0.8540806854634666

    vt[0] = K0 * (b0 * et[0])

    for i in range(1, nb):
        vco_phase_change = 2 * np.pi * fc * ts + Kv * vt[i - 1] * ts
        vco_phase[i] = vco_phase[i - 1] + vco_phase_change

        rt[i] = Av * np.cos(vco_phase[i])

        et[i] = Km * st[i] * rt[i]

        vt[i] = K0 * (b0 * et[i] + b1 * et[i - 1] - a1 * vt[i - 1])
    vt = (vt / np.max(vt)) * np.max(cos_m)
    # plt.subplot(212)
    plt.plot(t[1000:], vt[1000:], 'g')
    plt.axis([0, 0.06, -7, 7])
    plt.show()

    print(np.sum(vt[1000:] - cos_m[1000:]))


fs = 40000
Ts = 1 / fs
N = 5000
kf = 2000
fm = 500
fc = 10000
FM(fm, fc, kf, fs, N)
