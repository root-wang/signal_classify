import numpy as np


def awgn(x, snr, seed=7):
    """
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    """
    sample_nums = x.shape[0]
    sample_noise = []
    for sample_num in range(sample_nums):
        snr = np.random.randint(-18, 19, 1)[0]
        for channel in range(2):
            sample = x[sample_num, :, channel]
            # sample = x
            # np.random.seed(seed)  # 设置随机种子
            snr = 10 ** (snr / 10.0)
            xpower = np.sum(sample ** 2) / len(sample)
            npower = xpower / snr
            noise = np.random.randn(len(sample)) * np.sqrt(npower)
            sample_noise.append(sample + noise)
    return np.array(sample_noise).reshape(x.shape)
