from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D


class ConvolutionBlock:
    kernel_size = 1
    strides = 1
    padding = 'same'
    data_format = "channels_last"

    def __init__(self, x, filters):
        self.x = x
        self.filters = filters

    def unit(self):
        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(self.x)
        x = Activation('linear')(x)
        return x
