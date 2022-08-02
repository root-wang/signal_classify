from tensorflow.keras.layers import Activation,Add
from tensorflow.keras.layers import Conv1D


class ResidualBlock:
    kernel_size = 3
    strides = 1
    padding = 'same'
    data_format = "channels_last"

    def __init__(self, x, x_shortcut, filters):
        self.x = x
        self.filters = filters
        self.x_shortcut = x_shortcut

    def unit(self):
        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(self.x)
        x = Activation('relu')(x)
        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(x)
        x = Activation('linear')(x)
        # add skip connection
        if x.shape[1:] == self.x_shortcut.shape[1:]:
            x = Add()([x, self.x_shortcut])
        else:
            raise Exception('Skip Connection Failure!')
        return x