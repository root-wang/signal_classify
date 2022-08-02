from tensorflow.keras.layers import MaxPooling1D

from signal_classfy.network.Convolution_block import ConvolutionBlock
from signal_classfy.network.Residual_block import ResidualBlock

# Create residual stack
def residualStack(x, filters):
    x = ConvolutionBlock(x, filters)
    print('x')
    #     print(x.shape)
    print(x)
    x = x.unit()
    print('xunit')
    #     print(x.shape)
    print(x)

    x_shortcut = x
    x = ResidualBlock(x, x_shortcut, filters)
    x = x.unit()
    x_shortcut = x
    x = ResidualBlock(x, x_shortcut, filters)
    x = x.unit()

    # max pooling layer
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
    #     print('Residual stack created')
    return x