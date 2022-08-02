from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Dropout, Dense

from signal_classfy.network.residual_stack import residualStack
from tensorflow.keras.models import Model


# define resnet model
def ResNet(input_shape, classes):
    # create input tensor
    x_input = Input(input_shape)
    x = x_input
    # residual stack
    num_filters = 20
    x = residualStack(x, num_filters)
    x = residualStack(x, num_filters)
    x = residualStack(x, num_filters)
    x = residualStack(x, num_filters)
    x = residualStack(x, num_filters)

    # output layer
    x = Flatten()(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(x)

    # Create model
    model = Model(inputs=x_input, outputs=x)
    #     print('Model ResNet created')
    return model
