import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, Add, Conv2DTranspose

def build_residual_block(input_tensor):
    residual = input_tensor

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(residual)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    output_tensor = Add()([x, residual])

    return output_tensor

def build_generator(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(input_layer)
    x = PReLU()(x)

    for _ in range(8):
        x = build_residual_block(x)

    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = PReLU()(x)

    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = PReLU()(x)

    output_layer = Conv2D(3, kernel_size=(3, 3), activation='tanh', padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer, name='generator')

    return model

input_shape = (64, 64, 3)
generator = build_generator(input_shape)
generator.summary()
