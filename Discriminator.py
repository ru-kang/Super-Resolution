import tensorflow as tf
from tensorflow.keras.layers import Input, Model, Conv2D, LeakyReLU, Dense, Flatten, BatchNormalization

def DownSampling(input_, unit, kernel_size, strides=1, bn=True):
    x = Conv2D(unit, kernel_size=kernel_size, strides=strides, padding='same')(input)
    if bn:
        x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def build_discriminator(inputshape):
    input = Input(inputshape)
    model = DownSampling(input, unit= 64, kernel_size=3, bn=False)

    model = DownSampling(model, unit=128, kernel_size=3, strides=2)
    model = DownSampling(model, unit=256, kernel_size=3, strides=2)
    feature = DownSampling(model, unit=512, kernelsize=3, strides=2)

    model = Flatten()(feature)
    model = LeakyReLU(alpha=0.2)(model)
    output = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=input, outputs=output, name='discriminator')

    fn_model = Model(inputs=input, outputs=feature, name='fn')

    return model, fn_model
