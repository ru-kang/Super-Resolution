import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dense, Flatten, AveragePooling2D

def build_discriminator(input_shape):
    model = Sequential()

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(1024, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(AveragePooling2D())

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

input_shape = (256, 256, 3)
discriminator = build_discriminator(input_shape)