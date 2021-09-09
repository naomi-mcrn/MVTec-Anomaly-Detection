"""
Model inspired by: https://github.com/natasasdj/anomalyDetection
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    LeakyReLU,
    Activation,
    concatenate,
    Flatten,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


# Preprocessing parameters
RESCALE = 1.0 / 255
# SHAPE = (256, 256)
SHAPE = (160, 160)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    # input
    x = input_img = Input(shape=img_dim)



    # encoder
    base_size = img_dim[0]
    conv_chs = [32, 32, 64, 64, 128]
    conv_stacks = [1,1,1,1,1]
    last_fmap_size = base_size // (2 ** len(conv_chs))
    last_fmap_ch = conv_chs[-1]

    encoding_dim = 100 # 64  # 128

    for ch, stack in zip(conv_chs, conv_stacks):
        for _ in range(stack):
            x = Conv2D(ch, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

    x = Flatten()(x)
    x = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # encoded = x

    # decoder
    # x = Reshape((4, 4, encoding_dim // 16))(x)

    x = Dense(last_fmap_size * last_fmap_size * last_fmap_ch, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Reshape((last_fmap_size, last_fmap_size, last_fmap_ch))(x)


    for ch in reversed(conv_chs):
        x = Conv2D(ch, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

    x = Conv2D(
        img_dim[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder

if __name__ == '__main__':

    # def foo(src,dst):
    #     return tf.constant(1.0)

    vae = build_model('rgb')
    vae.compile(optimizer='adam', loss='mse')
    # vae.build(input_shape=(1,160,160,3))
    vae.summary(line_length=150)
