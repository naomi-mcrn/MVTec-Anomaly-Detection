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
    AveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


# Preprocessing parameters
RESCALE = 1.0 / 255
# SHAPE = (256, 256)
# SHAPE = (640, 640)
SHAPE = (160, 160)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN

SR_MULTIPLIER = 1

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
    encoding_dim = 16 # 64  # 128


    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # # added ---------------------------------------------------------------------------
    # x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    # x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    # x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    # x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D((2, 2), padding="same")(x)

    # 入力640にしたので、160まで下げるために強引にやる

    if SR_MULTIPLIER > 1:
        x = AveragePooling2D(pool_size=(SR_MULTIPLIER, SR_MULTIPLIER))(x)

    x = Flatten()(x)
    x = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    # encoded = x

    # decoder

    x = Dense(80 * 80 * 512, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Reshape((80, 80, 512))(x)

    # x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = UpSampling2D((2, 2))(x)

    # x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    # x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    # x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(512, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(
        img_dim[2], (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-6)
    )(x)
    # x = Dense(img_dim[0] * img_dim[1] * img_dim[2] // (SR_MULTIPLIER * SR_MULTIPLIER), kernel_regularizer=regularizers.l2(1e-6))(x)
    # x = Reshape((img_dim[0] // SR_MULTIPLIER, img_dim[1] // SR_MULTIPLIER, img_dim[2]))(x)

    if SR_MULTIPLIER > 1:
        x = UpSampling2D((SR_MULTIPLIER, SR_MULTIPLIER))(x)

        # super resolution CNN (SRCNN)
        x = Conv2D(64, (9,9), activation='relu', padding='same')(x)
        x = Conv2D(32, (1,1), activation='relu', padding='same')(x)
        x = Conv2D(3, (5,5), padding='same')(x)

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
