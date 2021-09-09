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
    Layer,
    Lambda
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


# class Sampling(Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.random.normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# https://qiita.com/kotai2003/items/3ffb3976ac240099faa8

def func_z_sample(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_log_var), mean=0, stddev=1)
    return z_mean + epsilon * tf.math.exp(z_log_var/2)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #     )
            # )
            reconstruction_loss = self.loss(data, reconstruction)
            # kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = 1 + z_log_var -tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "mssim": 1 - reconstruction_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
    img_dim = (*SHAPE, channels)

    latent_dim = 128 # 128 
    # intermid_dim = 625 * 4 # 512

    # encoder
    encoder_inputs = Input(shape=img_dim)
    x = Conv2D(32, (3, 3), padding="same")(
        encoder_inputs
    )
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(32, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(64, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(128, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Flatten()(x)
    # x = Dense(intermid_dim)(x)
    # x = LeakyReLU(alpha=0.1)(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    z = Lambda(func_z_sample, output_shape=(latent_dim))([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,))

    # x = Reshape((4, 4, latent_dim // 16))(x)
    x = Dense(5 * 5 * 128)(latent_inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = Reshape((5, 5, 128))(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(64, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(32, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)

    ## added ---------------------------------------------------------------------------
    x = Conv2D(32, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    # ---------------------------------------------------------------------------------

    x = Conv2D(
        img_dim[2], (3, 3), padding="same"
    )(x)
    # x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)
    decoded = x
    decoder = Model(latent_inputs, decoded, name="decoder")
    decoder.summary()

    # model
    autoencoder = VAE(encoder, decoder)
    return autoencoder

if __name__ == '__main__':

    def foo(src,dst):
        return tf.constant(1.0)

    vae = build_model('rgb')
    vae.compile(optimizer='adam', loss=foo)
    vae.build(input_shape=(1,160,160,3))
    vae.summary(line_length=150)