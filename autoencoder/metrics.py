import tensorflow as tf
import tensorflow.keras.backend as K


def ssim_metric(dynamic_range):
    def ssim(imgs_true, imgs_pred):
        return K.mean(tf.image.ssim(imgs_true, imgs_pred, dynamic_range), axis=-1)

    return ssim


def mssim_metric(dynamic_range):
    def mssim(imgs_true, imgs_pred):
        return K.mean(
            tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range, filter_size=9), axis=-1
        )

    return mssim

def l1_std_metric(dynamic_range):
    def l1_std(imgs_true, imgs_pred):
        return K.std(
            tf.math.reduce_mean(tf.math.sqrt(tf.math.square(imgs_true - imgs_pred)), axis=-1)
        )

    return l1_std

def blurred_l1_metric(imgs_true, imgs_pred):
    l1_loss = tf.math.reduce_mean(tf.math.sqrt(tf.math.square(imgs_true - imgs_pred)), axis=-1)
    return K.mean(tf.nn.avg_pool(l1_loss, 8, 1, 'VALID'))