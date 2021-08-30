import tensorflow as tf


def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        # return (1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)) / 2

        return 1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, dynamic_range))

    return loss


def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):

        # imgs_pred = tf.where(tf.math.is_nan(imgs_pred), tf.zeros_like(imgs_pred), imgs_pred)

        mssim_loss = 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # l2_loss = tf.nn.l2_loss(imgs_true - imgs_pred) / 102400.0
        # l2_loss = tf.math.reduce_mean(tf.math.square(imgs_true - imgs_pred))

        return mssim_loss # + (l2_loss * 0.001)

        # return 1 - tf.reduce_mean(
        #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # )

    return loss


def l2_loss(imgs_true, imgs_pred):
    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    return tf.nn.l2_loss(imgs_true - imgs_pred)
