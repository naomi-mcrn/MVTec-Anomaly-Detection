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

        # epsilon = 1e-8
        # epsilon_sqr = tf.constant(epsilon ** 2)

        # tf.print("TRUE", imgs_true)
        # tf.print("PRED", imgs_pred)
        # tf.print(tf.reduce_max(imgs_pred))
        # tf.print(tf.reduce_min(imgs_pred))

        mssim_loss = 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range, filter_size=9)
        # l2_loss = tf.nn.l2_loss(imgs_true - imgs_pred) / 102400.0
        # l2_loss = tf.math.reduce_mean(tf.math.square(imgs_true - imgs_pred))

        # l1_loss = blurred_l1_loss(imgs_true, imgs_pred)

        # l1_loss = tf.math.reduce_variance(tf.math.reduce_mean(tf.math.sqrt(tf.math.square(imgs_true - imgs_pred) + epsilon_sqr), axis=-1))
        # l1_diff = tf.math.reduce_mean(tf.math.sqrt(tf.math.square(imgs_true - imgs_pred) + epsilon_sqr), axis=-1)
        # l1_max = tf.math.reduce_max(l1_diff)
        # l1_min = tf.math.reduce_min(l1_diff)
        # l1_mass = tf.math.square(tf.reduce_sum(l1_diff)) 

        return mssim_loss #  + l2_loss(imgs_true, imgs_pred) # * (1.0 + l1_loss) #  + (l1_loss / (tf.reduce_mean(mssim_loss) * 100.0))

        # return 1 - tf.reduce_mean(
        #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # )

    return loss


def l2_loss(imgs_true, imgs_pred):
    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    return tf.nn.l2_loss(imgs_true - imgs_pred)

def blurred_l1_loss(imgs_true, imgs_pred):
    epsilon = 1e-6
    epsilon_sqr = tf.constant(epsilon ** 2)
    l1_loss = tf.math.reduce_mean(tf.math.sqrt(tf.math.square(imgs_true - imgs_pred) + epsilon_sqr), axis=-1)
    return tf.math.reduce_mean(tf.nn.avg_pool(l1_loss, 8, 1, 'VALID'))

