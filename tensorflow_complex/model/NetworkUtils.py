import tensorflow as tf


def init_weights(name, dims, initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):
    return tf.get_variable(name, trainable=True, shape=dims, initializer=initializer, regularizer=regularizer)


def init_conv(name, in_channels, out_channels, k=3):
    return init_weights(name, [k, k, in_channels, out_channels],
                        initializer=tf.contrib.layers.xavier_initializer_conv2d())


def conv2d(weights, inputs, strides=1):
    x = tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.relu(x)
    return maxpool2d(x)


def maxpool2d(inputs, k=2, strides=2):
    return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='VALID')


def tail(weights, inputs):
    inputs = tf.reshape(inputs, [-1, weights.get_shape().as_list()[0]])
    return tf.matmul(inputs, weights)


def train(optim, loss, scope, subscope):
    grads, lvs = zip(*optim.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+"/"+subscope)))
    grads_accum = [tf.Variable(tf.zeros_like(lv.initialized_value())) for lv in lvs]
    zero_ops = [grad_accum.assign(tf.zeros_like(grad_accum)) for grad_accum in grads_accum]
    accum_ops = [grads_accum[i].assign_add(grad) for i, grad in enumerate(grads)]
    gvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global/"+subscope)
    train_step = optim.apply_gradients(zip(grads_accum, gvs))
    return zero_ops, accum_ops, train_step