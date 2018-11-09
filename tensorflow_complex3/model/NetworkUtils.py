import tensorflow as tf


def init_weights(name, dims, initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):
    return tf.get_variable(name, trainable=True, shape=dims, initializer=initializer, regularizer=regularizer)


def init_conv(name, in_channels, out_channels, k=3):
    return init_weights(name, [k, k, in_channels, out_channels],
                        initializer=tf.contrib.layers.xavier_initializer_conv2d())


def create_network(frames_per_step, out_dims):
    return {
        "conv1": init_conv("conv1", frames_per_step * 3, 32),
        "conv2": init_conv("conv2", 32, 64),
        "conv3": init_conv("conv3", 64, 128),
        "conv4": init_conv("conv4", 128, 128),
        "conv5": init_conv("conv5", 128, 64),
        "fc": init_weights("fc", [64 * 1 * 3, out_dims])
    }


def eval(network, inputs):
    out = inputs / 255
    out = conv2d(network["conv1"], out)
    out = conv2d(network["conv2"], out)
    out = conv2d(network["conv3"], out)
    out = conv2d(network["conv4"], out)
    out = conv2d(network["conv5"], out)
    return tail(network["fc"], out)


def parse_move_attack(move_attack_out, move_classes, attack_classes):
    move_out = move_attack_out[:, 0:move_classes]
    attack_out = move_attack_out[:, move_classes:move_classes + attack_classes]
    return move_out, attack_out


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
    grads, lvs = zip(*optim.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/" + subscope)))
    grads_accum = [tf.Variable(tf.zeros_like(lv.initialized_value())) for lv in lvs]
    zero_ops = [grad_accum.assign(tf.zeros_like(grad_accum)) for grad_accum in grads_accum]
    accum_ops = [grads_accum[i].assign_add(grad) for i, grad in enumerate(grads)]
    gvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global/" + subscope)
    train_step = optim.apply_gradients(zip(grads_accum, gvs))
    return zero_ops, accum_ops, train_step
