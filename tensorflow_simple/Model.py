import tensorflow as tf


def init_weights(name, dims, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):
    return tf.get_variable(name, trainable=True, shape=dims, initializer=initializer, regularizer=regularizer)


def init_conv(name, in_channels, out_channels, k=3):
    return init_weights(name, [k, k, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())


def conv2d(inputs, weights, strides=1):
    x = tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.relu(x)


def maxpool2d(inputs, k=2, strides=2):
    return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='VALID')


def tail(inputs, weights):
    inputs = tf.reshape(inputs, [-1, weights.get_shape().as_list()[0]])
    return tf.matmul(inputs, weights)


class Model(object):

    def __init__(self, scope, learning_rate, frames_per_step=3, move_classes=9, attack_classes=10, optim=tf.train.AdamOptimizer, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        self.scope = scope
        self.move_classes = move_classes
        self.attack_classes = attack_classes

        with tf.variable_scope(scope):
            self.optim = optim(learning_rate)

            self.conv1 = init_conv("conv1", frames_per_step, 12)
            self.conv2 = init_conv("conv2", 12, 24)
            self.conv3 = init_conv("conv3", 24, 36)
            self.conv4 = init_conv("conv4", 36, 36)
            self.conv5 = init_conv("conv5", 36, 24)
            self.fc = init_weights("fc", [24*1*3, move_classes+attack_classes])

        self.observation_sym = tf.placeholder(tf.float32, [None, 61, 120, frames_per_step])
        self.move_action_sym = tf.placeholder(tf.float32, [None, move_classes])
        self.attack_action_sym = tf.placeholder(tf.float32, [None, attack_classes])
        self.reward_sym = tf.placeholder(tf.float32, [None])

        self.move_out_sym, self.attack_out_sym = (tf.nn.softmax(action_out, axis=1) for action_out in self.eval(self.observation_sym))
        self.zero_ops, self.accum_ops, self.train_step = self.train(criterion)

        dataset = tf.data.Dataset.from_tensor_slices((self.observation_sym,
                                                      self.move_action_sym,
                                                      self.attack_action_sym,
                                                      self.reward_sym))
        self.buffer_size = tf.placeholder(tf.int64, shape=())
        self.batch_size = batch_size
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(batch_size=batch_size)
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    # Applies forward propagation to the inputs
    def eval(self, inputs):
        out = maxpool2d(conv2d(inputs, self.conv1))
        out = maxpool2d(conv2d(out, self.conv2))
        out = maxpool2d(conv2d(out, self.conv3))
        out = maxpool2d(conv2d(out, self.conv4))
        out = maxpool2d(conv2d(out, self.conv5))
        out = tail(out, self.fc)
        return out[:, 0:self.move_classes], out[:, self.move_classes:self.move_classes+self.attack_classes]

    def train(self, criterion):
        grads, lvs = zip(*self.optim.compute_gradients(self.loss(criterion), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)))
        grads_accum = [tf.Variable(tf.zeros_like(lv.initialized_value())) for lv in lvs]
        zero_ops = [grad_accum.assign(tf.zeros_like(grad_accum)) for grad_accum in grads_accum]
        accum_ops = [grads_accum[i].assign_add(grad) for i, grad in enumerate(grads)]
        gvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
        train_step = self.optim.apply_gradients(zip(grads_accum, gvs))
        return zero_ops, accum_ops, train_step

    def loss(self, criterion):
        move_out, attack_out = self.eval(self.observation_sym)
        move_loss = criterion(logits=move_out, onehot_labels=self.move_action_sym, reduction=tf.losses.Reduction.NONE) + tf.losses.get_regularization_loss()
        attack_loss = criterion(logits=attack_out, onehot_labels=self.attack_action_sym, reduction=tf.losses.Reduction.NONE) + tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.reward_sym * (move_loss + attack_loss))
