from tensoflow_complex.model.NetworkUtils import *


class ModeNetwork(object):

    def __init__(self, scope, learning_rate, frames_per_step=3, optim=tf.train.AdamOptimizer, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        with tf.variable_scope(scope):
            with tf.variable_scope("mode"):
                self.optim = optim(learning_rate, name="mode_optim")

                self.conv1 = init_conv("conv1", frames_per_step, 12)
                self.conv2 = init_conv("conv2", 12, 24)
                self.conv3 = init_conv("conv3", 24, 36)
                self.conv4 = init_conv("conv4", 36, 36)
                self.conv5 = init_conv("conv5", 36, 24)
                self.fc = init_weights("fc", [24*1*3, 3])

        self.observation_sym = tf.placeholder(tf.float32, [None, 61, 120, frames_per_step])
        self.mode_sym = tf.placeholder(tf.float32, [None, 3])
        self.reward_sym = tf.placeholder(tf.float32, [None])

        self.mode_out_sym = tf.nn.softmax(self.eval(self.observation_sym), axis=1)
        if scope is not "global":
            self.zero_ops, self.accum_ops, self.train_step = train(self.optim, self.loss_mode(criterion), scope, "mode")

            self.buffer_size = tf.placeholder(tf.int64, shape=())
            self.batch_size = batch_size
            dataset = tf.data.Dataset.from_tensor_slices((self.observation_sym, self.mode_sym, self.reward_sym))
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            dataset = dataset.batch(batch_size=batch_size)
            self.iterator = dataset.make_initializable_iterator()
            self.next_batch = self.iterator.get_next()

    # Applies forward propagation to the inputs
    def eval(self, inputs):
        out = conv2d(self.conv1, inputs)
        out = conv2d(self.conv2, out)
        out = conv2d(self.conv3, out)
        out = conv2d(self.conv4, out)
        out = conv2d(self.conv5, out)
        return tail(self.fc, out)

    def loss_mode(self, criterion):
        mode_out = self.eval(self.observation_sym)
        mode_loss = criterion(logits=mode_out, onehot_labels=self.mode_sym, reduction=tf.losses.Reduction.NONE)
        mode_loss += tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.reward_sym * mode_loss)

    def train(self, sess, all_observations, all_modes, all_rewards):
        buffer_size = len(all_observations)
        batches = round(buffer_size / self.batch_size)

        sess.run(self.iterator.initializer, feed_dict={self.buffer_size: buffer_size,
                                                       self.observation_sym: all_observations,
                                                       self.mode_sym: all_modes,
                                                       self.reward_sym: all_rewards})
        sess.run(self.zero_ops)
        for i in range(batches):
            observation, mode, rewards = sess.run(self.next_batch)
            sess.run(self.accum_ops, feed_dict={
                self.observation_sym: observation,
                self.mode_sym: mode,
                self.reward_sym: rewards
            })
        sess.run(self.train_step)
