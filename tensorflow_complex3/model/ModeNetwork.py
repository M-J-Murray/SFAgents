from tensorflow_complex3.model.NetworkUtils import *


class ModeNetwork(object):

    def __init__(self, scope, learning_rate, frames_per_step=3, optim=tf.train.AdamOptimizer, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        with tf.variable_scope(scope):
            with tf.variable_scope("mode"):
                self.optim = optim(learning_rate, name="mode_optim")
                self.network = create_network(frames_per_step, 3)

        self.observation_sym = tf.placeholder(tf.float32, [None, 61, 120, frames_per_step*3])
        self.mode_sym = tf.placeholder(tf.float32, [None, 3])
        self.reward_sym = tf.placeholder(tf.float32, [None])

        self.mode_out_sym = tf.nn.softmax(eval(self.network, self.observation_sym), axis=1)
        if scope is not "global":
            self.compact_observation_sym = tf.placeholder(tf.int8, [None, 61, 120, frames_per_step * 3])
            self.compact_mode_sym = tf.placeholder(tf.uint8, [None, 3])
            self.compact_reward_sym = tf.placeholder(tf.int8, [None])

            self.zero_ops, self.accum_ops, self.train_step = train(self.optim, self.loss_mode(criterion), scope, "mode")

            self.buffer_size = tf.placeholder(tf.int64, shape=())
            self.batch_size = batch_size
            dataset = tf.data.Dataset.from_tensor_slices((self.compact_observation_sym, self.compact_mode_sym, self.compact_reward_sym))
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            dataset = dataset.batch(batch_size=batch_size)
            self.iterator = dataset.make_initializable_iterator()
            self.next_batch = self.iterator.get_next()

    def loss_mode(self, criterion):
        mode_out = eval(self.network, self.observation_sym)
        mode_loss = criterion(logits=mode_out, onehot_labels=self.mode_sym, reduction=tf.losses.Reduction.NONE)
        mode_loss += tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.reward_sym * mode_loss)

    def train(self, sess, all_observations, all_modes, all_rewards):
        buffer_size = len(all_observations)
        batches = round(buffer_size / self.batch_size)

        sess.run(self.iterator.initializer, feed_dict={self.buffer_size: buffer_size,
                                                       self.compact_observation_sym: all_observations,
                                                       self.compact_mode_sym: all_modes,
                                                       self.compact_reward_sym: all_rewards})
        sess.run(self.zero_ops)
        for i in range(batches):
            observation, mode, rewards = sess.run(self.next_batch)
            sess.run(self.accum_ops, feed_dict={
                self.observation_sym: observation,
                self.mode_sym: mode,
                self.reward_sym: rewards
            })
        sess.run(self.train_step)
