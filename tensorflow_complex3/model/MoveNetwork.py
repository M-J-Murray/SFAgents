from tensorflow_complex3.model.NetworkUtils import *


class MoveNetwork(object):

    def __init__(self, scope, learning_rate, frames_per_step=3, move_classes=8, optim=tf.train.AdamOptimizer, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        self.move_classes = move_classes

        with tf.variable_scope(scope):
            with tf.variable_scope("move"):
                self.optim = optim(learning_rate, name="move_optim")
                self.network = create_network(frames_per_step, move_classes)

        self.observation_sym = tf.placeholder(tf.float32, [None, 61, 120, frames_per_step * 3])
        self.move_action_sym = tf.placeholder(tf.float32, [None, move_classes])
        self.reward_sym = tf.placeholder(tf.float32, [None])

        self.move_out_sym = tf.nn.softmax(eval(self.network, self.observation_sym), axis=1)

        if scope is not "global":
            self.compact_observation_sym = tf.placeholder(tf.int8, [None, 61, 120, frames_per_step * 3])
            self.compact_move_action_sym = tf.placeholder(tf.uint8, [None, move_classes])
            self.compact_reward_sym = tf.placeholder(tf.int8, [None])

            self.zero_ops, self.accum_ops, self.train_step = train(self.optim, self.loss(criterion), scope, "move")

            dataset = tf.data.Dataset.from_tensor_slices((self.compact_observation_sym,
                                                          self.compact_move_action_sym,
                                                          self.compact_reward_sym))
            self.buffer_size = tf.placeholder(tf.int64, shape=())
            self.batch_size = batch_size
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            dataset = dataset.batch(batch_size=batch_size)
            self.iterator = dataset.make_initializable_iterator()
            self.next_batch = self.iterator.get_next()

    def loss(self, criterion):
        move_out = eval(self.network, self.observation_sym)
        move_loss = criterion(logits=move_out, onehot_labels=self.move_action_sym, reduction=tf.losses.Reduction.NONE)
        move_loss += tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.reward_sym * move_loss)

    def train(self, sess, all_observations, all_move_actions, all_rewards):
        buffer_size = len(all_observations)
        batches = round(buffer_size / self.batch_size)

        sess.run(self.iterator.initializer, feed_dict={self.buffer_size: buffer_size,
                                                       self.compact_observation_sym: all_observations,
                                                       self.compact_move_action_sym: all_move_actions,
                                                       self.compact_reward_sym: all_rewards})
        sess.run(self.zero_ops)
        for i in range(batches):
            observation, move_actions, rewards = sess.run(self.next_batch)
            sess.run(self.accum_ops, feed_dict={
                self.observation_sym: observation,
                self.move_action_sym: move_actions,
                self.reward_sym: rewards
            })
        sess.run(self.train_step)
