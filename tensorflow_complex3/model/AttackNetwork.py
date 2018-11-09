from tensorflow_complex3.model.NetworkUtils import *


class AttackNetwork(object):

    def __init__(self, scope, learning_rate, frames_per_step=3, attack_classes=9, optim=tf.train.AdamOptimizer, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        self.attack_classes = attack_classes

        with tf.variable_scope(scope):
            with tf.variable_scope("attack"):
                self.optim = optim(learning_rate, name="attack_optim")
                self.network = create_network(frames_per_step, attack_classes)

        self.observation_sym = tf.placeholder(tf.float32, [None, 61, 120, frames_per_step*3])
        self.attack_action_sym = tf.placeholder(tf.float32, [None, attack_classes])
        self.reward_sym = tf.placeholder(tf.float32, [None])

        self.attack_out_sym = tf.nn.softmax(eval(self.network, self.observation_sym), axis=1)

        if scope is not "global":
            self.compact_observation_sym = tf.placeholder(tf.int8, [None, 61, 120, frames_per_step * 3])
            self.compact_attack_action_sym = tf.placeholder(tf.uint8, [None, attack_classes])
            self.compact_reward_sym = tf.placeholder(tf.int8, [None])

            self.zero_ops, self.accum_ops, self.train_step = train(self.optim, self.loss(criterion), scope, "attack")

            dataset = tf.data.Dataset.from_tensor_slices((self.compact_observation_sym,
                                                          self.compact_attack_action_sym,
                                                          self.compact_reward_sym))
            self.buffer_size = tf.placeholder(tf.int64, shape=())
            self.batch_size = batch_size
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            dataset = dataset.batch(batch_size=batch_size)
            self.iterator = dataset.make_initializable_iterator()
            self.next_batch = self.iterator.get_next()

    def loss(self, criterion):
        attack_out = eval(self.network, self.observation_sym)
        attack_loss = criterion(logits=attack_out, onehot_labels=self.attack_action_sym, reduction=tf.losses.Reduction.NONE)
        attack_loss += tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.reward_sym * attack_loss)

    def train(self, sess, all_observations, all_attack_actions, all_rewards):
        buffer_size = len(all_observations)
        batches = round(buffer_size / self.batch_size)

        sess.run(self.iterator.initializer, feed_dict={self.buffer_size: buffer_size,
                                                       self.compact_observation_sym: all_observations,
                                                       self.compact_attack_action_sym: all_attack_actions,
                                                       self.compact_reward_sym: all_rewards})
        sess.run(self.zero_ops)
        for i in range(batches):
            observation, attack_actions, rewards = sess.run(self.next_batch)
            sess.run(self.accum_ops, feed_dict={
                self.observation_sym: observation,
                self.attack_action_sym: attack_actions,
                self.reward_sym: rewards
            })
        sess.run(self.train_step)
