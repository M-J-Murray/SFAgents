from tensoflow_complex.model.NetworkUtils import *


class MoveAttackNetwork(object):

    def __init__(self, scope, learning_rate, frames_per_step=3, move_classes=8, attack_classes=9, optim=tf.train.AdamOptimizer, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        self.move_classes = move_classes
        self.attack_classes = attack_classes

        with tf.variable_scope(scope):
            with tf.variable_scope("move_attack"):
                self.optim = optim(learning_rate, name="move_attack_optim")

                self.conv1 = init_conv("conv1", frames_per_step, 12)
                self.conv2 = init_conv("conv2", 12, 24)
                self.conv3 = init_conv("conv3", 24, 36)
                self.conv4 = init_conv("conv4", 36, 36)
                self.conv5 = init_conv("conv5", 36, 24)
                self.fc = init_weights("fc", [24*1*3, move_classes+attack_classes])

        self.observation_sym = tf.placeholder(tf.float32, [None, 61, 120, frames_per_step])
        self.mode_sym = tf.placeholder(tf.float32, [None, 3])
        self.move_action_sym = tf.placeholder(tf.float32, [None, move_classes])
        self.attack_action_sym = tf.placeholder(tf.float32, [None, attack_classes])
        self.reward_sym = tf.placeholder(tf.float32, [None])

        self.move_out_sym, self.attack_out_sym = (tf.nn.softmax(out, axis=1) for out in self.eval(self.observation_sym))

        if scope is not "global":
            self.zero_ops, self.accum_ops, self.train_step = train(self.optim, self.loss(criterion), scope, "move_attack")

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

    def eval(self, inputs):
        out = conv2d(self.conv1, inputs)
        out = conv2d(self.conv2, out)
        out = conv2d(self.conv3, out)
        out = conv2d(self.conv4, out)
        out = conv2d(self.conv5, out)
        move_attack_out = tail(self.fc, out)
        move_out = move_attack_out[:, 0:self.move_classes]
        attack_out = move_attack_out[:, self.move_classes:self.move_classes+self.attack_classes]
        return move_out, attack_out

    def loss(self, criterion):
        move_out, attack_out = self.eval(self.observation_sym)
        move_loss = criterion(logits=move_out, onehot_labels=self.move_action_sym, reduction=tf.losses.Reduction.NONE) + tf.losses.get_regularization_loss()
        attack_loss = criterion(logits=attack_out, onehot_labels=self.attack_action_sym, reduction=tf.losses.Reduction.NONE) + tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.reward_sym * (move_loss + attack_loss))

    def train(self, sess, all_observations, all_move_actions, all_attack_actions, all_rewards):
        buffer_size = len(all_observations)
        batches = round(buffer_size / self.batch_size)

        sess.run(self.iterator.initializer, feed_dict={self.buffer_size: buffer_size,
                                                       self.observation_sym: all_observations,
                                                       self.move_action_sym: all_move_actions,
                                                       self.attack_action_sym: all_attack_actions,
                                                       self.reward_sym: all_rewards})
        sess.run(self.zero_ops)
        for i in range(batches):
            observation, move_actions, attack_actions, rewards = sess.run(self.next_batch)
            sess.run(self.accum_ops, feed_dict={
                self.observation_sym: observation,
                self.move_action_sym: move_actions,
                self.attack_action_sym: attack_actions,
                self.reward_sym: rewards
            })
        sess.run(self.train_step)
