from Main.Agent.TensorflowMulti.Model.NetworkUtils import *


class MoveNetwork(object):

    def __init__(self, scope, optim=None, frames_per_step=3, move_classes=8, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        self.move_classes = move_classes

        with tf.variable_scope(scope):
            with tf.variable_scope("move"):
                self.conv1 = init_conv("conv1", frames_per_step, 12)
                self.conv2 = init_conv("conv2", 12, 24)
                self.conv3 = init_conv("conv3", 24, 36)
                self.conv4 = init_conv("conv4", 36, 36)
                self.conv5 = init_conv("conv5", 36, 24)
                self.fc = init_weights("fc", [24*1*3, move_classes])

        if optim:
            self.observation_sym = tf.placeholder(tf.float32, [None, 61, 120, frames_per_step])
            self.move_action_sym = tf.placeholder(tf.float32, [None, move_classes])
            self.reward_sym = tf.placeholder(tf.float32, [None])

            self.move_out_sym = tf.nn.softmax(self.eval(self.observation_sym), axis=1)
            self.zero_ops, self.accum_ops, self.train_step = train(optim, self.loss(criterion), scope, "move")

            dataset = tf.data.Dataset.from_tensor_slices((self.observation_sym,
                                                          self.move_action_sym,
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
        return tail(self.fc, out)

    def loss(self, criterion):
        move_out = self.eval(self.observation_sym)
        move_loss = criterion(logits=move_out, onehot_labels=self.move_action_sym, reduction=tf.losses.Reduction.NONE)
        move_loss += tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.reward_sym * move_loss)

    def train(self, sess, all_observations, all_move_actions, all_rewards):
        buffer_size = len(all_observations)
        batches = round(buffer_size / self.batch_size)

        sess.run(self.iterator.initializer, feed_dict={self.buffer_size: buffer_size,
                                                       self.observation_sym: all_observations,
                                                       self.move_action_sym: all_move_actions,
                                                       self.reward_sym: all_rewards})
        sess.run(self.zero_ops)
        for i in range(batches):
            observation, move_actions, rewards = sess.run(self.next_batch)
            sess.run(self.accum_ops, feed_dict={
                self.observation_sym: observation,
                self.move_action_sym: move_actions,
                self.reward_sym: rewards
            })
        sess.run(self.train_step)
