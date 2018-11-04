import numpy as np


# pre-processes the frames returned by the game, so that they are suitable for the network
def prepro(frames):
    x = []
    for frame in frames: 
        frame = frame[32:214, 12:372]  # crop
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2] # greyscale
        frame = frame[::3, ::3] # downsample
        frame = frame/255
        frame = frame-frame.mean()
        x.append(frame.reshape(1, 61, 120))
    return np.stack(x, axis=3).astype("float32")


# Randomly selects an action from the supplied distribution f
def choose_action(f):
    hot = np.zeros_like(f, dtype="uint8")
    th = np.random.uniform(0, 1)
    run_sum = 0
    i = 0
    for i in range(f.size):
        run_sum += f[0, i]
        if th < run_sum:
            break
    hot[0, i] = 1
    return hot


# Processes the supplied rewards (r)
# Spreads the rewards so that every time step had a reward
# Uses the gamma to decay a running rewards backwards across the supplied rewards vector
def discount_rewards(r, gamma=0.92):
    discounted_r = np.zeros_like(r, dtype="float64")
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def store_history(data_bins, worker_no, history):
    rewards = discount_rewards(history["reward"])
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i in range(len(history["observation"])):
        data_bins.insert(worker_no, history["observation"][i], history["move_action"][i], history["attack_action"][i], rewards[i])


# def compileHistories(history):
#     all_observations = []
#     all_move_actions = []
#     all_attack_actions = []
#     all_rewards = []
#     for round_history in history:
#         all_observations.append(np.concatenate(round_history["observation"]))
#         all_move_actions.append(np.concatenate(round_history["move_action"]))
#         all_attack_actions.append(np.concatenate(round_history["attack_action"]))
#
#         rewards = np.stack(round_history["reward"])
#         rewards = discount_rewards(rewards)
#         all_rewards.append((rewards - rewards.mean())/rewards.std())
#
#     return np.concatenate(all_observations), \
#            np.concatenate(all_move_actions), \
#            np.concatenate(all_attack_actions), \
#            np.concatenate(all_rewards)


# trains a model using the training dataset by randomly sub-sampling batches based on the batch_size.
# Note how the gradient is kept from every batch and then used to adjust the network weights
def train(sess, model, all_observations, all_move_actions, all_attack_actions, all_rewards):
    buffer_size = len(all_observations)
    batches = round(buffer_size/model.batch_size)

    sess.run(model.iterator.initializer, feed_dict={model.buffer_size: buffer_size,
                                                    model.observation_sym: all_observations,
                                                    model.move_action_sym: all_move_actions,
                                                    model.attack_action_sym: all_attack_actions,
                                                    model.reward_sym: all_rewards})
    sess.run(model.zero_ops)
    for i in range(batches):
        observation, move_actions, attack_actions, rewards = sess.run(model.next_batch)
        sess.run(model.accum_ops, feed_dict={
            model.observation_sym: observation,
            model.move_action_sym: move_actions,
            model.attack_action_sym: attack_actions,
            model.reward_sym: rewards
        })
    sess.run(model.train_step)
