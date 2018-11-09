import numpy as np


# pre-processes the frames returned by the game, so that they are suitable for the network
def prepro(frames, frames_per_step):
    assert len(frames) == frames_per_step, "Too many frames passed for preprocessing"
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
    discounted_r = np.zeros_like(r, dtype="float32")
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def compile_rewards(rewards):
    discounted_rewards = [[], [], []]
    for round_history in rewards:
        discounted_round = discount_rewards(round_history["reward"])
        for i in range(len(discounted_round)):
            discounted_rewards[round_history["mode"][i]].append(discounted_round[i])
    return [np.stack(discounted_rewards[0]), np.stack(discounted_rewards[1]), np.stack(discounted_rewards[2])]


# def store_history(data_bins, worker_no, history):
#     rewards = discount_rewards(history["reward"])
#     rewards = (rewards - rewards.mean()) / rewards.std()
#     for i in range(len(history["observation"])):
#         mode = np.argmax(history["mode"][i])
#         if mode == 0:
#             data_bins.insert_move_bin(worker_no, history["observation"][i], history["move_action"][i], rewards[i])
#         elif mode == 1:
#             data_bins.insert_attack_bin(worker_no, history["observation"][i], history["attack_action"][i], rewards[i])
#         elif mode == 2:
#             data_bins.insert_move_attack_bin(worker_no, history["observation"][i], history["move_action"][i], history["attack_action"][i], rewards[i])
