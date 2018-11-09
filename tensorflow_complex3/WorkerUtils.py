import numpy as np


# pre-processes the frames returned by the game, so that they are suitable for the network
def prepro(frames, frames_per_step):
    assert len(frames) == frames_per_step, "Too many frames passed for preprocessing"
    x = []
    for frame in frames: 
        frame = frame[32:214, 12:372]  # crop
        frame = frame[::3, ::3]  # downsample
        frame = frame.astype("int16")-128
        x.append(frame.reshape(1, 61, 120, 3))
    return np.concatenate(x, axis=3).astype("int8")


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
def discount_rewards(r, amount=1):
    discounted_r = np.zeros_like(r, dtype="int8")
    for t in reversed(range(len(r))):
        if r[t] != 0:
            val = amount if r[t] > 0 else -amount
            for i in range(25 if t > 24 else t):
                discounted_r[t-i] += val

    return discounted_r


def compile_rewards(rewards):
    discounted_rewards = [[], [], []]
    for round_history in rewards:
        discounted_round = discount_rewards(round_history["reward"])
        for i in range(len(discounted_round)):
            discounted_rewards[round_history["mode"][i]].append(discounted_round[i])
    return np.stack(discounted_rewards[0]) if len(discounted_rewards[0]) > 0 else None, \
           np.stack(discounted_rewards[1]) if len(discounted_rewards[1]) > 0 else None, \
           np.stack(discounted_rewards[2]) if len(discounted_rewards[2]) > 0 else None
