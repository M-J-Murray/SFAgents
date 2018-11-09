import random
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from MAMEToolkit.sf_environment import Environment


def prepro(frame, isGrey):
    frame = frame[32:214, 12:372]  # crop
    frame = frame[::3, ::3]
    if isGrey:
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
    return frame


isGrey = True

roms_path = "../roms/"  # Replace this with the path to your ROMs
env = Environment("env1", roms_path, frames_per_step=1, frame_ratio=3, throttle=False)

fig = plt.figure()
plt.ion()
im: AxesImage = plt.imshow(prepro(env.start(), isGrey), cmap="gray" if isGrey else None)
plt.axis("off")
plt.show()
while True:
    move_action = random.randint(0, 8)
    attack_action = random.randint(0, 9)
    frame, reward, round_done, stage_done, game_done = env.step(move_action, attack_action)
    im.set_data(prepro(frame, isGrey))
    plt.pause(0.0001)
    if game_done:
        env.new_game()
    elif stage_done:
        env.next_stage()
    elif round_done:
        env.next_round()