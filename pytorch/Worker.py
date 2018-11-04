import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch.WorkerUtils as wu
from MAMEToolkit.sf_environment import Environment
import traceback
import logging

logging.basicConfig(filename='logs/pytorch_stats.log', level=logging.INFO)
logger = logging.getLogger(__name__)


# The worker class for running agent vs Computer training, aka story mode training
class Worker(mp.Process):

    def __init__(self, env_id, roms_path, difficulty, epoch_size, model, optim, criterion, rewardQueue, frameRatio, framesPerStep):
        super(Worker, self).__init__()
        self.env_id = env_id
        self.roms_path = roms_path
        self.difficulty = difficulty
        self.epoch_size = epoch_size
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.rewardQueue = rewardQueue
        self.frameRatio = frameRatio
        self.framesPerStep = framesPerStep

    def run(self):
        try:
            logger.info("Starting Worker")
            self.env = Environment(self.env_id, self.roms_path, difficulty=self.difficulty, frame_ratio=self.frameRatio, frames_per_step=self.framesPerStep, throttle=False)
            frames = self.env.start()
            while True:
                self.model.eval()

                observations, histories, frames = self.generate_playthrough(frames)

                self.model.train()

                dataset = wu.compileHistories(observations, histories)
                wu.train(self.model, self.optim, self.criterion, dataset)

        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())

    def generate_playthrough(self, frames):
        observations = [[]]
        histories = [{"moveAction": [], "attackAction": [], "reward": []}]
        epoch_reward = 0
        total_round = 0
        game_done = False

        for i in range(self.epoch_size):

            while not game_done:
                x = wu.prepro(frames)

                observations[total_round].append(x.cpu())

                moveOut, attackOut = self.model(Variable(x))
                moveAction = wu.chooseAction(F.softmax(moveOut, dim=1))
                attackAction = wu.chooseAction(F.softmax(attackOut, dim=1))

                histories[total_round]["moveAction"].append(torch.FloatTensor(1).fill_(moveAction))
                histories[total_round]["attackAction"].append(torch.FloatTensor(1).fill_(attackAction))

                frames, r, round_done, stage_done, game_done = self.env.step(moveAction, attackAction)

                histories[total_round]["reward"].append(torch.FloatTensor(1).fill_(r["P1"]))

                epoch_reward += r["P1"]

                if round_done:
                    total_round += 1
                    histories.append({"moveAction": [], "attackAction": [], "reward": []})
                    observations.append([])
                    if game_done:
                        self.rewardQueue.put({"reward": epoch_reward, "stage": self.env.stage})
                    frames = self.env.reset()

        return observations, histories, frames
