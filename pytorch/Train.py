from pytorch.Worker import Worker
from pytorch.Model import Model
from pytorch.Statistics import Statistics

import torch
import torch.optim as opt
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import logging

logging.basicConfig(filename='logs/pytorch_stats.log', level=logging.INFO)


# Creates the model used for training
def setupModel(learning_rate, episode, framesPerStep, loadPath):
    model = Model(framesPerStep, 9, 10)
    optim = opt.Adam(model.parameters(), lr=learning_rate)
    if episode > 0:  # For loading a saved model
        model.load_state_dict(torch.load(loadPath + "models/" + str(episode), map_location=lambda storage, loc: storage))
        optim.load_state_dict(torch.load(loadPath + "optims/" + str(episode)))
    model.cuda()  # Moves the network matrices to the GPU
    model.share_memory()  # For multiprocessing
    return model, optim


def setupWorkers(roms_path, difficulty, epoch_size, learning_rate, frameRatio, framesPerStep, episode, workerCount, loadPath):
    model, optim = setupModel(learning_rate, episode, framesPerStep, loadPath)
    criterion = nn.CrossEntropyLoss(reduce=False)

    rewardQ = Queue()

    workers = [Worker(f"worker{i}", roms_path, difficulty, epoch_size, model, optim, criterion, rewardQ, frameRatio, framesPerStep) for i in range(workerCount)]
    return workers, model, optim, rewardQ


# Check pytorch_stats.log for results
def simulate(episode, workers, model, optim, rewardQueue, batch_save, path):
    [w.start() for w in workers]

    stats = Statistics(episode)

    while True:
        episode += 1
        if episode % batch_save == 0:
            torch.save(model.state_dict(), path + "models/" + str(episode))
            torch.save(optim.state_dict(), path + "optims/" + str(episode))

        reward = rewardQueue.get()
        stats.update(reward)


# spawn must be called inside main
if __name__ == '__main__':
    mp.set_start_method('spawn')

    roms_path = "../roms"
    frameRatio = 2
    framesPerStep = 3
    learning_rate = 5e-5
    episode = 2200  # Change episode to load from presaved model, check saves for saves
    epoch_size = 1  # How many episodes before training
    batch_save = 100  # How many results before saving the current model and optimiser
    workerCount = 2  # How many workers to train the model in parallel
    difficulty = 3  # Story mode difficulty
    loadPath = "saves/"

    workers, model, optim, rewardQueue = setupWorkers(roms_path, difficulty, epoch_size, learning_rate, frameRatio, framesPerStep, episode, workerCount, loadPath)

    simulate(episode, workers, model, optim, rewardQueue, batch_save, loadPath)
