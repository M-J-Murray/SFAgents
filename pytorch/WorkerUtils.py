import torch
import torch.utils.data as td
import matplotlib.pyplot as plt
from torch.autograd import Variable


# pre-processes the frames returned by the game, so that they are suitable for the network
def prepro(frames):
    x = []
    for frame in frames:
        frame = frame[32:214, 12:372]  # crop
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]  # greyscale
        frame = frame[::3, ::3]  # downsample
        frame = frame / 255
        frame = frame - frame.mean()
        x.append(torch.cuda.FloatTensor(frame.reshape(1, 61, 120)))
    return torch.stack(x, dim=1)


# Randomly selects an action from the supplied distribution f
def chooseAction(f):
    th = torch.cuda.FloatTensor(1).uniform_()
    runSum = torch.cuda.FloatTensor(1).fill_(0)
    for i in range(f.size(1)):
        runSum += f.data[0, i]
        if th[0] < runSum[0]:
            break
    return i


# Processes the supplied rewards (r)
# Spreads the rewards so that every time step had a reward
# Uses the gamme to decay a running rewards backwards across the supplied rewards vector
def discount_rewards(r, gamma=0.92):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size(0))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


# A function to help with debugging
# Creates a nice graph using matplot lib
# Compares the rewards for each player before and after they have been processed
def plotRewards(p1History, p2History, gamma):
    moveActions = torch.cat(p1History["moveAction"])
    plt.scatter(range(moveActions.size(0)), moveActions.numpy())

    p1Rewards = torch.cat(p1History["reward"])
    p2Rewards = torch.cat(p2History["reward"])
    plt.plot(p1Rewards.numpy())
    plt.plot(p2Rewards.numpy())

    plt.plot(discount_rewards(p1Rewards, gamma=gamma).numpy())
    plt.plot(discount_rewards(p2Rewards, gamma=gamma).numpy())

    plt.show()


# Converts the history of gameplay from a list of timesteps into the pytorch Tensor
# Returns a dataset object which is used for randomly sampling mini-batches of history for training
def compileHistory(observations, history):
    observations = torch.cat(observations)

    moveActions = torch.cat(history["moveAction"])
    attackActions = torch.cat(history["attackAction"])

    rewards = torch.cat(history["reward"])
    rewards = discount_rewards(rewards)
    rewards = (rewards - rewards.mean()) / rewards.std()

    return td.TensorDataset(observations, torch.stack((moveActions, attackActions, rewards), dim=1))


# The same as the "compileHistory" function but for multiple rounds
# The processing of rewards should be done in seperate batches then put together, otherwise rewards from one round could be assigned to a different round.
def compileHistories(observations, histories):
    all_obs = []
    all_moveActions = []
    all_attackActions = []
    all_rewards = []
    for i in range(len(observations)):
        all_obs.append(torch.cat(observations[i]))

        all_moveActions.append(torch.cat(histories[i]["moveAction"]))
        all_attackActions.append(torch.cat(histories[i]["attackAction"]))

        rewards = torch.cat(histories[i]["reward"])
        rewards = discount_rewards(rewards)
        all_rewards.append((rewards - rewards.mean()) / rewards.std())

    return td.TensorDataset(torch.cat(all_obs), torch.stack((torch.cat(all_moveActions), torch.cat(all_attackActions), torch.cat(all_rewards)), dim=1))


# trains a model using the training dataset by randomly sub-sampling batches based on the batch_size.
# Note how the gradient is kept from every batch and then used to adust the network weights 
def train(model, optim, criterion, dataset, batch_size=128):
    N = len(dataset)
    data_sampler = td.sampler.RandomSampler(range(N))
    dataloader = td.DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)

    optim.zero_grad()  # Resets the models gradients
    for i, (x, t) in enumerate(dataloader):
        observation = Variable(x.cuda())
        moveOut, attackOut = model(observation)

        moveActions = Variable(t[:, 0].type(torch.cuda.LongTensor))
        attackActions = Variable(t[:, 1].type(torch.cuda.LongTensor))
        rewards = Variable(t[:, 2].cuda())

        # Calculates the loss for the movement outputs and the attack outputs
        loss = torch.sum(rewards * (criterion(moveOut, moveActions) + criterion(attackOut, attackActions)))
        loss.backward()

    optim.step()  # Updates the network weights based on the calculated gradients