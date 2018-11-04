from multiprocessing import Value, Lock
import logging


class Statistics(object):

    def __init__(self, episode):
        self.episode = Value("i", episode)
        self.lock = Lock()
        self.has_run = Value("i", 0)
        self.best = {"reward": Value("d", 0), "stage": Value("d", 0)}
        self.running = {"reward": Value("d", 0), "stage": Value("d", 0)}

    def update(self, reward):
        stats_string = "episode {:5.0f} complete".format(self.episode.value)
        with self.lock:
            for k in reward.keys():
                if self.has_run.value == 0:
                    self.best[k].value = reward[k]
                elif reward[k] > self.best[k].value:
                    self.best[k].value = reward[k]

                self.running[k].value = reward[k] if self.has_run.value == 0 else self.running[k].value * 0.99 + reward[k] * 0.01

                stats_string += " - " + k + "(new:{:5.0f}, avg:{:5.0f}, best:{:5.0f})".format(reward[k], self.running[k].value, self.best[k].value)
            self.has_run.value = 1
        logging.info(stats_string)
