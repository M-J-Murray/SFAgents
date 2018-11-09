from multiprocessing import Value, Lock
import logging


class Statistics(object):

    def __init__(self, episode):
        self.lock = Lock()
        self.episode = Value("i", episode)
        self.has_run = Value("i", 0)
        self.best_score = Value("d", 0)
        self.running_score = Value("d", 0)
        self.best_stage = Value("d", 0)
        self.running_stage = Value("d", 0)

    def update(self, reward):
        with self.lock:
            self.episode.value += 1

            if self.has_run.value == 0:
                self.best_score.value = reward["score"]
            elif reward["score"] > self.best_score.value:
                self.best_score.value = reward["score"]
            self.running_score.value = reward["score"] if self.has_run.value == 0 else self.running_score.value * 0.99 + reward["score"] * 0.01

            if self.has_run.value == 0:
                self.best_stage.value = reward["stage"]
            elif reward["stage"] > self.best_stage.value:
                self.best_stage.value = reward["stage"]
            self.running_stage.value = reward["stage"] if self.has_run.value == 0 else self.running_stage.value * 0.99 + reward["stage"] * 0.01

            self.has_run.value = 1
            output = "Episode {:5.0f} Complete - " \
                     "Score(new:{:5.0f}, avg:{:5.1f}, best:{:5.0f}) - " \
                     "Stage(new:{:5.0f}, avg:{:5.1f}, best:{:5.0f})" \
                     .format(self.episode.value,
                             reward["score"], self.running_score.value, self.best_score.value,
                             reward["stage"], self.running_stage.value, self.best_stage.value)
            logging.info(output)
            print(output)

    def get_episode(self):
        with self.lock:
            return self.episode.value
