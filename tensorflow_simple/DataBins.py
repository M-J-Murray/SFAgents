from pathlib import Path
import numpy as np
import os


def delete_old_bins(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(e)


def generate_bins(path, worker_count):
    workers_bins = []
    for i in range(worker_count):
        full_path = str(Path(path + "/worker%d.bin" % i).absolute())
        bin_file = os.open(full_path, os.O_CREAT | os.O_RDWR)
        workers_bins.append({"file": bin_file, "size": 0})
    return workers_bins


class DataBins(object):

    def __init__(self, path, worker_count, input_metadata=((1, 61, 120, 3), "float32"), move_metadata=((1, 9), "uint8"), attack_metadata=((1, 10), "uint8"), reward_metadata=((1,), "float64")):
        delete_old_bins(path)
        self.path = path
        self.workers_bins = generate_bins(path, worker_count)
        self.input_metadata = input_metadata
        self.move_metadata = move_metadata
        self.attack_metadata = attack_metadata
        self.reward_metadata = reward_metadata
        self.input_bytes = len(np.zeros(*input_metadata).tobytes())
        self.move_bytes = len(np.zeros(*move_metadata).tobytes())
        self.attack_bytes = len(np.zeros(*attack_metadata).tobytes())
        self.reward_bytes = len(np.zeros(*reward_metadata).tobytes())
        self.line_bytes = self.input_bytes + self.move_bytes + self.attack_bytes + self.reward_bytes

    def insert(self, worker_no, observation, move_action, attack_action, reward):
        data = observation.tobytes()
        if len(data) != self.input_bytes:
            raise IOError("Invalid observation inserted")
        data += move_action.tobytes()
        if len(data) != self.input_bytes+self.move_bytes:
            raise IOError("Invalid move action inserted")
        data += attack_action.tobytes()
        if len(data) != self.input_bytes + self.move_bytes + self.attack_bytes:
            raise IOError("Invalid attack action inserted")
        data += reward.tobytes()
        if len(data) != self.input_bytes + self.move_bytes + self.attack_bytes + self.reward_bytes:
            raise IOError("Invalid reward inserted")

        data_bin = self.workers_bins[worker_no]
        os.write(data_bin["file"], data)
        data_bin["size"] += 1

    def parse_data(self, data_bytes):
        observation = np.frombuffer(data_bytes[0:self.input_bytes], dtype=self.input_metadata[1]).reshape(self.input_metadata[0])
        move_action = np.frombuffer(data_bytes[self.input_bytes:self.input_bytes + self.move_bytes], dtype=self.move_metadata[1]).reshape(self.move_metadata[0])
        attack_action = np.frombuffer(data_bytes[self.input_bytes + self.move_bytes:self.input_bytes + self.move_bytes + self.attack_bytes], dtype=self.attack_metadata[1]).reshape(self.attack_metadata[0])
        reward = np.frombuffer(data_bytes[self.input_bytes + self.move_bytes + self.attack_bytes:], dtype=self.reward_metadata[1]).reshape(self.reward_metadata[0])
        return observation, move_action, attack_action, reward

    def empty_bin(self, worker_no):
        data_bin = self.workers_bins[worker_no]
        offset = 0
        os.lseek(data_bin["file"], offset, 0)

        all_observations = np.zeros(shape=[data_bin["size"], *self.input_metadata[0][1:]], dtype=self.input_metadata[1])
        all_move_actions = np.zeros(shape=[data_bin["size"], *self.move_metadata[0][1:]], dtype=self.move_metadata[1])
        all_attack_actions = np.zeros(shape=[data_bin["size"], *self.attack_metadata[0][1:]], dtype=self.attack_metadata[1])
        all_rewards = np.zeros(shape=[data_bin["size"]], dtype=self.reward_metadata[1])

        for i in range(data_bin["size"]):
            data_bytes = os.pread(data_bin["file"], self.line_bytes, offset)
            observation, move_action, attack_action, reward = self.parse_data(data_bytes)
            all_observations[i] = observation
            all_move_actions[i] = move_action
            all_attack_actions[i] = attack_action
            all_rewards[i] = reward

            offset += self.line_bytes

        data_bin["size"] = 0
        os.ftruncate(data_bin["file"], 0)
        return all_observations, all_move_actions, all_attack_actions, all_rewards

    def close(self):
        for data_bin in self.workers_bins:
            os.close(data_bin["file"])