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
        move_bin_file = os.open(str(Path(path + "/worker%d_move.bin" % i).absolute()), os.O_CREAT | os.O_RDWR)
        attack_bin_file = os.open(str(Path(path + "/worker%d_attack.bin" % i).absolute()), os.O_CREAT | os.O_RDWR)
        move_attack_bin_file = os.open(str(Path(path + "/worker%d_move_attack.bin" % i).absolute()), os.O_CREAT | os.O_RDWR)
        workers_bins.append({"move": {"file": move_bin_file, "size": 0},
                             "attack": {"file": attack_bin_file, "size": 0},
                             "move_attack": {"file": move_attack_bin_file, "size": 0}})
    return workers_bins


class DataBins(object):

    def __init__(self, path, worker_count, frames_per_step):

        delete_old_bins(path)
        self.path = path
        self.workers_bins = generate_bins(path, worker_count)

        self.input_metadata = ((1, 61, 120, frames_per_step*3), "int8")
        self.move_metadata = ((1, 8), "uint8")
        self.attack_metadata = ((1, 9), "uint8")
        self.mode_metadata = ((1, 3), "uint8")

        self.input_bytes = len(np.zeros(*self.input_metadata).tobytes())
        self.move_bytes = len(np.zeros(*self.move_metadata).tobytes())
        self.attack_bytes = len(np.zeros(*self.attack_metadata).tobytes())

    def insert_move_bin(self, worker_no, observation, move_action):
        observation_bytes = observation.tobytes()
        move_action_bytes = move_action.tobytes()

        assert (len(observation_bytes) == self.input_bytes), "Invalid observation inserted"
        assert (len(move_action_bytes) == self.move_bytes), "Invalid move action inserted"

        data_bin = self.workers_bins[worker_no]["move"]
        os.write(data_bin["file"], observation_bytes+move_action_bytes)
        data_bin["size"] += 1

    def insert_attack_bin(self, worker_no, observation, attack_action):
        observation_bytes = observation.tobytes()
        attack_action_bytes = attack_action.tobytes()

        assert (len(observation_bytes) == self.input_bytes), "Invalid observation inserted"
        assert (len(attack_action_bytes) == self.attack_bytes), "Invalid attack action inserted"

        data_bin = self.workers_bins[worker_no]["attack"]
        os.write(data_bin["file"], observation_bytes+attack_action_bytes)
        data_bin["size"] += 1

    def insert_move_attack_bin(self, worker_no, observation, move_action, attack_action):
        observation_bytes = observation.tobytes()
        move_action_bytes = move_action.tobytes()
        attack_action_bytes = attack_action.tobytes()

        assert (len(observation_bytes) == self.input_bytes), "Invalid observation inserted"
        assert (len(move_action_bytes) == self.move_bytes), "Invalid move action inserted"
        assert (len(attack_action_bytes) == self.attack_bytes), "Invalid attack action inserted"

        data_bin = self.workers_bins[worker_no]["move_attack"]
        os.write(data_bin["file"], observation_bytes+move_action_bytes+attack_action_bytes)
        data_bin["size"] += 1

    def parse_move_data(self, data_bytes):
        observation = np.frombuffer(data_bytes[0:self.input_bytes], dtype=self.input_metadata[1]).reshape(self.input_metadata[0])
        move_action = np.frombuffer(data_bytes[self.input_bytes:], dtype=self.move_metadata[1]).reshape(self.move_metadata[0])
        return observation, move_action

    def parse_attack_data(self, data_bytes):
        observation = np.frombuffer(data_bytes[0:self.input_bytes], dtype=self.input_metadata[1]).reshape(self.input_metadata[0])
        attack_action = np.frombuffer(data_bytes[self.input_bytes:], dtype=self.attack_metadata[1]).reshape(self.attack_metadata[0])
        return observation, attack_action

    def parse_move_attack_data(self, data_bytes):
        observation = np.frombuffer(data_bytes[0:self.input_bytes], dtype=self.input_metadata[1]).reshape(self.input_metadata[0])
        move_action = np.frombuffer(data_bytes[self.input_bytes:self.input_bytes + self.move_bytes], dtype=self.move_metadata[1]).reshape(self.move_metadata[0])
        attack_action = np.frombuffer(data_bytes[self.input_bytes + self.move_bytes:], dtype=self.attack_metadata[1]).reshape(self.attack_metadata[0])
        return observation, move_action, attack_action

    def empty_move_bin(self, worker_no):
        data_bin = self.workers_bins[worker_no]["move"]
        offset = 0
        os.lseek(data_bin["file"], offset, 0)

        all_observations = np.zeros(shape=[data_bin["size"], *self.input_metadata[0][1:]], dtype=self.input_metadata[1])
        all_move_actions = np.zeros(shape=[data_bin["size"], *self.move_metadata[0][1:]], dtype=self.move_metadata[1])
        all_modes = np.zeros(shape=[data_bin["size"], *self.mode_metadata[0][1:]], dtype=self.mode_metadata[1])

        line_bytes = self.input_bytes + self.move_bytes

        for i in range(data_bin["size"]):
            data_bytes = os.pread(data_bin["file"], line_bytes, offset)
            observation, move_action = self.parse_move_data(data_bytes)
            all_observations[i] = observation
            all_move_actions[i] = move_action
            all_modes[i][0] = 1

            offset += line_bytes

        data_bin["size"] = 0
        os.ftruncate(data_bin["file"], 0)
        return all_observations, all_move_actions, all_modes

    def empty_attack_bin(self, worker_no):
        data_bin = self.workers_bins[worker_no]["attack"]
        offset = 0
        os.lseek(data_bin["file"], offset, 0)

        all_observations = np.zeros(shape=[data_bin["size"], *self.input_metadata[0][1:]], dtype=self.input_metadata[1])
        all_attack_actions = np.zeros(shape=[data_bin["size"], *self.attack_metadata[0][1:]], dtype=self.attack_metadata[1])
        all_modes = np.zeros(shape=[data_bin["size"], *self.mode_metadata[0][1:]], dtype=self.mode_metadata[1])

        line_bytes = self.input_bytes + self.attack_bytes

        for i in range(data_bin["size"]):
            data_bytes = os.pread(data_bin["file"], line_bytes, offset)
            observation, attack_action = self.parse_attack_data(data_bytes)
            all_observations[i] = observation
            all_attack_actions[i] = attack_action
            all_modes[i][1] = 1

            offset += line_bytes

        data_bin["size"] = 0
        os.ftruncate(data_bin["file"], 0)
        return all_observations, all_attack_actions, all_modes

    def empty_move_attack_bin(self, worker_no):
        data_bin = self.workers_bins[worker_no]["move_attack"]
        offset = 0
        os.lseek(data_bin["file"], offset, 0)

        all_observations = np.zeros(shape=[data_bin["size"], *self.input_metadata[0][1:]], dtype=self.input_metadata[1])
        all_move_actions = np.zeros(shape=[data_bin["size"], *self.move_metadata[0][1:]], dtype=self.move_metadata[1])
        all_attack_actions = np.zeros(shape=[data_bin["size"], *self.attack_metadata[0][1:]], dtype=self.attack_metadata[1])
        all_modes = np.zeros(shape=[data_bin["size"], *self.mode_metadata[0][1:]], dtype=self.mode_metadata[1])

        line_bytes = self.input_bytes + self.move_bytes + self.attack_bytes

        for i in range(data_bin["size"]):
            data_bytes = os.pread(data_bin["file"], line_bytes, offset)
            observation, move_action, attack_action = self.parse_move_attack_data(data_bytes)
            all_observations[i] = observation
            all_move_actions[i] = move_action
            all_attack_actions[i] = attack_action
            all_modes[i][2] = 1

            offset += line_bytes

        data_bin["size"] = 0
        os.ftruncate(data_bin["file"], 0)
        return all_observations, all_move_actions, all_attack_actions, all_modes

    def close(self):
        for worker_bins in self.workers_bins:
            for data_bin in worker_bins.values():
                os.close(data_bin["file"])