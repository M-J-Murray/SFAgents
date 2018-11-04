from tensorflow_simple.Worker import run
from tensorflow_simple.DataBins import DataBins
from tensorflow_simple.Statistics import Statistics
from multiprocessing import Process
import tensorflow as tf
import logging
logging.basicConfig(filename='logs/tensor_stats.log', level=logging.INFO)


def run_ps(cluster):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)
    server.join()


def train(roms_path, learning_rate, worker_count, episode, saves_path, save_frequency):
    stats = Statistics(episode)
    data_bins = DataBins("databins", worker_count)

    tasks = ["localhost:" + str(2223 + i) for i in range(worker_count)]
    jobs = {"ps": ["localhost:2222"], "worker": tasks}
    cluster = tf.train.ClusterSpec(jobs)

    ps_proc = Process(target=run_ps, args=[cluster])
    worker_procs = [Process(target=run, args=[i, roms_path, learning_rate, cluster, data_bins, stats, saves_path, save_frequency]) for i in range(worker_count)]

    ps_proc.start()
    [proc.start() for proc in worker_procs]

    ps_proc.join()
    [proc.join() for proc in worker_procs]


# spawn must be called inside main
if __name__ == '__main__':
    train(roms_path="../roms", learning_rate=5e-5, worker_count=2, episode=0, saves_path="saves", save_frequency=1)
