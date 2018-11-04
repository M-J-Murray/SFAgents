from tensoflow_complex.Worker import run
from tensoflow_complex.DataBins import DataBins
from tensoflow_complex.Statistics import Statistics
from multiprocessing import Process
import tensorflow as tf
import logging

logging.basicConfig(filename='logs/tensor_stats.log', level=logging.INFO)


def run_ps(cluster):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)
    server.join()


def train(roms_path, learning_rate, worker_count, frames_per_step, episode, save_frequency, saves_path):
    stats = Statistics(episode)
    data_bins = DataBins("databins", worker_count, frames_per_step)

    tasks = ["localhost:" + str(2223 + i) for i in range(worker_count)]
    jobs = {"ps": ["localhost:2222"], "worker": tasks}
    cluster = tf.train.ClusterSpec(jobs)

    ps_proc = Process(target=run_ps, args=[cluster])
    worker_procs = [Process(target=run, args=[i, roms_path, learning_rate, frames_per_step, cluster, data_bins, stats, save_frequency, saves_path]) for i in range(worker_count)]

    ps_proc.start()
    [proc.start() for proc in worker_procs]

    ps_proc.join()
    [proc.join() for proc in worker_procs]


# spawn must be called inside main
if __name__ == '__main__':
    train(roms_path="../roms",
          learning_rate=5e-5,
          worker_count=4,
          frames_per_step=3,
          episode=0,
          save_frequency=100,
          saves_path="saves")
