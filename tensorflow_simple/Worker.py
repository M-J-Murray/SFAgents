from MAMEToolkit.sf_environment import Environment
from tensorflow_simple.Model import Model
import tensorflow_simple.WorkerUtils as wu
import tensorflow as tf
import numpy as np
import traceback
import logging


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def run(worker_no, roms_path, learning_rate, cluster, data_bins, stats, saves_path, save_frequency):
    name = "worker%d" % worker_no

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % worker_no, cluster=cluster)):
        Model("global", learning_rate)

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global"))

    local_model = Model(name, learning_rate)

    update_local_ops = update_target_graph('global', name)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="worker", task_index=worker_no, config=config)
    with tf.train.MonitoredTrainingSession(master=server.target) as sess:
        try:
            if stats.get_episode() != 0:
                saver.restore(sess._sess._sess._sess._sess, f'{saves_path}/model.ckpt')

            sess.run(update_local_ops)
            print("Started Worker Updates")
            env = Environment(name, roms_path, difficulty=3)
            frames = env.start()

            while True:
                history = {"observation": [], "move_action": [], "attack_action": [], "reward": []}
                game_done = False
                total_reward = 0

                while not game_done:

                    observation = wu.prepro(frames)

                    history["observation"].append(observation)

                    move_out, attack_out = sess.run([local_model.move_out_sym, local_model.attack_out_sym], feed_dict={local_model.observation_sym: observation})

                    move_action_hot = wu.choose_action(move_out)
                    attack_action_hot = wu.choose_action(attack_out)

                    history["move_action"].append(move_action_hot)
                    history["attack_action"].append(attack_action_hot)

                    frames, r, round_done, stage_done, game_done = env.step(np.argmax(move_action_hot), np.argmax(attack_action_hot))
                    total_reward += r["P1"]

                    history["reward"].append(r["P1"])

                    if round_done:
                        wu.store_history(data_bins, worker_no, history)
                        history = {"observation": [], "move_action": [], "attack_action": [], "reward": []}
                        if game_done:
                            wu.train(sess, local_model, *data_bins.empty_bin(worker_no))
                            sess.run(update_local_ops)
                            stats.update({"score": total_reward, "stage": env.stage})
                            if stats.get_episode() > 0 and stats.get_episode() % save_frequency == 0:
                                saver.save(sess._sess._sess._sess._sess, f'{saves_path}/model.ckpt')
                        frames = env.reset()
        except:
            error = traceback.format_exc()
            print(error)
            logging.error(error)
            exit(1)
