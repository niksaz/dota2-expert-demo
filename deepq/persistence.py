# Author: Mikita Sazanovich

import os
import tensorflow as tf


def get_last_episode(rewards_dir):
    last_episode = 0
    for reward_file in os.listdir(rewards_dir):
        reward_path = os.path.join(rewards_dir, reward_file)
        last_episode += len(list(tf.train.summary_iterator(reward_path)))
    return last_episode
