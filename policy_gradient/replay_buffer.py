import itertools
import os
import pickle
import random
from collections import deque
from random import shuffle


class ReplayBuffer:
    """
    Replay buffer for storing sampled data.
    """
    __slots__ = ('data', 'filename')

    def __init__(self, directory='./', max_size=1000000):
        """
        Create new buffer with given params.
        :param directory: directory to work in
        :param max_size: maximal size of a buffer
        """
        self.filename = os.path.join(directory, 'replay_buffer.pkl')
        self.data = deque(maxlen=max_size)

    def save_data(self):
        """
        Save buffer data to file.
        """
        with open(self.filename, 'wb') as output_file:
            pickle.dump(obj=self.data, file=output_file)

    def load_data(self):
        """
        Load buffer data from file.
        """
        with open(self.filename, 'rb') as input_file:
            self.data = pickle.load(file=input_file)

    def __len__(self):
        """
        :return: length of this buffer
        """
        return len(self.data)

    def append(self, element):
        """
        Add single element to buffer.
        :param element: element to add
        """
        self.data.append(element)

    def extend(self, elements):
        """
        Extend buffer with list of elements
        :param elements: list of elements
        """
        self.data.extend(elements)

    def get_data(self, batch_size):
        """
        Get randomly sampled batch of data from the buffer.
        :param batch_size: batch size
        :return: 3 iterators: states, actions, rewards
        """
        size = len(self.data)
        idx_data = set(random.sample(range(size), batch_size))
        data = [elem for i, elem in enumerate(self.data) if i in idx_data]

        return [s for s, a, r in data], \
               [a for s, a, r in data], \
               [r for s, a, r in data]

    def get_batch(self, i, batch_size):
        """
        Get batch of data.
        :param i: from
        :param batch_size: batch size
        :return: 3 iterators: states, actions, rewards
        """
        size = len(self.data)
        data = list(itertools.islice(self.data, i, min(batch_size + i, size)))
        return [s for s, a, r in data], \
               [a for s, a, r in data], \
               [r for s, a, r in data]

    def shuffle_data(self):
        shuffle(shuffle(self.data))
