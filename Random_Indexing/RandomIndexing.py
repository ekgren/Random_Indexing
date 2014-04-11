__author__ = 'Ariel Ekgren, https://github.com/ekgren/'

import random
from collections import deque

import Math

import numpy as np


# noinspection PyBroadException
class RandomIndexing(object):
    """
    Random Indexing
    """

    def __init__(self, window_size=2):
        """
        Constructor
        """

        self.dimensionality = 2000
        self.random_degree = 20

        self.random_seed = 1

        self.window_start_position = 1
        self.window_size = window_size
        self.window = range(self.window_start_position,
                            self.window_start_position + self.window_size)

        self.sequence_queues = [deque(maxlen=i + 1) for i in self.window]

        self.size = 1000

        self.base_vectors = None
        self.context_vectors = None

        self.mapping = None
        self.stop = None

    @staticmethod
    def create_base_vectors(size, dimensionality, random_degree):
        """
        Creates and returns a matrix of base vectors as a numpy int8 ndarray.
        :rtype : ndarray
        :param size: int
        :param dimensionality: int
        :param random_degree: int
        """
        base_vectors = np.zeros((size, dimensionality),
                                dtype=np.int8)

        for i in xrange(size):
            base_vectors[i] = random_vector(i, dimensionality,
                                            random_degree)

        return base_vectors

    @staticmethod
    def create_context_vectors(size, dimensionality, dtype=np.int32):
        """
        Create and returns a matrix of context vectors as a numpy array of
        specified data type.
        :param size:
        :param dimensionality:
        :param dtype:
        :rtype : ndarray
        """
        context_vectors = np.zeros((size, dimensionality), dtype=dtype)
        return context_vectors

    def create_regular_word_space(self):
        """
        Creates base and context vectors for the Random Indexing object.
        """
        self.base_vectors = self.create_base_vectors(self.size,
                                                     self.dimensionality,
                                                     self.random_degree)
        self.context_vectors = self.create_context_vectors(self.size,
                                                           self.dimensionality,
                                                           np.int32)

    def update_context_vectors(self, sequence):
        """
        """
        for i, window in enumerate(self.window):
            # Create sequence queue.
            que = self.sequence_queues[i]
            for item in sequence:
                que.append(item)
                try:
                    self.context_vectors[que[window]] += self.base_vectors[
                        que[0]]
                except:
                    pass
                try:
                    self.context_vectors[que[0]] += self.base_vectors[que[
                        window]]
                except:
                    pass
            que.clear()

    def nearest_neighbours(self, target, n=5):
        d = np.zeros(self.size, dtype=np.float)
        for i, vector in enumerate(self.context_vectors):
            d[i] = Math.cosine(self.context_vectors[target], vector)
        d[d.argmin()] = np.inf

        neighbs = []

        for x in range(n):
            neighbs.append((d.argmin(),d[d.argmin()]))
            d[d.argmin()] = np.inf

        return neighbs



def random_vector(random_seed, dimensionality, random_degree):
    """
    Creates and return a numpy int8 Random Index vector.
    :param random_seed:
    :param dimensionality:
    :param random_degree:
    :rtype ndarray:
    """
    random.seed(random_seed)

    vector = np.zeros(dimensionality, dtype=np.int8)

    for i in range(random_degree):
        if i % 2 == 0:
            vector[random.randint(0, dimensionality - 1)] = -1
        else:
            vector[random.randint(0, dimensionality - 1)] = 1

    return vector


if __name__ == '__main__':
    pass
