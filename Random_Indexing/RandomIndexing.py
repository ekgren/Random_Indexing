import random
from collections import deque

import numpy as np

import VectorMath

__author__ = "Ariel Ekgren, https://github.com/ekgren/"


# noinspection PyBroadException
class RandomIndexing(object):
    """
    Class that creates a Random Indexing word space.

    The class contains a numpy int8 matrix of base vectors and a numpy
    int32 matrix of context vectors.
    """

    def __init__(self, window_size=2):
        """
        Initiates the Random Indexing object.

        :param window_size: int
        """

        self.dimensionality = 2000
        self.random_degree = 20

        self.random_seed = 1

        self.window_start_position = 1
        self.window_size = window_size
        self.window = range(self.window_start_position,
                            self.window_start_position + self.window_size)

        self.test_window = range(-self.window_size, self.window_size+1)
        self.test_window.remove(0)
        self.test_window = np.array(self.test_window)

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

        :rtype : ndarray
        :param size: int
        :param dimensionality: int
        :param dtype: numpy data type
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

    def create_special_word_space(self):
        """
        Creates base and context vectors for the Random Indexing object.
        """
        self.base_vectors = self.create_base_vectors(self.size,
                                                     self.dimensionality,
                                                     self.random_degree)
        self.context_vectors = self.create_context_vectors(self.size,
                                                           self.dimensionality,
                                                           np.float)

    def update_context_vectors(self, sequence):
        """
        Incrementally update the context matrix.

        :param sequence: list, tuple or ndarray of integers
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

    def experimental_update(self, sequence):
        """
        Incrementally update the context matrix.

        :param sequence: list, tuple or ndarray of integers
        """
        for i, window in enumerate(self.window):
            # Create sequence queue.
            que = self.sequence_queues[i]
            for item in sequence:
                que.append(item)
                try:
                    self.context_vectors[que[window]] += np.array(self.base_vectors[
                        que[0]], dtype=np.float)/(1 + np.sum(np.abs(self.context_vectors[que[0]])))
                except:
                    pass
                try:
                    self.context_vectors[que[0]] += np.array(self.base_vectors[que[
                        window]], dtype=np.float)/(1 + np.sum(np.abs(self.context_vectors[que[window]])))
                except:
                    pass
            que.clear()

    def update_base_vectors(self, sequence):
        """
        Incrementally update the context matrix.

        :param sequence: list, tuple or ndarray of integers
        """
        for i, window in enumerate(self.window):
            # Create sequence queue.
            que = self.sequence_queues[i]
            for item in sequence:
                que.append(item)
                try:
                    self.base_vectors[que[window]] += self.base_vectors[
                        que[0]]
                except:
                    pass
                try:
                    self.base_vectors[que[0]] += self.base_vectors[que[
                        window]]
                except:
                    pass
            que.clear()

    def nn_context_context(self, target, n=5):
        """
        Find and return the n nearest neighbours to a target vector from
        the context matrix.

        :rtype : 2-tuple of ndarrays
        :param target: int
        :param n: int
        """
        d = np.zeros(self.size, dtype=np.float)

        for i, vector in enumerate(self.context_vectors):
            d[i] = VectorMath.cosine(self.context_vectors[target], vector)

        args = np.argsort(d)[1:n + 1]

        vals = d[args]

        return args, vals

    def nn_context_base(self, target, n=5):
        """
        Find and return the n nearest base neighbours to a target vector from
        the context matrix.

        :rtype : 2-tuple of ndarrays
        :param target: int
        :param n: int
        """
        d = np.zeros(self.size, dtype=np.float)

        for i, vector in enumerate(self.base_vectors):
            d[i] = VectorMath.cosine(self.context_vectors[target], vector)

        args = np.argsort(d)[1:n + 1]

        vals = d[args]

        return args, vals

    def nn_base_context(self, target, n=5):
        """
        Find and return the n nearest base neighbours to a target vector from
        the context matrix.

        :rtype : 2-tuple of ndarrays
        :param target: int
        :param n: int
        """
        d = np.zeros(self.size, dtype=np.float)

        for i, vector in enumerate(self.context_vectors):
            d[i] = VectorMath.cosine(self.base_vectors[target], vector)

        args = np.argsort(d)[1:n + 1]

        vals = d[args]

        return args, vals

    def nn_base_base(self, target, n=5):
        """
        Find and return the n nearest base neighbours to a target vector from
        the base matrix.

        :rtype : 2-tuple of ndarrays
        :param target: int
        :param n: int
        """
        d = np.zeros(self.size, dtype=np.float)

        for i, vector in enumerate(self.base_vectors):
            d[i] = VectorMath.cosine(self.base_vectors[target], vector)

        args = np.argsort(d)[1:n + 1]

        vals = d[args]

        return args, vals

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