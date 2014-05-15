import numpy as np

__author__ = "Ariel Ekgren, https://github.com/ekgren/"


class Mapping(object):
    """
    Map object that maps arbitrary data onto integers.
    F: Object -> Integer
    """

    def __init__(self):
        self.map = {}
        self.inverse_map = None
        self.word_space = None

    def create_map(self, data):
        """
        """
        for i, point in enumerate(data):
            self.map[point] = i

        self.create_inverse_map()

    def create_inverse_map(self):
        """
        """
        self.inverse_map = np.zeros(len(self.map), dtype=np.dtype)

        for x in self.map.iteritems():
            self.inverse_map[x[1]] = x[0]

    def map_sequence(self, sequence):
        """
        Transform a sequence of objects into a sequence of integers.
        """
        mapped_sequence = np.zeros(len(sequence), dtype=np.int32)

        for i, item in enumerate(sequence):
            if item in self.map:
                mapped_sequence[i] = self.map[item]
            else:
                mapped_sequence[i] = -2147483648

        return mapped_sequence

    def nn_context_context(self, target, n=5):
        """
        Nearest neighbour search.
        Distance from context vector to other context vectors.
        """
        if self.word_space != None:
            if target in self.map and self.map[target] < self.word_space\
                    .size:
                neighb_args, neighb_vals = self.word_space.nn_context_context(
                    self.map[target], n)
                return target, self.inverse_map[neighb_args]
            else:
                return "No target!"

    def nn_context_base(self, target, n=5):
        """
        Nearest neighbour search.
        Distance from context vector to base vectors.
        """
        if self.word_space != None:
            if target in self.map and self.map[target] < self.word_space\
                    .size:
                neighb_args, neighb_vals = self.word_space\
                    .nn_context_base(
                    self.map[target], n)
                return target, self.inverse_map[neighb_args]
            else:
                return "No target!"

    def nn_base_base(self, target, n=5):
        """
        Nearest neighbour seach.
        Distance from base vector to other base vectors.
        """
        if self.word_space != None:
            if target in self.map and self.map[target] < self.word_space\
                    .size:
                neighb_args, neighb_vals = self.word_space\
                    .nn_base_base(
                    self.map[target], n)
                return target, self.inverse_map[neighb_args]
            else:
                return "No target!"