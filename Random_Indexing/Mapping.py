import numpy as np

__author__ = "Ariel Ekgren, https://github.com/ekgren/"


class Mapping(object):
    """
    Map object that maps arbitrary data to integer in the order provided
    from 0 to length of input.
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
        Yeah.
        """
        #print sequence
        mapped_sequence = np.zeros(len(sequence))

        for i, item in enumerate(sequence):
            #print item
            #print self.map[item]
            if item in self.map:
                #print "test"
                mapped_sequence[i] = self.map[item]
            else:
                mapped_sequence[i] = np.nan

        return mapped_sequence

    def nearest_neighbours(self, target, n=5):
        """
        """
        if self.word_space != None:
            if target in self.map and self.map[target] < self.word_space\
                    .size:
                neighb_args, neighb_vals = self.word_space.nearest_neighbours(
                    self.map[target], n)
                return target, self.inverse_map[neighb_args]
            else:
                return "No target!"