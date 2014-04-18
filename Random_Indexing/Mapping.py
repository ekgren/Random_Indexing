__author__ = 'Ariel Ekgren, https://github.com/ekgren/'

import numpy as np

class Mapping(object):
    """
    Map object that maps arbitrary data to integer in the order provided
    from 0 to length of input.
    """

    def __init__(self, map={}, inverse_map=None):
        self.map = map
        self.inverse_map = inverse_map

    def create_map(self, data):
        """
        """
        for i, point in enumerate(data):
            self.map[point] = i

    def create_inverse_map(self):
        """
        """
        self.inverse_map = np.zeros(len(self.map), dtype=np.dtype)

        for x in self.map.iteritems():
            self.inverse_map[x[0]] = x[1]