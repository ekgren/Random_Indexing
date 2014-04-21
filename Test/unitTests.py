import unittest
import Random_Indexing

import numpy as np

__author__ = "Ariel Ekgren, https://github.com/ekgren/"

class TestVectorMath(unittest.TestCase):

    def setUp(self):
        self.vector1 = np.array([1,2,3,4])
        self.vector2 = np.array([3,2,1,0])

    def test_cosine(self):
        self.assertEqual(Random_Indexing.VectorMath.cosine(np.zeros(4),
                                                           self.vector1),
                         np.inf)

if __name__ == '__main__':
    unittest.main()