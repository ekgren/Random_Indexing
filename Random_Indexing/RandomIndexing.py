# !/usr/bin/python
# vim: set fileencoding=iso-8859-15 :
# Created on 15 jan 2014
# @author: Ariel

import codecs
import cPickle
import random
from collections import deque

import numpy as np

from TEXT import TEXT


# noinspection PyBroadException
class RandomIndexing(object):
    """
    Random Indexing
    """

    def __init__(self):
        """
        Constructor
        """

        self.dimensionality = 2000
        self.random_degree = 20

        self.seed_count = 1

        self.stop = 0

        self.window_start = 1
        self.window_size = 2


        self.size = min(len(self.mapping), 200000)

        # File path to corpus.

        #file_path = 'ukwac_plain_text/ukwac_preproc'
        self.file_path = 'ukwac_plain_text/ukwac_subset_10M.txt'

        # Instantiate text reader.
        self.t = TEXT()

    def load_dictionary(self, path):
        """
        Load vocabulary in form of a dict mapping a word to an int.
        The mapping is in order of frequency. Highest frequency words first.
        """
        pkl_file = open(path, 'rb')
        self.mapping = cPickle.load(pkl_file)
        pkl_file.close()

    def random_vector(self):
        """
        Creates and return a numpy int8 Random Indexing word vector.
        """

        random.seed(self.seed_count)

        vector = np.zeros(self.dimensionality, dtype=np.int8)

        for i in range(self.random_degree):
            if i % 2 == 0:
                vector[random.randint(0, self.dimensionality - 1)] = -1
            else:
                vector[random.randint(0, self.dimensionality - 1)] = 1

        return vector

    def create_word_vectors(self):
        """
        Creates and returns a matrix of base vectors as a numpy int8 ndarray.
        """
        word_vectors = np.zeros((self.size, self.dimensionality),
                                dtype=np.int8)

        for word in self.mapping:
            try:
                word_vectors[self.mapping[word]] = self.random_vector()
                self.seed_count += 1
            except:
                pass

        return word_vectors

    def create_regular_context_vectors(self):
        """
        Create Random Indexing context vectors.
        """
        print 'Regular Random Indexing'

        self.seed_count = 1

        # Instantiate word space matrix.
        context_vectors = np.zeros((self.size, self.dimensionality),
                                   dtype=np.int32)
        word_vectors = self.create_word_vectors()

        # Creating one matrix per step removed.
        for window in range(self.window_start,
                            self.window_start + self.window_size):

            print window

            # Create word queue.
            que = deque(maxlen=window + 1)

            # Open corpus file and go over each line in corpus.
            with codecs.open(self.file_path, 'r', "ISO-8859-1") as handle:

                for line in handle:

                    if line.startswith('CURRENT'):
                        # If line starts with current it is just a reference in
                        # UKWAC to the homepage which text was taken from.
                        pass

                    else:
                        # Process text from corpora and return it with the
                        # output() command.
                        # It is then return as a python list with sentences
                        # that are also in
                        # in list format with one word per entry.
                        self.t.read(line)

                        for sentence in self.t.output():
                            que.clear()
                            for word in sentence:
                                que.append(word)

                                # Check if there exist a non stop-word word
                                # at window distance
                                # and add it to matrix if it does.
                                try:
                                    context_vectors[
                                        self.mapping[que[window]]] += \
                                        word_vectors[
                                            self.mapping[que[0]]]  # pos
                                except:
                                    pass
                                try:
                                    context_vectors[self.mapping[que[0]]] += \
                                        word_vectors[
                                            self.mapping[que[window]]]  # neg
                                except:
                                    pass

        np.save('npy_files/RIS_regRI_s' + str(self.window_start) + 'w' +
                str(self.window_size) + '_10M.npy', context_vectors)

    def create_extended_context_vectors(self):
        """
        Create extended Random Indexing context vectors.
        """

        print 'Extended Random Indexing'

        self.seed_count = 1

        # Instantiate word space matrix.
        context_vectors = np.zeros((self.size, self.dimensionality),
                                   dtype=np.int32)

        # Creating one matrix per step removed.
        for window in range(self.window_start,
                            self.window_start + self.window_size):

            print window

            word_vectors_pos = self.create_word_vectors()
            word_vectors_neg = self.create_word_vectors()

            # Create word queue.
            que = deque(maxlen=window + 1)

            # Open corpus file and go over each line in corpus.
            with codecs.open(self.file_path, 'r', "ISO-8859-1") as handle:

                for line in handle:

                    if line.startswith('CURRENT'):
                        # If line starts with current it is just a reference in
                        # UKWAC to the homepage which text was taken from.
                        pass

                    else:
                        # Process text from corpora and return it with the
                        # output() command.
                        # It is then return as a python list with sentences
                        # that are also in
                        # in list format with one word per entry.
                        self.t.read(line)

                        for sentence in self.t.output():
                            que.clear()
                            for word in sentence:
                                que.append(word)

                                # Check if there exist a non stop-word word
                                # at window distance
                                # and add it to matrix if it does.
                                try:
                                    context_vectors[
                                        self.mapping[que[window]]] += \
                                        word_vectors_pos[
                                            self.mapping[que[0]]]  # pos
                                except:
                                    pass
                                try:
                                    context_vectors[self.mapping[que[0]]] += \
                                        word_vectors_neg[
                                            self.mapping[que[window]]]  # neg
                                except:
                                    pass

        np.save('npy_files/RIS_extRI_s' + str(self.window_start) + 'w' +
                str(self.window_size) + '_10M.npy', context_vectors)

if __name__ == '__main__':
    RI = RandomIndexing()
    #RI.create_regular_context_vectors()
    RI.create_mod_regular_context_vectors()
    #RI.create_extended_context_vectors()
