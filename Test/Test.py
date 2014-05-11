import Random_Indexing
import nltk
import cPickle

__author__ = "Ariel Ekgren, https://github.com/ekgren/"

word_space = Random_Indexing.RandomIndexing()
word_space.size = 2000
word_space.dimensionality = 1000
word_space.random_degree = 10
word_space.create_regular_word_space()