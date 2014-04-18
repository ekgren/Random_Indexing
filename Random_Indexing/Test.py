__author__ = 'Ariel Ekgren, https://github.com/ekgren/'

import RandomIndexing
import nltk

RI = RandomIndexing.RandomIndexing()
RI.size = 500
RI.dimensionality = 500
RI.random_degree = 6
RI.create_regular_word_space()

f = open('../data/freud.txt')
raw = f.read()

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

tokens = nltk.word_tokenize(raw)
fdist = nltk.FreqDist(tokens)
mapping = {}

for i, word in enumerate(fdist.keys()):
    mapping[word] = i

text = sent_detector.tokenize(raw)

for S in text:
    sent = []
    for x in nltk.word_tokenize(S):
        try:
            sent.append(mapping[x])
        except:
            pass
    RI.update_context_vectors(sent)

target = 'the'

neighb_args, neighb_vals = RI.nearest_neighbours(mapping[target], 5)

print neighb_vals

print "Target word:", target

for i, x in enumerate(neighb_args):
    print "    ", i+1, fdist.keys()[x]

print "\nStop."