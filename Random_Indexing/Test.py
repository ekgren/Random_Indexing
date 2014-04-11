__author__ = 'Ariel Ekgren, https://github.com/ekgren/'

import RandomIndexing
import nltk

print "Start.\n"

RI = RandomIndexing.RandomIndexing()
RI.size = 500
RI.dimensionality = 1000
RI.random_degree = 10
RI.create_regular_word_space()

f = open('C:/Alpha/datasets/freud.txt')
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

target = 'libido'

neighbs = RI.nearest_neighbours(mapping[target], 5)

print neighbs

print target
for x in neighbs:
    print fdist.keys()[x[0]]

print "\nStop."