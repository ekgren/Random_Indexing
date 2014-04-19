__author__ = 'Ariel Ekgren, https://github.com/ekgren/'

import Random_Indexing
import nltk

f = open('../data/freud.txt')
raw = f.read()

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

tokens = nltk.word_tokenize(raw)
fdist = nltk.FreqDist(tokens)

word_space = Random_Indexing.RandomIndexing()
word_space.size = 2000
word_space.dimensionality = 500
word_space.random_degree = 6
word_space.create_regular_word_space()

mapping = Random_Indexing.Mapping()
mapping.word_space = word_space
mapping.create_map(fdist.keys())

text = sent_detector.tokenize(raw)

for sentence in text:
    mapped_sentence = mapping.map_sequence(nltk.word_tokenize(sentence))
    word_space.update_context_vectors(mapped_sentence)

print mapping.nearest_neighbours('ego')
print mapping.nearest_neighbours('child')
print mapping.nearest_neighbours('father')
print mapping.nearest_neighbours('think')
print mapping.nearest_neighbours('xxxxxxxxxx')