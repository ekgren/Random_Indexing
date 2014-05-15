from collections import Counter
import time

import Random_Indexing

__author__ = "Ariel Ekgren, https://github.com/ekgren/"

# Create word space.
t = time.time()

word_space = Random_Indexing.RandomIndexing()
word_space.size = 4000
word_space.dimensionality = 2000
word_space.random_degree = 20
word_space.create_regular_word_space()

print "Word space created in " + str(time.time()-t) + " seconds."

# Create vocabulary.
t = time.time()

vocabulary = Counter()

with open('../../data/freud_fixed.txt') as f:
    for line in f:
        for word in line.split(' '):
            vocabulary[word] += 1

vocabulary = [x[0] for x in vocabulary.most_common(word_space.size)]

mapping = Random_Indexing.Mapping()
mapping.create_map(vocabulary)
mapping.word_space = word_space

print "Vocabulary created in " + str(time.time()-t) + " seconds."

# Update word space.
t = time.time()

with open('../../data/freud_fixed.txt') as f:
    for line in f:
        mapped_sentence = mapping.map_sequence(line.split(' '))
        word_space.update_context_vectors(mapped_sentence)
        #word_space.experimental_update(mapped_sentence)

print "Word space updated in " + str(time.time()-t) + " seconds."

# Nearest neighbours.
t = time.time()

print
print mapping.nn_context_context('genitals')
print
print mapping.nn_context_context('intelligence')
print
print mapping.nn_context_context('sexual')
print

print "Neighbours found in " + str(time.time()-t) + " seconds."
