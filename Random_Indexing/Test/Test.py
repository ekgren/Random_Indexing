from collections import Counter
from collections import deque
import time
from random import randint

import Random_Indexing

__author__ = "Ariel Ekgren, https://github.com/ekgren/"

# Create word space.
t = time.time()

word_space = Random_Indexing.RandomIndexing(2)
word_space.size = 2000
word_space.dimensionality = 1000
word_space.random_degree = 10
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
        #word_space.update_context_vectors(mapped_sentence)
        word_space.experimental_update(mapped_sentence, False)

print "Word space updated in " + str(time.time()-t) + " seconds."

# Nearest neighbours.
t = time.time()
'''
print
for word in ['dreams']:
    print mapping.nn_context_context(word)
    print mapping.nn_context_base(word)
    print mapping.nn_base_context(word)

print mapping.nn_multi_context_base(['he', 'discovers'])
print

generated_sentence = []
current_word = 'dreams'

for x in range(20):
    generated_sentence.append(current_word)
    current_word = mapping.nn_base_context(current_word)[1][randint(0,2)]

print ' '.join(generated_sentence)
print

generated_sentence = []
current_word = 'dreams'

for x in range(20):
    generated_sentence.append(current_word)
    current_word = mapping.nn_context_context(current_word)[1][randint(0,2)]

print ' '.join(generated_sentence)
print

generated_sentence = []
current_word = 'dreams'

for x in range(20):
    generated_sentence.append(current_word)
    current_word = mapping.nn_context_base(current_word, 10)[1][randint(0,9)]

print ' '.join(generated_sentence)
print

generated_sentence = []
previous_word = 'the'
current_word = 'father'
generated_sentence.append(previous_word)
for x in range(20):
    generated_sentence.append(current_word)
    previous_word = current_word
    current_word = mapping.nn_multi_base_context([previous_word, current_word], 10)[1][randint(0,9)]

print ' '.join(generated_sentence)
print
'''

word1 = 'forced'
word2 = 'courage'
full_sentence = [word1, word2]
generated_sentence = deque(maxlen=2)
generated_sentence.append(word1)
generated_sentence.append(word2)
for x in range(100):
    current_word = mapping.nn_multi_context_base(generated_sentence, 8)[1][randint(0, 7)]
    generated_sentence.append(current_word)
    full_sentence.append(current_word)

print
print ' '.join(full_sentence)
print

print "Neighbours found in " + str(time.time()-t) + " seconds."