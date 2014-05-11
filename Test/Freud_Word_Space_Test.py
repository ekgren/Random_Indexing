import Random_Indexing
import nltk

__author__ = "Ariel Ekgren, https://github.com/ekgren/"

f = open('../data/freud.txt')
raw = f.read()

sent_detector = nltk.tokenize.PunktSentenceTokenizer(raw)

tokens = nltk.word_tokenize(raw)
fdist = nltk.FreqDist(tokens)

word_space = Random_Indexing.RandomIndexing()
word_space.size = 2000
word_space.dimensionality = 1000
word_space.random_degree = 10
word_space.create_regular_word_space()

mapping = Random_Indexing.Mapping()
mapping.create_map(fdist.keys())
mapping.word_space = word_space

text = sent_detector.tokenize(raw)

for sentence in text:
    mapped_sentence = mapping.map_sequence(nltk.word_tokenize(sentence))
    word_space.update_context_vectors(mapped_sentence)

print mapping.nn_context_context('dream')
print mapping.nn_context_base('dream')