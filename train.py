import sys
import os

import numpy as np 

from gensim.models import Word2Vec, KeyedVectors
from nltk import sent_tokenize
from nltk.tokenize import wordpunct_tokenize

import pickle
import time

start_time = time.time()

input_data_folder_path = sys.argv[1]
model_path = sys.argv[2]
pretrained_embeddings_path = sys.argv[3]


tokenized_sent = []
for i in os.listdir(input_data_folder_path):
    with open(input_data_folder_path + '/' + i, 'r') as file:
        document = file.read()
        sentence = sent_tokenize(document)
        for j in sentence:
            tokenized_sent.append(j)

word_tokenize = [wordpunct_tokenize(sent) for sent in tokenized_sent]
print("Checkpoint 1 passed")

model_2 = Word2Vec(size = 300, window = 5, alpha = 0.01, min_count = 2, iter = 100, hs=1, negative=0)
model_2.build_vocab(word_tokenize)

total_examples = model_2.corpus_count

model = KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)

print("Checkpoint 2 passed")

model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format(pretrained_embeddings_path, binary=True)
model_2.train(word_tokenize, total_examples=total_examples, epochs=200)


model_file = open(model_path, 'wb')
pickle.dump(model_2, model_file)
model_file.close()

end_time = time.time()

print("Time taken = ", (end_time - start_time)/60, 'minutes')

