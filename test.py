import sys
import os

import numpy as np 

from gensim.models import Word2Vec, KeyedVectors
from nltk import sent_tokenize
from nltk.tokenize import wordpunct_tokenize

import pickle
import time

start_time = time.time()

eval_data = sys.argv[1]
eval_data_td = sys.argv[2]
model_path = sys.argv[3]
pretrained_embeddings_path = sys.argv[4]

def text(eval_data):

    left = []
    right = []
    ground_truth = []
        
    eval_data_txt = open(eval_data, 'r')

    for lines in eval_data_txt.readlines():
        sent = lines.split('::::')
        ground_truth.append(sent[1].split())
        line = ''.join(sent[0])
        left_right = line.split('<<target>>')
        left_tok = wordpunct_tokenize(left_right[0])[-2:]
        left.append(left_tok)
        right_tok = wordpunct_tokenize(left_right[1])[:2]
        right.append(right_tok)

    return left, right, ground_truth

def labels(eval_data_td):

    labels = []
    eval_data_txt_td = open(eval_data_td, 'r')

    for lines in eval_data_txt_td.readlines():
        labels.append(lines.split(' '))
    
    return labels

left, right, _ = text(eval_data)
labels = labels(eval_data_td)

model_file = open(model_path, 'rb')
model = pickle.load(model_file)

output_file = open('output.txt', 'w')

test_data_size = len(labels)

for i in range(test_data_size):
    predicted = []
    for label in range(len(labels[i])):
        predicted.append(left[i] + [labels[i][label]] + right[i])
    score = model.score(predicted)
    scores = []
    for label in range(len(labels[i])):
        if labels[i][label] in model.wv.vocab:
            scores.append((1, float(score[label]), labels[i][label]))
        else: scores.append((0, float(score[label]), labels[i][label]))

    scores.sort(reverse = True)
    
    ranking = {}
    rank = 1
    for _, _, label in scores:
        ranking[label] = rank
        rank += 1

    for label in labels[i]:
        output_file.write('%d ' %ranking[label])
    output_file.write('\n')

output_file.close()    
        
end_time = time.time()
print("Time taken = ", (end_time - start_time)/60, 'minutes')






