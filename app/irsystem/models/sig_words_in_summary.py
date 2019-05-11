#import statemtns
from __future__ import print_function
import re
import string
from operator import itemgetter
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import json
from nltk.stem import PorterStemmer

#constants and loading data
path = os.path.join(os.getcwd(),"app", "irsystem", "models", "cleaned_comprehensive_data.csv")
data = pd.read_csv(path)
num_dramas = len(data)
path2 = os.path.join(os.getcwd(),"app", "irsystem", "models",'cosine_matrix.npy')
drama_sims_cos = np.load(path2)

path3 = os.path.join(os.getcwd(), "app", "irsystem", "models", "doc_to_vocab.npy")
doc_to_vocab = np.load(path3)

with open(os.path.join(os.getcwd(), "app", "irsystem", "models",'tfidf_index_to_vocab.json')) as fp8:
    tfidf_index_to_vocab = json.load(fp8)


def generate_sig_words_matrix():
	(r, c) = drama_sims_cos.shape

	#create 2d array of lists
	sig_words = np.empty( (r,c), dtype=object)

	plain = []

	#find the top 10 words of similarity between indices 
	for i in range (r):
		for j in range(i,c):
			if i == j:
				sig_words[i][j] = plain
			show1 = doc_to_vocab[i]
			show2 = doc_to_vocab[j]
			mult = np.multiple(show1, show2)
			mult.argsort()[-10:][::-1]

			words = []
			for index in mult:
				word = tfidf_index_to_vocab[index]
				words.append(word)

			sig_words[i][j] = words
			sig_words[j][i] = words

	return sig_words

sig_words = generate_sig_words_matrix()
np.save('sig_words.npy', sig_words)
