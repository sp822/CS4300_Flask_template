from __future__ import print_function
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#import requests
import re
import string
from operator import itemgetter
'''from nltk.stem import PorterStemmer'''
import os
import numpy as np
import pandas as pd
from collections import Counter
#import csv
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#import seaborn as sns
import math
#import scipy

path = os.path.join("Data-Set-Final.csv")
data = pd.read_csv(path)

def cleanhtml(raw_html):
    clean = re.compile('<.*?>')
    cleantext = re.sub(clean, '', raw_html)
    return cleantext

def tokenize(text):
    """Returns a list of words that make up the text.
    Params: {text: String}
    Returns: List
    """
    return list(filter(str.strip, list(map(lambda x: x, re.findall(r'[a-zA-Z]*', text)))))

'''def stem(text):
    stemmer=PorterStemmer()
    stems = [stemmer.stem(w) for w in tokenize(text)]
    return " ".join(stems)'''

def preprocess_text(text):
    text = str(text)
    text = cleanhtml(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def build_inverted_index_and_regular_index(data):
    """ Builds an inverted index from the movies
    """
    result = {}
    data_index = {}
    for (index,value) in data['Summary'].items():
        value = preprocess_text(value)
        toks = value.split()
        counts = Counter(toks)
        data_index[index] = counts.items()
        for word, count in counts.items():
            if word not in result.keys():
                result[word] = []
            result[word].append((index, count))
    return (result, data_index)

def compute_idf(inv_idx, n_movies):
    idf_dict = {}
    for word in inv_idx.keys():
        DF = len(inv_idx[word])
        idf_dict[word] = math.log(n_movies/(1+DF),2)
    return idf_dict

def preprocess(data):
    for (index,value) in data['Summary'].items():
        value = preprocess_text(value)
        data.loc[index,'Summary'] = value
    return data

def build_tfidf_matrix(data_index, n_movies,vocab_to_index, idf_dict):
    result = np.zeros((n_movies, len(vocab_to_index.keys())))
    for idx in range(n_movies):
        sum_terms = 1
        for (word, count) in data_index[idx]:
            sum_terms = sum_terms + count
            idx2 = vocab_to_index[word]
            idf = idf_dict[word]
            tfidf = idf*count
            result[idx, idx2] = tfidf
        result[idx,:]=np.divide(result[idx,:],sum_terms)

    return result

def get_sim(mov1, mov2, input_doc_mat, movie_name_to_index):
    """Returns a float giving the cosine similarity of
       the two movie transcripts.

    Params: {mov1: String,
             mov2: String,
             input_doc_mat: Numpy Array,
             movie_name_to_index: Dict}
    Returns: Float (Cosine similarity of the two movie transcripts.)
    """
    idx1 = movie_name_to_index[mov1]
    idx2 = movie_name_to_index[mov2]
    movie1 = input_doc_mat[idx1,]
    movie2 = input_doc_mat[idx2,]



    #dot_product = 1 - scipy.spatial.distance.cosine(movie1, movie2)
    dot_product = np.dot(movie1, movie2)
    #/(np.linalg.norm(movie1)* np.linalg.norm(movie2))
    return dot_product

def build_movie_sims_cos(n_mov, movie_index_to_name, input_doc_mat, movie_name_to_index, input_get_sim_method):
    """Returns a movie_sims matrix of size (num_movies,num_movies) where for (i,j):
        [i,j] should be the cosine similarity between the movie with index i and the movie with index j

    Params: {n_mov: Integer,
             movie_index_to_name: Dict,
             input_doc_mat: Numpy Array,
             movie_name_to_index: Dict,
             input_get_sim_method: Function}
    Returns: Numpy Array
    """
    result = np.zeros((n_mov, n_mov))
    for i in range(n_mov):
        for j in range(n_mov):
            if i == j:
                result[i,j] = 0
            else:
                mov1 = movie_index_to_name[i]
                mov2 = movie_index_to_name[j]
                result[i,j] = input_get_sim_method(mov1, mov2, input_doc_mat, movie_name_to_index)
    return result

data = preprocess(data)
num_movies = len(data)
res = build_inverted_index_and_regular_index(data)
inv_index = res[0]
reg_index = res[1]
idf_dict = compute_idf(inv_index, num_movies)
index_to_vocab = {i:v for i, v in enumerate(inv_index.keys())}
vocab_to_index = {v: k for k, v in index_to_vocab.items()}
doc_by_vocab = build_tfidf_matrix(reg_index, num_movies,vocab_to_index, idf_dict)
movie_index_to_name = data['Title'].to_dict()
movie_name_to_index = {v: k for k, v in movie_index_to_name.items()}
movie_sims_cos = build_movie_sims_cos(num_movies, movie_index_to_name, doc_by_vocab, movie_name_to_index, get_sim)
np.save(file = os.path.join('cosinematrix.npy'), arr = movie_sims_cos)
data.to_csv(os.path.join('cleaneddata.csv'))
