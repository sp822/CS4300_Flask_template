from __future__ import print_function
import requests
import re
import string
from operator import itemgetter
from nltk.stem import PorterStemmer
import os

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import csv

cwd = os.getcwd()
print(cwd)
data = pd.read_csv(os.path.join(cwd,app,irsystem, models, Data-Set-Final.csv))


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

def stem(text):
    stemmer=PorterStemmer()
    stems = [stemmer.stem(w) for w in tokenize(text)]
    return " ".join(stems)

def preprocess_text(text):
    text = str(text)
    text = cleanhtml(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def preprocess(data):
    for (index,value) in data['Summary'].items():
        value = preprocess_text(value)
        value = stem(value)
        data.loc[index,'Summary'] = value
    return data

n_feats = 5000
doc_by_vocab = np.empty([len(data), n_feats])

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    """Returns a TfidfVectorizer object with the above preprocessing properties.

    Params: {max_features: Integer,
             max_df: Float,
             min_df: Float,
             norm: String,
             stop_words: String}
    Returns: TfidfVectorizer
    """

    result = TfidfVectorizer(max_features = max_features, stop_words = stop_words, max_df = max_df, min_df = min_df, norm = norm)
    return result

data = preprocess(data)
tfidf_vec = build_vectorizer(n_feats, "english")
doc_by_vocab = tfidf_vec.fit_transform([value for _,value in data['Summary'].items()]).toarray()
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}
movie_index_to_name = data['Title'].to_dict()
movie_name_to_index = {v: k for k, v in movie_index_to_name.items()}
num_movies = len(data)

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
    dot_product = np.dot(movie1, movie2)
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

movie_sims_cos = build_movie_sims_cos(num_movies, movie_index_to_name, doc_by_vocab, movie_name_to_index, get_sim)

def best_match(n_mov, movie_sims_cos, data, movie_index_to_name, movie_name_to_index, dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, num_results):
    feature_list = ['Summary_Similarity', 'Genre_Similarity', 'Network_Similarity', 'Total']
    result = pd.DataFrame(0, index=np.arange(n_mov), columns=feature_list)
    genres = set()
    preferred_genres = [preprocess_text(value) for value in preferred_genres]
    genres.update(preferred_genres)
    for drama in dramas_enjoyed:
        if drama in movie_name_to_index.keys():
            index = movie_name_to_index[drama]
            sim = movie_sims_cos[index,:]
            result['Summary_Similarity']+= pd.Series(sim)

    for drama in dramas_disliked:
        if drama in movie_name_to_index.keys():
            index = movie_name_to_index[drama]
            sim = movie_sims_cos[index,:]
            result['Summary_Similarity']-= pd.Series(sim)

    for index, value in data.iterrows():
        gen = str(value['Genre'])
        gen = preprocess_text(gen)
        drama_genres = set()
        drama_genres.update(gen.split())
        result.loc[index,'Genre_Similarity'] = len(genres.intersection(drama_genres))/len(genres.union(drama_genres))
        if preferred_network == data.iloc[index]['Network']:
            result['Network_Similarity']+=1
    result['Total'] = result.sum(axis = 1)
    result = result.sort_values(by='Total', ascending=False)
    result = result[:num_results]
    indices =  result.index.tolist()
    best_dramas = pd.Series([movie_index_to_name[index] for index in indices],index = result.index)
    result.insert(loc=0, column='Drama_Title', value=best_dramas)
    result.reset_index()
    return result

def display (n_mov, movie_sims_cos, data, movie_index_to_name, movie_name_to_index, dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, num_results):
    dramas_enjoyed = dramas_enjoyed.split(',')
    dramas_disliked = dramas_disliked.split(',')
    preferred_genres = preferred_genres.split(',')
    preferred_network = preferred_network.split(',')
    best = best_match(n_mov, movie_sims_cos, data, movie_index_to_name, movie_name_to_index, dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, num_results)
    title = list(zip(best['Drama_Title'], best["Total"]))
    final = ["Drama Titles: {}".format(final_title[0]) + "            " +"Total Similarity {}".format(final_title[1]) for final_title in title]
    return final
