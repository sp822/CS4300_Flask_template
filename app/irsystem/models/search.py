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

path = os.path.join(os.getcwd(), "app", "irsystem", "models", "cleaneddata.csv")
data = pd.read_csv(path)
num_movies = len(data)
path2 = os.path.join(os.getcwd(), "app", "irsystem", "models", 'cosinematrix.npy')
movie_sims_cos = np.load(path2)
movie_index_to_name = data['Title'].to_dict()
movie_name_to_index = {v: k for k, v in movie_index_to_name.items()}
# def display_sim_matrix(sim_matrix, diag = False):
#     fig, ax = plt.subplots()
#     plt_title = "KDramas Cos-Sim Heatmap"
#     plt.title(plt_title, fontsize = 18)
#     ttl = ax.title
#     ttl.set_position([0.5, 1.05])

#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.axis('off')

#     mask = None
#     if diag:
#         mask = np.tri(sim_matrix.shape[0], k=-1)

#     heatmap = sns.heatmap(sim_matrix, fmt="", cmap='BuGn_r', linewidths=0, mask = mask, ax=ax)

#     plt.show()
#     fig = heatmap.get_figure()
#     fig.savefig('sim_heatmap1.png', dpi=400)

#     m_size = len(sim_matrix)
#     scores = np.zeros((m_size+1)//2*m_size)
#     cnt = 0
#     for i in range (0, m_size):
#         for j in range(i, m_size):
#             scores[cnt] = sim_matrix[i][j]
#             cnt+=1

#     sns.distplot(scores, hist=True, kde=True,
#              bins=int(180/5), color = 'darkblue',
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 4})

def cleanhtml(raw_html):
    clean = re.compile('<.*?>')
    cleantext = re.sub(clean, '', raw_html)
    return cleantext

def preprocess_text(text):
    text = str(text)
    text = cleanhtml(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

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
    dramas_enjoyed = dramas_enjoyed.split(', ')
    dramas_disliked = dramas_disliked.split(', ')
    preferred_genres = preferred_genres.split(', ')
    preferred_network = preferred_network.split(', ')
    best = best_match(n_mov, movie_sims_cos, data, movie_index_to_name, movie_name_to_index, dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, num_results)
    title = list(zip(best['Drama_Title'], best["Total"]))
    final = {}
    for x in title:
        title_name = x[0]
        final.update({x[0]: ''})
        final[title_name] = data['Summary'][list(data['Title']).index(title_name)]
    return ['Drama Title: {},  Summary: {},  Total Similarity Score: {}'.format(x[0], final[x[0]], x[1]) for x in title]
