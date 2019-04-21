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


path = os.path.join(os.getcwd(),"app", "irsystem", "models", "cleaned_comprehensive_data.csv")
data = pd.read_csv(path)
num_dramas = len(data)
path2 = os.path.join(os.getcwd(),"app", "irsystem", "models",'cosine_matrix.npy')
drama_sims_cos = np.load(path2)
path3 = os.path.join(os.getcwd(),"app", "irsystem", "models",'korean_data.csv')
non_processed_data = pd.read_csv(path3)
path4 = os.path.join(os.getcwd(),"app", "irsystem", "models",'genre_inclusion_matrix.npy')
genre_inclusion_matrix  = np.load(path4)
path5 = os.path.join(os.getcwd(),"app", "irsystem", "models",'actors_inclusion_matrix.npy')
actors_inclusion_matrix  = np.load(path5)
path6 = os.path.join(os.getcwd(),"app", "irsystem", "models", 'years_inclusion_matrix.npy')
years_inclusion_matrix  = np.load(path6)
non_processed_data = pd.read_csv(path3)
drama_index_to_name = non_processed_data['Title'].to_dict()
process_dict = data['Title'].to_dict()
drama_name_to_index = {v: k for k, v in process_dict.items()}
drama_name_to_index_unprocess = {v: k for k, v in drama_index_to_name.items()}

with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'genre_name_to_index.json')) as fp:
    genre_name_to_index = json.load(fp)
with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'actors_name_to_index.json')) as fp2:
    actors_name_to_index = json.load(fp2)
with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'years_name_to_index.json')) as fp3:
    years_name_to_index = json.load(fp3)
with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'genres_dict.json')) as fp:
    genre_dict = json.load(fp)
with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'actors_dict.json')) as fp2:
    actors_dict = json.load(fp2)
with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'years_dict.json')) as fp3:
    years_dict = json.load(fp3)

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
    """Removes stems from a string.
    Params: {text: String}
    Returns: String
    """
    stemmer=PorterStemmer()
    stems = [stemmer.stem(w) for w in tokenize(text)]
    return " ".join(stems)

def preprocess_text(text):
    text = str(text)
    text = cleanhtml(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = stem(text)
    return text

def map_network(network,x):
    if x == network:
        return 1
    else:
        return 0

def best_match(actors_dict, genre_inclusion_matrix, actors_inclusion_matrix, years_inclusion_matrix, genre_name_to_index, actors_name_to_index, years_name_to_index, drama_sims_cos, data, drama_index_to_name, drama_name_to_index, dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, preferred_actors, preferred_time_frame, num_results):

    feature_list = ['Summary_Similarity', 'Actor_Similarity', 'Genre_Similarity', 'Network_Similarity','Year_Similarity', 'Total']
    result = pd.DataFrame(0, index=np.arange(1466), columns=feature_list)
    genres = set()
    preferred_genres = [preprocess_text(value) for value in preferred_genres]
    genres.update(preferred_genres)
    years = preferred_time_frame.split('-')
    start_year = int(years[0])
    end_year = int(years[1])
    preferred_actors = preferred_actors.split(", ")
    preferred_actors_set = set()
    preferred_actors_set.update(preferred_actors)
    d = {k:len(v) for k, v in actors_dict.items()}
    actors_len_df = pd.DataFrame.from_dict(d, orient='index')
    actors_len_df.columns = ['Length']
    for drama in dramas_enjoyed:
        drama = drama.lower()
        drama = drama.strip()
        if drama in drama_name_to_index.keys():
            index = drama_name_to_index[drama]
            sim = drama_sims_cos[index,:1466]
            result['Summary_Similarity']+= pd.Series(sim)
    for drama in dramas_disliked:
        drama = drama.lower()
        drama = drama.strip()
        if drama in drama_name_to_index.keys():
            index = drama_name_to_index[drama]
            sim = drama_sims_cos[index,:1466]
            result['Summary_Similarity']-= pd.Series(sim)
    for genre in preferred_genres:
        if genre in genre_name_to_index.keys():
            index = genre_name_to_index[genre]
            result['Genre_Similarity']= genre_inclusion_matrix[:,index]
    for actor in preferred_actors:
        if actor in actors_name_to_index.keys():
            index = actors_name_to_index[actor]
            result['Actor_Similarity']+= actors_inclusion_matrix[:,index]
    actors_len_df['Length'] =  actors_len_df['Length'] + len(preferred_actors)
    actors_len_df['Length'] = actors_len_df['Length'].subtract(result['Actor_Similarity'], fill_value = 0)
    actor_sim2=result['Actor_Similarity']
    for idx in range(1466):
        result['Actor_Similarity'] = actor_sim2.iloc[idx]/actors_len_df['Length'].iloc[idx]
    if str(start_year) in years_name_to_index.keys():
        index = years_name_to_index[str(start_year)]
        result['Year_Similarity'] = years_inclusion_matrix[:,index]
    if str(end_year) in years_name_to_index.keys():
        index = years_name_to_index[str(end_year)]
        result['Year_Similarity'] = pd.concat([pd.Series(years_inclusion_matrix[:,index]), result['Year_Similarity']], axis=1).min(axis=1)
    result['Network_Similarity'] = data['Network'].apply(lambda x: map_network(x, preferred_network))
    result['Year_Similarity'] = 1 - result['Year_Similarity']/(result['Year_Similarity'].max()+1)
    result['Total'] = round(result['Summary_Similarity']*.6 + result['Actor_Similarity']*.1 + result['Year_Similarity']*.05 + result['Genre_Similarity']*.2 + result['Network_Similarity']*.05,5)
    result = result.sort_values(by='Total', ascending=False)
    result = result[:num_results]
    indices =  result.index.tolist()
    best_dramas = pd.Series([drama_index_to_name[index] for index in indices],index = result.index)
    result.insert(loc=0, column='Drama_Title', value=best_dramas)
    result.reset_index()
    return result

def display (dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, preferred_actors, preferred_time_frame, num_results):
    dramas_enjoyed = dramas_enjoyed.split(', ')
    dramas_disliked = dramas_disliked.split(', ')
    preferred_genres = preferred_genres.split(', ')
    best = best_match(actors_dict, genre_inclusion_matrix, actors_inclusion_matrix, years_inclusion_matrix, genre_name_to_index, actors_name_to_index, years_name_to_index,drama_sims_cos, data, drama_index_to_name, drama_name_to_index,  dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, preferred_actors, preferred_time_frame, num_results)
    title = list(zip(best['Drama_Title'], best["Total"]))
    final = {}
    for idx, x in enumerate(title):
        title_name = x[0]
        final.update({x[0]: ''})
        summary = str(non_processed_data['Summary'].loc[drama_name_to_index_unprocess[title_name]])
        if summary != "nan":
            final[title_name] += summary
    return ['Drama Title: {},  Summary: {},  Total Similarity Score: {}'.format(x[0], final[x[0]], x[1]) for x in title]

print(display("House, The Mindy Project", "The Office, Game of Thrones", "medical", "", "","2010-2015", 10))
