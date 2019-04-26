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

emb_sim_matrix = np.load('emb_sim_matrix_1.npy')

with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'sentiment_analysis.json')) as fp4:
    sentiment_dict = json.load(fp4)


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
def best_match(dramas_enjoyed, dramas_disliked, preferred_genres, preferred_actors, preferred_time_frame, num_results):
    feature_list = ['Embedding_Similarity','Summary_Similarity', 'Actor_Similarity', 'Genre_Similarity','Sentiment_Analysis', 'Total']
    result = pd.DataFrame(0, index=np.arange(1466), columns=feature_list)
    genres = set()
    preferred_genres = [preprocess_text(value) for value in preferred_genres]
    for genre in preferred_genres:
        genres.update(preferred_genres)
    years = preferred_time_frame
    start_year = int(years[0])
    end_year = int(years[1])
    embedding_bool = True
    preferred_actors_set = set()
    preferred_actors_set.update(preferred_actors)
    d = {k:len(v) for k, v in actors_dict.items()}
    actors_len_df = pd.DataFrame.from_dict(d, orient='index')
    actors_len_df.columns = ['Length']
    d2 = {int(k):float(v) for k, v in sentiment_dict.items()}
    result['Sentiment_Analysis']= pd.DataFrame.from_dict(d2, orient='index')
    for drama in dramas_enjoyed:
        drama = drama.lower()
        drama = drama.strip()
        if drama in drama_name_to_index.keys():
            index = drama_name_to_index[drama]
            sim = drama_sims_cos[index,:1466]
            result['Summary_Similarity']+= pd.Series(sim)
            sim_doc = emb_sim_matrix[index]
            result['Embedding_Similarity']+= pd.Series(sim_doc)

    for drama in dramas_disliked:
        drama = drama.lower()
        drama = drama.strip()
        if drama in drama_name_to_index.keys():
            index = drama_name_to_index[drama]
            sim = drama_sims_cos[index,:1466]
            result['Summary_Similarity']-= pd.Series(sim)
            sim_doc = emb_sim_matrix[index]
            result['Embedding_Similarity']-= pd.Series(sim_doc)

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
    result['Embedding_Similarity'] = result['Embedding_Similarity']/(result['Embedding_Similarity'].max()+1)
    result['Total'] = round(result['Embedding_Similarity']*.1 + result['Summary_Similarity']*.5 + result['Sentiment_Analysis']*.1 + result['Actor_Similarity']*.15 + result['Genre_Similarity']*.15,4)
    result = result.sort_values(by='Total', ascending=False)
    index1 = years_name_to_index[str(start_year)]
    index2 = years_name_to_index[str(end_year)]
    for idx, res in result.iterrows():
        mat = years_inclusion_matrix[idx, index1:index2]
        if sum(mat) == 0:
            result = result[result.index != idx]
    result = result[:num_results]
    indices =  result.index.tolist()
    best_dramas = pd.Series([drama_index_to_name[index] for index in indices],index = result.index)
    result.insert(loc=0, column='Drama_Title', value=best_dramas)
    result.reset_index()
    return result


def display (dramas_enjoyed, dramas_disliked, preferred_genres, preferred_actors, preferred_time_frame, num_results):
    dramas_enj = dramas_enjoyed.split(', ')
    dramas_dis = dramas_disliked.split(', ')
    preferred_acts =  preferred_actors.split(', ')
    preferred_genres = preferred_genres.split(', ')
    """
    print("dramas_enjoyed: " + dramas_enjoyed)
    print("dramas_disliked: " + dramas_disliked)
    print("preferred_genres: " + str(preferred_genres))
    print("preferred_network: " +preferred_network)
    print("preferred_actors: " + preferred_actors)
    print("preferred_time_frame: " + str(preferred_time_frame))
    """

    best = best_match(dramas_enj, dramas_dis, preferred_genres, preferred_acts, preferred_time_frame, num_results)
    """
    print(best)
    """
    result = list(zip(best['Drama_Title'], best["Total"]))
    titles = {}
    summaries = {}
    genres = {}
    ratings = {}
    runtimes = {}
    actors = {}
    votes = {}
    years = {}
    networks = {}

    for title, score in result:
        idx = drama_name_to_index_unprocess[title]
        summary = str(non_processed_data['Summary'].loc[idx])
        if summary != "nan":
            summaries[title] = summary
        else:
            summaries[title] = "No summary information is available."
        genre = str(non_processed_data['Genre'].loc[idx])
        if genre != "nan":
            genre = genre.strip('[]')
            genre = genre.replace("'", "")
            genres[title] = genre
        else:
            genres[title] = "No genre information is available."
        rating = str(data['Rating'].loc[idx])
        if rating != "nan":
            ratings[title] = rating
        else:
            ratings[title] = "No rating information is available."
        runtime = str(non_processed_data['Runtime'].loc[idx])
        if runtime != "nan":
            runtimes[title] = rating
        else:
            runtimes[title] = "No runtime information is available."
        actor = str(non_processed_data['Actors'].loc[idx])
        if actor != "nan":
            actors[title] = actor
        else:
            actors[title] = "No actor information is available."
        network = str(non_processed_data['Network'].loc[idx])
        if network != "nan":
            networks[title] = network
        else:
            networks[title] = "No actor information is available."
        vote = str(non_processed_data['Votes'].loc[idx])
        if vote != "nan":
            votes[title] = vote
        else:
            votes[title] = "No votes information is available."
        year = str(data['Year'].loc[idx])
        if year != "nan":
            years[title] = year
        else:
            years[title] = "No timeframe information is available."
    return ['{},  Summary: {},  Genre: {}, Rating: {}, Runtime: {}, Actors: {}, Votes: {}, Years: {}, Total Similarity Score: {}'.format(title, summaries[title], genres[title], ratings[title], runtimes[title], actors[title], votes[title], years[title], str(round(100*score,4)) + " %") for title, score in result]

print(display("", "","fantasy","", [1958, 2019], 5))
print(display("", "", "","", [1958, 1962], 5))
print(display("the mindy project, grey's anatomy, house", "","", "",[1958, 2019], 5))
print(display("doctors, good doctor, doctor stranger", "", "","", [1958, 2019], 5))
print(display("", "","", "Shin-Hye Park", [1958, 2019], 5))
