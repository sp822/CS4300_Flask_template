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
drama_name_to_index = {v.strip(): k for k, v in process_dict.items()}
print(drama_name_to_index)
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
with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'reviews_sentiment.json')) as fp5:
    reviews_sentiment_dict = json.load(fp5)
j = [0]
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
    dramas_enjoyed = [drama.lower().strip() for drama in dramas_enjoyed]
    dramas_disliked = [drama.lower().strip() for drama in dramas_disliked]
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
    length = {int(idx):len(str(value).split(", ")) for idx, value in non_processed_data["Actors"].items()}
    s = pd.Series(length)
    length_arr = s.values
    d2 = {int(k):float(v) for k, v in sentiment_dict.items()}
    result['Sentiment_Analysis']= pd.DataFrame.from_dict(d2, orient='index')

    for drama in dramas_enjoyed:
        if drama in drama_name_to_index.keys():
            index = drama_name_to_index[drama]
            sim = drama_sims_cos[index,:1466]
            result['Summary_Similarity']+= pd.Series(sim)
            sim_doc = emb_sim_matrix[index]
            result['Embedding_Similarity']+= pd.Series(sim_doc)

    for drama in dramas_disliked:
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
    actor_sim = np.zeros((1466,))
    for actor in preferred_actors:
        if actor in actors_name_to_index.keys():
            index = actors_name_to_index[actor]
            actor_sim+= actors_inclusion_matrix[:,index]
    actor_union = length_arr - actor_sim
    actor_union+=len(preferred_actors)
    actor_sim = actor_sim/(actor_union+1)
    result['Actor_Similarity'] = actor_sim
    min_summary = result['Summary_Similarity'].min()
    min_embedding = result['Embedding_Similarity'].min()
    if min_summary < 0:
        result['Summary_Similarity'] = result['Summary_Similarity'] + min_summary*-1
        result['Summary_Similarity'] = result['Summary_Similarity']/(1+min_summary*-1)
    if min_embedding < 0:
        result['Embedding_Similarity'] = result['Embedding_Similarity'] + min_embedding*-1
        result['Embedding_Similarity'] = result['Embedding_Similarity']/(1+min_summary*-1)
    if result['Embedding_Similarity'].max() != 0 and result['Embedding_Similarity'].max() > 1:
        result['Embedding_Similarity'] = result['Embedding_Similarity']/(result['Embedding_Similarity'].max())
    if result['Summary_Similarity'].max() > 1:
        result['Summary_Similarity'] = result['Summary_Similarity']/(result['Summary_Similarity'].max())
    result['Total'] = round(result['Embedding_Similarity']*.2 + result['Summary_Similarity']*.4 + result['Actor_Similarity']*.1 + result['Genre_Similarity']*.2,4)
    result = result.sort_values(by='Total', ascending=False)
    index1 = years_name_to_index[str(start_year)]
    index2 = years_name_to_index[str(end_year)]
    if not(str(start_year) == "1958" and str(end_year) == "2019"):
        for idx, res in result.iterrows():
            mat = years_inclusion_matrix[idx, index1:index2]
            mat2 = years_inclusion_matrix[idx, :]
            if sum(mat) == 0:
                result = result[result.index != idx]

    for idx, res in result.iterrows():
        title = drama_index_to_name[idx]
        title = title.lower().strip()
        if title in dramas_enjoyed or title in dramas_disliked:
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

    print(best)

    network_list = ['Channel A','Naver tvcast','Mnet', 'tvN', 'KM' 'Onstyle', 'SBS' 'Netflix', 'KBS', 'MBC', 'DramaX', 'MBN', 'Oksusu',
    'UMAX', 'O’live', 'CGV', 'TBS', 'Sohu TV', 'Tooniverse', 'DRAMAcube', 'KBSN', 'E-Channel', 'Fuji TV', 'OCN', 'Yunsae University',
    'EBS', 'tvN', 'DramaH','Onstyle', 'CSTV', 'jTBC', 'Viki']
    result = list(zip(best['Drama_Title'], best['Total'],best["Sentiment_Analysis"],best['Embedding_Similarity'], best['Summary_Similarity'], best['Actor_Similarity'], best['Genre_Similarity']))
    titles = {}
    summaries = {}
    genres = {}
    ratings = {}
    runtimes = {}
    actors = {}
    votes = {}
    years = {}
    networks = {}
    sentiment_output = {}
    sentiment_reviews_output = {}
    """feature_list = ['Title','Summary','Genre', 'Rating', 'Runtime','Actors', 'Network', 'Votes', 'Year','Similarity_Score', 'Sentiment_Score']
    result_exp = pd.DataFrame(None, index=np.arange(num_results), columns=feature_list)"""
    i = 0
    for title, score, sentiment_score,_,_,_,_ in result:
        idx = drama_name_to_index_unprocess[title]
        summary = str(non_processed_data['Summary'].loc[idx])
        """result_exp['Summary'].iloc[i] = summary"""
        if summary != "nan":
            summaries[title] = summary
        else:
            summaries[title] = "No summary information is available."
        genre = str(non_processed_data['Genre'].loc[idx])
        """result_exp['Genre'].iloc[i] = genre"""
        if genre != "NaN":
            genre = genre.strip('[]')
            genre = genre.replace("'", "")
            genres[title] = genre
        else:
            genres[title] = "No genre information is available."
        rating = str(data['Rating'].loc[idx])
        """result_exp['Rating'].iloc[i] = rating"""
        if rating != "nan":
            ratings[title] = rating
        else:
            ratings[title] = "No rating information is available."
        runtime = str(non_processed_data['Runtime'].loc[idx])
        """result_exp['Runtime'].iloc[i] = runtime"""
        if runtime != "nan":
            runtimes[title] = rating
        else:
            runtimes[title] = "No runtime information is available."
        actor = str(non_processed_data['Actors'].loc[idx])
        """result_exp['Actors'].iloc[i] = actor"""
        if actor != "nan":
            actors[title] = actor
        else:
            actors[title] = "No actor information is available."
        network_loc = str(non_processed_data['Network'].loc[idx])
        network = ""
        for net in network_list:
            if net in network_loc:
                network = network + net + ", "
        if len(network) > 0:
            network = network[:-2]
        """result_exp['Network'].iloc[i] = network"""
        if network != "":
            networks[title] = network
        else:
            networks[title] = "No network information is available."
        vote = str(non_processed_data['Votes'].loc[idx])
        """result_exp['Votes'].iloc[i] = vote"""
        if vote != "nan":
            votes[title] = vote
        else:
            votes[title] = "No votes information is available."
        year = str(data['Year'].loc[idx])
        """result_exp['Year'].iloc[i] = year"""
        if year != "nan":
            years[title] = year
        else:
            years[title] = "No timeframe information is available."
        sentiment_dictionary = reviews_sentiment_dict[str(idx)]
        sentiment_output[title] = sentiment_dictionary["Predicted Sentiment"]
        sentiment_reviews_output[title] = sentiment_dictionary['Reviews']
        """result_exp['Title'].iloc[i] = title
        result_exp['Similarity_Score'].iloc[i] = score
        result_exp['Sentiment_Score'].iloc[i] = sentiment_score"""
        i+=1

    """j[0]+=1
    result_exp.to_csv(os.path.join("app", "irsystem", "models", 'test_results', str("result" + str(j[0])+ ".csv")))"""
    return ['{},  Summary: {},  Genre: {}, Rating: {}, Runtime: {}, Actors: {}, Votes: {}, Years: {},  Sentiment: {}, Sentiment Reviews: {}, Total Similarity Score: {}, Sentiment Score: {}, Embedding Score: {}, Summary Score: {}, Actor Score: {}, Genre Score: {}'.format(title, summaries[title], \
    genres[title], ratings[title], runtimes[title], actors[title], votes[title], years[title], sentiment_output[title], sentiment_reviews_output[title], round(100*score,4), round(100*sentiment_score,4), round(100*embedding_score,4), round(100*summary_score,4), \
    round(100*actor_score,4), round(100*genre_score,4)) for title, score, sentiment_score, embedding_score, summary_score, actor_score, genre_score in result]
"""
display("", "","fantasy","", [1958, 2019], 5)
display("", "","romantic","", [1958, 2019], 5)
display("", "","medical","", [1958, 2019], 5)

display("", "", "","", [1958, 1962], 5)
display("", "", "","", [1980, 2000], 5)
display("", "", "","", [2000, 2010], 5)

display("the mindy project, grey's anatomy, house", "","", "",[1958, 2019], 5)
display("doctors, good doctor, doctor stranger", "", "","", [1958, 2019], 5)
display("game of thrones, nikita, teen wolf", "", "","", [1958, 2019], 5)

display("","the mindy project, grey's anatomy, house","", "",[1958, 2019], 5)
display("","doctors, good doctor, doctor stranger", "","", [1958, 2019], 5)
display("","game of thrones, nikita, teen wolf", "","", [1958, 2019], 5)



display("","Confession", "", "", [1958, 2019], 21)"""
