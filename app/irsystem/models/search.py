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
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

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
with open(os.path.join(os.getcwd(),"app", "irsystem", "models",'korean_user_reviews.json')) as fp4:
    reviews_dict = json.load(fp4)
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
replace_no_space = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
replace_with_space =re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
no_space = ""
blankspace = " "
def preprocess_reviews(datas):
    for index in datas:
        if len(datas[index])==0:
                datas[index] = []
        else:
            datas[index] = [replace_no_space.sub(no_space, line.lower()) for line in datas[index]]
            datas[index] = [replace_with_space.sub(blankspace, line) for line in datas[index]]

    return datas
datas = preprocess_reviews(reviews_dict)
nltk.download('movie_reviews')

def extract_features(word_list):
    return dict([(word, True) for word in word_list])
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')
features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                     'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                     'Negative') for f in negative_fileids]
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))
features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
classifier = NaiveBayesClassifier.train(features_train)

def get_sentiment(datas):
    sentiment_dict = {}
    for index in datas: 
        if len(datas[index])==0:
            sentiment_dict[index] = [{'Reviews': 'N/A', 'Sentiment': 'Positive', 'Probability': 0.0}]
        for review in datas[index]:
            probdist = classifier.prob_classify(extract_features(review.split()))
            if index in sentiment_dict:
                sentiment_dict[index].append({'Reviews': review, 'Sentiment': probdist.max(), 'Probability': round(probdist.prob(probdist.max()), 2)})
            else: 
                sentiment_dict[index] = [{'Reviews': review, 'Sentiment': probdist.max(), 'Probability': round(probdist.prob(probdist.max()), 2)}]
    return sentiment_dict

sentiment_dict = get_sentiment(datas)

def get_sentiment_score(sentiment_dict):
    sentiment_scores = {}
    for index in sentiment_dict:
        sum = 0
        for element in sentiment_dict[index]:
            if element['Sentiment'] == 'Negative':
                sum -= element['Probability']
            else:
                sum += element['Probability']
        sentiment_scores[index] = sum/len(sentiment_dict[index])
    return sentiment_scores
sentiment_scores = get_sentiment_score(sentiment_dict)

def best_match(actors_dict, genre_inclusion_matrix, actors_inclusion_matrix, years_inclusion_matrix, genre_name_to_index, actors_name_to_index, years_name_to_index, drama_sims_cos, data, drama_index_to_name, drama_name_to_index, dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, preferred_actors, preferred_time_frame, num_results):

    feature_list = ['Summary_Similarity', 'Actor_Similarity', 'Genre_Similarity', 'Network_Similarity','Year_Similarity', 'Sentiment_Analysis', 'Total']
    result = pd.DataFrame(0, index=np.arange(1466), columns=feature_list)
    genres = set()
    preferred_genres = [preprocess_text(value) for value in preferred_genres]
    genres.update(preferred_genres)
    years = preferred_time_frame
    start_year = int(years[0])
    end_year = int(years[1])
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
    preferred_actors =  preferred_actors.split(', ')
    best = best_match(actors_dict, genre_inclusion_matrix, actors_inclusion_matrix, years_inclusion_matrix, genre_name_to_index, actors_name_to_index, years_name_to_index,drama_sims_cos, data, drama_index_to_name, drama_name_to_index,  dramas_enjoyed, dramas_disliked, preferred_genres, preferred_network, preferred_actors, preferred_time_frame, num_results)
    result = list(zip(best['Drama_Title'], best["Total"]))
    titles = {}
    summaries = {}
    genres = {}
    ratings = {}
    runtimes = {}
    networks = {}
    actors = {}
    votes = {}
    years = {}
    for title, score in result:
        idx = drama_name_to_index_unprocess[title]
        summary = str(non_processed_data['Summary'].loc[idx])
        if summary != "nan":
            summaries[title] = summary
        else:
            summaries[title] = ""
        genre = str(non_processed_data['Genre'].loc[idx])
        if genre != "nan":
            genres[title] = genre
        else:
            genres[title] = ""
        rating = str(non_processed_data['Rating'].loc[idx])
        if rating != "nan":
            ratings[title] = rating
        else:
            ratings[title] = ""
        runtime = str(non_processed_data['Runtime'].loc[idx])
        if runtime != "nan":
            runtimes[title] = rating
        else:
            runtimes[title] = ""
        network = str(non_processed_data['Network'].loc[idx])
        if network != "nan":
            networks[title] = network
        else:
            networks[title] = ""
        actor = str(non_processed_data['Actors'].loc[idx])
        if actor != "nan":
            actors[title] = actor
        else:
            actors[title] = ""
        vote = str(non_processed_data['Votes'].loc[idx])
        if vote != "nan":
            votes[title] = vote
        else:
            votes[title] = ""
        year = str(data['Year'].loc[idx])
        if year != "nan":
            years[title] = year
        else:
            years[title] = ""
    return ['Drama Title: {},  Summary: {},  Genre: {}, Rating: {}, Runtime: {}, Network: {}, Actors: {}, Votes: {}, Years: {}, Total Similarity Score: {}'.format(title, summaries[title], genres[title], ratings[title], runtimes[title], networks[title], actors[title], votes[title], years[title], score) for title, score in result]

# print(display("The Mindy Project, Doctor Stranger, Doctors, House, Grey's Anatomy", "City Hunter, Game of Thrones, New Girl, Nikita", "medical, romance, comedy", "", "Park Shin-Hye","2010-2015", 10))
