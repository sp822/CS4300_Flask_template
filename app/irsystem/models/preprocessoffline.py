from __future__ import print_function
import re
import string
from operator import itemgetter
from nltk.stem import PorterStemmer
import os
import numpy as np
import pandas as pd
from collections import Counter
import math
from nltk.stem import PorterStemmer
import json


def cleanhtml(raw_html):
    """Cleans raw html from a string.
    Params: {text: String}
    Returns: String
    """
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
    """Preprocesses summary text.
    Params: {text: String}
    Returns: String
    """
    text = str(text)
    text = cleanhtml(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = stem(text)
    return text

def build_inverted_index_and_regular_index(data):
    """ Builds an inverted index and regular index from the dramas
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

def compute_idf(inv_idx, n_dramas):
    """ Computes dictionary of idf scores
    """
    idf_dict = {}
    for word in inv_idx.keys():
        DF = len(inv_idx[word])
        idf_dict[word] = math.log(n_dramas/(1+DF),2)
    return idf_dict

def preprocess(data, summary_dict, actors):
    """ Preprocesses data frame and adds new summary information.
    """
    new_data = data
    new_data = new_data.drop(columns = 'Unnamed: 0')
    actors_dict = {}
    genres_dict = {}
    years_dict = {}
    actor_col = {}
    actors.columns = ["Actor_Name", "Drama"]
    title_series = data['Title']
    for (index,row) in actors.iterrows():
        value = row["Drama"]
        value = str(value).strip("[]")
        value = value.split("', '")
        value = [v.replace("'", "").strip() for v in value]
        for drama in value:
            if drama in title_series.values:
                idx1 = title_series.str.contains(str(drama), regex=True)
                idx = [i for i in idx1.index if idx1[i]]
                for i in idx:
                    if i in actor_col.keys():
                        actor_col[i] += str(row["Actor_Name"]) + ", "
                    else:
                        actor_col[i] = str(row["Actor_Name"]) + ", "
    actor_col_new = actor_col
    for (idx, row) in actor_col.items():
        if len(actor_col[idx]) > 0:
            actor_str = str(actor_col[idx])[:-2]
            actor_col_new[idx] = actor_str
        print(actor_col_new[idx])
    new_data['Actors']= pd.DataFrame.from_dict(actor_col_new, orient='index')
    data['Actors']= pd.DataFrame.from_dict(actor_col_new, orient='index')
    print(data['Actors'])
    for (index,value) in data.iterrows():
        rating = value['Rating']
        if not math.isnan(rating):
            new_data['Rating'].iloc[index] = float(rating)
        summary = " ".join(summary_dict[str(index)])
        summary1 = str(value['Summary'])
        if summary1 != "nan" and summary1:
            summary = summary1 + summary
        summary = preprocess_text(summary)
        new_data['Summary'].iloc[index] = summary
        genre = str(value['Genre'])
        genre = preprocess_text(genre)
        new_data['Genre'].iloc[index] = genre
        genres_dict[index] = genre.split()
        title = value['Title']
        title = title.lower()
        title = title.strip()
        title = re.sub(r'\([^)]*\)', '', title)
        title = title.replace('â€™', "'")
        title = title.replace('â€', "'")
        title = title.replace ("'“", "")
        new_data['Title'].iloc[index] = title
        year = str(value['Year'])
        year_list = []
        if year != 'nan' and year:
            year = year[:-2]
            if len(year) > 4:
                start = int(year[:4])
                end = int(year[4:])
                for i in range(start,end+1,1):
                    year_list.append(i)
                year = year[:4] + ', ' + year[4:]
            else:
                year_list = [int(year[:4])]
            new_data['Year'].iloc[index] = year
        years_dict[index] = year_list
        runtime = str(value['Runtime'])
        if runtime != 'nan' and runtime:
            runtime = int(re.sub("[^0-9]", "", runtime))
            new_data['Runtime'].iloc[index] = runtime
        votes = value['Votes']
        if not math.isnan(votes):
            new_data['Votes'].iloc[index] = int(votes)
        actors = str(value['Actors'])
        actors = actors.split(", ")
        actors_dict[index] = actors

    return (new_data, genres_dict, actors_dict, years_dict)

def build_tfidf_matrix(data_index, n_dramas,vocab_to_index, idf_dict):
    """Returns a numpy tfidf matrix giving the tfidf score of
        a term within a drama

    Params: {drama1: String,
            drama2: String,
            input_doc_mat: Numpy Array,
            drama_name_to_index: Dict}
    Returns: Float (Cosine similarity of the two drama transcripts.)
    """
    result = np.zeros((n_dramas, len(vocab_to_index.keys())))
    for idx in range(n_dramas):
        sum_terms = 1
        for (word, count) in data_index[idx]:
            sum_terms = sum_terms + count
            idx2 = vocab_to_index[word]
            idf = idf_dict[word]
            tfidf = idf*count
            result[idx, idx2] = tfidf
        result[idx,:]=np.divide(result[idx,:],sum_terms)
    return result

def inclusion_matrix(dictionary):
    unique = set()
    for index in dictionary.keys():
        unique.update(dictionary[index])
    unique_index_to_name = dict(enumerate(list(unique)))
    unique_name_to_index = {v: k for k, v in unique_index_to_name.items()}
    matrix = np.zeros((len(dictionary.keys()), len(unique)))
    for index in unique_index_to_name.keys():
        for index2 in dictionary.keys():
            value = dictionary[index2]
            if unique_index_to_name[index] in value:
                matrix[int(index2),index] = 1
            else:
                matrix[int(index2),index] = 0
    return (matrix, unique_name_to_index)

def year_inclusion_matrix(year_dict):
    unique = range(1958,2020,1)
    unique_index_to_name = dict(enumerate(list(unique)))
    unique_name_to_index = {v: k for k, v in unique_index_to_name.items()}
    matrix = np.zeros((len(year_dict.keys()), len(unique)))
    for index, year in unique_index_to_name.items():
        for index2,value in year_dict.items():
            if year in value:
                matrix[int(index2),int(index)] = 1
            else:
                matrix[int(index2),int(index)] = 0
    return (matrix, unique_name_to_index)


def get_sim(drama1, drama2, input_doc_mat, drama_name_to_index):
    """Returns a float giving the cosine similarity of
       the two drama transcripts.

    Params: {drama1: String,
             drama2: String,
             input_doc_mat: Numpy Array,
             drama_name_to_index: Dict}
    Returns: Float (Cosine similarity of the two drama transcripts.)
    """
    idx1 = drama_name_to_index[drama1]
    idx2 = drama_name_to_index[drama2]
    drama1 = input_doc_mat[idx1,]
    drama2 = input_doc_mat[idx2,]
    if np.linalg.norm(drama1)* np.linalg.norm(drama2) != 0:
        sim = np.dot(drama1, drama2)/(np.linalg.norm(drama1)* np.linalg.norm(drama2))
    else:
        sim = np.dot(drama1, drama2)
    return sim

def build_drama_sims_cos(n_drama, drama_index_to_name, input_doc_mat, drama_name_to_index, input_get_sim_method):
    """Returns a drama_sims matrix of size (num_dramas,num_dramas) where for (i,j):
        [i,j] should be the cosine similarity between the drama with index i and the drama with index j

    Params: {n_drama: Integer,
             drama_index_to_name: Dict,
             input_doc_mat: Numpy Array,
             drama_name_to_index: Dict,
             input_get_sim_method: Function}
    Returns: Numpy Array
    """
    result = np.zeros((n_drama, n_drama))
    for i in range(n_drama):
        for j in range(n_drama):
            if i == j:
                result[i,j] = 0
            else:
                drama1 = drama_index_to_name[i]
                drama2 = drama_index_to_name[j]
                result[i,j] = input_get_sim_method(drama1, drama2, input_doc_mat, drama_name_to_index)
    return result
actors = pd.read_csv(os.path.join("actors.csv"))
korean_data = pd.read_csv(os.path.join("korean_data.csv"))
n_dramas_korean = len(korean_data)
with open('korean_summaries.json') as f:
    korean_summaries = json.load(f)
korean_packed = preprocess(korean_data, korean_summaries, actors)
korean_df = korean_packed[0]
genres_dict = korean_packed[1]
actors_dict = korean_packed[2]
years_dict = korean_packed[3]
with open('genres_dict.json', 'w') as fp:
    json.dump(genres_dict, fp)
with open('actors_dict.json', 'w') as fp2:
    json.dump(actors_dict, fp2)
with open('years_dict.json', 'w') as fp3:
    json.dump(years_dict, fp3)
genre_inclusion_unpacked = inclusion_matrix(genres_dict)
genre_inclusion_matrix = genre_inclusion_unpacked[0]
genre_name_to_index = genre_inclusion_unpacked[1]
actors_inclusion_unpacked = inclusion_matrix(actors_dict)
actors_inclusion_matrix = actors_inclusion_unpacked[0]
actors_name_to_index = actors_inclusion_unpacked[1]
years_inclusion_unpacked = year_inclusion_matrix(years_dict)
years_inclusion_matrix = years_inclusion_unpacked[0]
years_name_to_index = years_inclusion_unpacked[1]

np.save(file = os.path.join('genre_inclusion_matrix.npy'), arr = genre_inclusion_matrix)
np.save(file = os.path.join('actors_inclusion_matrix.npy'), arr = actors_inclusion_matrix)
np.save(file = os.path.join('years_inclusion_matrix.npy'), arr = years_inclusion_matrix)

with open('genre_name_to_index.json', 'w') as fp:
    json.dump(genre_name_to_index, fp)
with open('actors_name_to_index.json', 'w') as fp2:
    json.dump(actors_name_to_index, fp2)
with open('years_name_to_index.json', 'w') as fp3:
    json.dump(years_name_to_index, fp3)
korean_data['Actors'] = korean_df['Actors']
korean_data.to_csv(os.path.join('korean_data.csv'))
"""
korean_data.to_csv(os.path.join('cleaned_korean_data.csv'))

american_data = pd.read_csv(os.path.join("American_data.csv"))
n_dramas_american = len(american_data)
with open('American_summaries.json') as f:
    american_summaries = json.load(f)
american_packed = preprocess(american_data, american_summaries)
american_data = american_packed[0]
american_data.to_csv(os.path.join('cleaned_american_data.csv'))
comprehensive_data = pd.concat([korean_data, american_data])
comprehensive_data = pd.read_csv(os.path.join('cleaned_comprehensive_data.csv'))
comprehensive_data = comprehensive_data.reset_index(drop=True)
comprehensive_data.to_csv(os.path.join('cleaned_comprehensive_data.csv'))
res = build_inverted_index_and_regular_index(comprehensive_data)
inv_index = res[0]
reg_index = res[1]
num_dramas = n_dramas_korean + n_dramas_american
idf_dict = compute_idf(inv_index, num_dramas)
index_to_vocab = {i:v for i, v in enumerate(inv_index.keys())}
vocab_to_index = {v: k for k, v in index_to_vocab.items()}
doc_by_vocab = build_tfidf_matrix(reg_index, num_dramas,vocab_to_index, idf_dict)
drama_index_to_name = comprehensive_data['Title'].to_dict()
drama_name_to_index = {v: k for k, v in drama_index_to_name.items()}
drama_sims_cos = build_drama_sims_cos(num_dramas, drama_index_to_name, doc_by_vocab, drama_name_to_index, get_sim)
np.save(file = os.path.join('cosine_matrix.npy'), arr = drama_sims_cos)"""
