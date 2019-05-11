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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
#for doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import random
nltk.download('punkt')
import pickle
import time


path = os.path.join(os.getcwd(),"app", "irsystem", "models", "cleaned_comprehensive_data.csv")
data = pd.read_csv(path)
num_dramas = len(data)

path3 = os.path.join(os.getcwd(),"app", "irsystem", "models",'korean_data.csv')
non_processed_k_data = pd.read_csv(path3)

path4 = os.path.join(os.getcwd(),"app", "irsystem", "models",'american_data.csv')
non_processed_a_data = pd.read_csv(path4)
                                 
kdrama_index_to_name = non_processed_k_data['Title'].to_dict()
kdrama_name_to_index = {v: k for k, v in kdrama_index_to_name.items()}

ustv_index_to_name = non_processed_a_data['Title'].to_dict()
ustv_name_to_index = {v: k for k, v in ustv_index_to_name.items()}



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

def preprocess_text(text):

    text = str(text)
    text = cleanhtml(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    text = text.strip()
    #text = stem(text)
    return text

def remove_stop(list):
    stop_words = set(stopwords.words('english'))
    no_stop = [w for w in list if not w in stop_words]
    return no_stop

def doc2vec():
    
    data_series = non_processed_k_data['Summary']
    data_list = pd.Series.tolist(data_series)
    
    processed_all = [preprocess_text(entry) for entry in data_series]
    processed_token = [word_tokenize(token.lower()) for token in processed_all]
    processed_no_stop = [remove_stop(entry) for entry in processed_token]
    
    processed_sample = random.sample(processed_no_stop, 1000)
        
    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(processed_sample)]
    
    max_epochs = 50
    vec_size = 50
    alpha = 0.025
    
    model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
    model.build_vocab(tagged_data)
    
    for epoch in range(max_epochs):
        start = time.time()
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
        end = time.time()
        print(end-start)
    model.save("d2v.model")
    print("Model Saved")
    
    
def test_model():
    
    data_series = data[:10]
    
    model= Doc2Vec.load("d2v.model")
    
    ranks = []
    second_ranks = []
    for doc_id, entry in enumerate(data_series):
        processed = [preprocess_text(entry)]
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(processed)]
        
        print(doc_id)
        inferred_vector = model.infer_vector(tagged_data[0].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
        
    print('Document ({}): «{}»\n'.format(doc_id, ' '.join(tagged_data[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    
    
def create_embeddings():
    model= Doc2Vec.load("d2v.model")
    #Itterate through the documents to get embeddings
    
    kdata_series = non_processed_k_data['Summary']
    kdata_list = pd.Series.tolist(data_series)
    kprocessed_all = [preprocess_text(entry) for entry in data_list]
    ktagged_docs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(kprocessed_all)]
    
    kembeddings = {}
    for idx, summary in enumerate(ktagged_docs):
        if idx in kdrama_index_to_name:
            vec = model.infer_vector(summary.words)
            kembeddings[kdrama_index_to_name[idx]] = vec
            
    np.save('k_emb.npy', kembeddings)
    #data2 = np.load('k_emb.npy')
    #with open('kdrama_emb.txt', 'w', encoding='utf-8-sig') as fp:
        #fp.write(repr(kembeddings))

    adata_series = non_processed_a_data['Summary']
    adata_list = pd.Series.tolist(data_series)
    aprocessed_all = [preprocess_text(entry) for entry in data_list]
    atagged_docs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(aprocessed_all)]
    
    aembeddings = {}
    for idx, summary in enumerate(atagged_docs):
        if idx in ustv_index_to_name:
            vec = model.infer_vector(summary.words)
            aembeddings[ustv_index_to_name[idx]] = vec
            
    np.save('a_emb.npy', aembeddings)
            
    #with open('ustv_emb.txt', 'w') as fp:
        #fp.write(repr(aembeddings))
        
def create_sim_matrix():
    a_emb = np.load('a_emb.npy')
    k_emb = np.load('k_emb.npy')
    
   # a_emb_sim = np.zeros((767, 767))
    emb_sim = np.zeros((1466+767, 1466))
    
    for i in range (1466+767):
        
        for j in range(i, 1466):
            name1 = kdrama_index_to_name[j]
            vec1 = k_emb[()][name1]
            if i == j :
                emb_sim[i][j] = 0
                
            if i < 1466:
                
                name2 = kdrama_index_to_name[j]
                vec2 = k_emb[()][name2]
                
                dot = np.multiply(vec1, vec2)
                emb_sim[i][j] = np.sum(dot)
                emb_sim[j][i] = np.sum(dot)
            
            if i > 1466:
                name2 = ustv_index_to_name[i-1466]
                vec2 = a_emb[()][name2]
                
                dot = np.multiply(vec1, vec2)
                emb_sim[i][j] = np.sum(dot)
            
    np.save('emb_sim_matrix_1.npy', emb_sim)


    
    
    
    
    
    
    
    
    
    
    
        