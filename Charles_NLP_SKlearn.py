# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 23:50:38 2016

@author: Charles
"""

#%%

#########################
#Read dataset reviews##
#######################
import os
import pandas as pd

os.chdir('C:\\Users\\Charles\\OneDrive\\capstoners')
########################
# original files code ##
########################
#rev = pd.read_csv('yelp_academic_dataset_review.csv')
#bsn = pd.read_csv('yelp_academic_dataset_business.csv')
#print rev.head(10)

##################
# Queryied code ##
##################
data = pd.read_csv('review_123.csv')

#%%
###########################
# Preprocess Review text ##
###########################
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

porter = nltk.PorterStemmer()

#Define tokenizer function

def my_tokenizer(w):
    rev = nltk.regexp_tokenize(re.sub(r'\d+', '', w), r'\w+')
    words = [porter.stem(i) for i in rev]
    return words

tfv = TfidfVectorizer(min_df = 3, 
                      max_features = None, 
                      strip_accents = 'ascii', 
                      analyzer = 'word', 
                      tokenizer = my_tokenizer,
                      #token_pattern  = r'\w{1,}',
                      #ngram_range = [1], 
                      use_idf = 1, 
                      smooth_idf = 1, 
                      sublinear_tf = 1, 
                      stop_words = 'english'
                      )

text_tfv = tfv.fit_transform(data['text'])

#%%
import pickle
pickle.dump(tfv, open('vectorizer.p', 'wb'), -1)                      

#%%
# Create dictionary from words

tokens = [my_tokenizer(w.decode('utf-8')) for w in data['text']]


#%%
# Convert to Corpus that can feed into gensim's LDA model
import gensim
reviews = gensim.corpora.Dictionary(tokens)
revCorp = gensim.matutils.Sparse2Corpus(text_tfv, documents_columns = True)

#%%
# Train LDA model
revLDA = gensim.models.ldamodel.LdaModel(revCorp, 10, reviews)
print revLDA.print_topics(num_topics = 5, num_words = 5)