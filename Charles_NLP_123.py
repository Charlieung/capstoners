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
#data[data['text'].isnull()]
# index 238239 has missing Values
#data.drop(238239, inplace = True)


#%%
###########################
# Preprocess Review text ##
###########################
import re
import nltk

#porter = nltk.PorterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
# Unicode conversion, Lowercase, Strip Numbers, Strip Punctuation and Tokenize
rev = [nltk.regexp_tokenize(re.sub(r'\d+', '', w.decode('utf-8').lower()), r'\w+') for w in data['text']]
#Lemmatize
words = [[lemma.lemmatize(w) for w in lis if w not in stopwords] for lis in rev]

#%%
'''
#Save the words list because lemmatization takes forever
import pickle
pickle.dump(words, open("AZ_lemma.p", "wb"))
'''
#%%
'''
#load words if starting a new instance
import pickle #if code is starting from the load to save time
words = pickle.load(open("AZ_lemma.p", "wb"))
'''
#%%
################################
# Create Document Term Matrix ##
################################

from gensim import corpora, models

#Create corpus from words within docs.
reviews = corpora.Dictionary(words)
#reviews.save('reviews.dict')

#%%
# create list of bag of words
revBoW = [reviews.doc2bow(w) for w in words]

revTfIdf = models.tfidfmodel.TfidfModel(revBoW)
# save corpus in Blei's lda-c format because creating corpus takes forever
#corpora.BleiCorpus.serialize('revCorp.lda-c', revCorp)

#%%
#load corpora if starting a new instance
'''
from gensim import corpora #if code is starting from the load to save time
revCorp = corpora.BleiCorpus.load('revCorp.lda-c')
'''
#%%
#################################
# LDA algorithm with 10 topics ##
#################################
from gensim import models

revLDA = models.LdaModel(revTfIdf[revBoW], num_topics = 10, id2word = reviews, passes = 10)
#%%
#Save the model, because it takes forever
import pickle
pickle.dump(revLDA, open('revLDA123.model', 'wb'))

#%%
import pickle
revLDA = pickle.load(open('revLDA123.model'))
#%%
#Show what the 10 topics are, and the words that comprise them
for i in range(10):
    print revLDA.print_topics(num_topics=10, num_words=5)[i]



#%%
#Reverse transform the LDA model back onto the reviews
#Get probabilities for the ten topics
revTopics = [revLDA[reviews.doc2bow(w)] for w in words]

#%%
#Move into dataframe
topicsTab = pd.DataFrame(columns = (0,1,2,3,4,5,6,7,8,9))

for row in range(len(revTopics)):
    topicsTab.loc[row] = [dict(revTopics[row]).get(i,0) for i in range(10)]

#%%
#Convert to CSV for later loading
topicsTab.to_csv('AZtopicsTab.csv')
#%%

#Create total Food %
food = topicsTab[0]

for col in [1,2,3,4,6,7,8]:
    food = food.copy() + topicsTab[col]
    
#Create total Service %
service = topicsTab[5] + topicsTab[9]

#Reproportion food/service to sum = 1

adjust = 1 / (food + service)
food = food.copy() * adjust * data['stars.y']
service = service.copy() * adjust * data['stars.y']

#%%
# merge table

data['food.stars'] = food
data['service.stars'] = service
#data['tokens'] = words
data.to_csv('AZdata.csv')