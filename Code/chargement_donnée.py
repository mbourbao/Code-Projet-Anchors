# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:31:01 2021

@author: 33651
"""
import numpy as np 
import pandas as pd 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
import os
import string 


from nltk import word_tokenize


os.chdir("C:/Users/33651/Documents/Projet_Anchors/CODE/")




############################
## CHARGEMENT DES DONNEES ##
############################

filepath_dict = {'yelp':   'C:/Users/33651/Documents/Projet_Anchors/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'C:/Users/33651/Documents/Projet_Anchors/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb':   'C:/Users/33651/Documents/Projet_Anchors/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)
df_list


df = pd.concat(df_list)
df.set_axis(['sentence', 'label',"source"],  axis='columns', inplace=True)
sentences = df['sentence'].values 
#Récupérer les labels
y = df['label'].values




train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(sentences, y, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)




vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)


c = sklearn.linear_model.LogisticRegression()
# c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
 
