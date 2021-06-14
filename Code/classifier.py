# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:25:35 2021

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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer




 
# =============================================================================
# Code : BOITE NOIRE - REGRESSION LINEAIRE - PACKAGE SKLEARN
# =============================================================================

def classif2(df,taille_test,l):
    #Renomme les colonnes pour facilité la lecture
    df.set_axis(['sentence', 'label',"source"],  axis='columns', inplace=True)
    
   
    #Récupére mes phrases (mes variables explicatives )
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
    

    test_vectors = vectorizer.transform(l)
    preds = c.predict(test_vectors)
        
        
    
    

    
    
    return(preds)
