# -*- coding: utf-8 -*-
"""
Created on Tue May 25 22:28:11 2021

@author: 33651
"""
import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
import string 
import matplotlib.pyplot as plt
import copy
os.chdir("C:/Users/33651/Documents/Projet_Anchors/CODE/")
from classifier import *
from perturbation import * 
import itertools
from select_cov import * 
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# ============================================================================#
#                                                                             #
# CODE POUR LE CALCUL DE LA COUVERTURE D UNE ANCRE                            #
#                                                                             #
# ============================================================================#



def cov_bis(ancre,data_sentence):
# =============================================================================
#     Pour calculer la couverture d une ancre, il faut compter combien de fois ce mot apparait dans nos phrases 
# =============================================================================
# =============================================================================
# 
# Toutes nos phrases pour pouvoir compter 
# =============================================================================
    cov_testl = list(data_sentence)
    
    cov = 0 #initialisation
    total = len(data_sentence) #nbre total d'instance
    present = [0]*len(ancre) #liste remplie de 0 de la taille de notre "ancre" exemple: {chien,chat} : present =[0,0]
    
    for j in range(len(data_sentence)): #on parcourt toute les instances
        present = [0]*len(ancre)  #liste remplie de 0 de la taille de notre "ancre" exemple: {chien,chat} : present =[0,0]
        for i in range(len(ancre)): #on parcourt l'ancre
            
            if ancre[i] in cov_testl[j] : #si le mot 1 et le mot 2 sont présent dans la phrase cov_testl[j] alors on aurait [1,1]
                present[i]=1 #L'instance remplie les conditions de la regle alors on a un vecteur remplie de 1
   
# =============================================================================
#     Si tous les mots de l'ancre sont présents dans la phrase cov_testl[j] alors 
#     l'indicateur "present" contient autant de 1 que de mot dans l'ancre 
# =============================================================================
      
        if present.count(1) == len(ancre): 
            cov = cov + 1 # +1 à chaque fois qu'on a une phrase avec les mots de l'ancre (on compte le nombre de phrase remplissant les cond de l'ancre)
    res = (cov/total)*100
    return(res)



    
    
