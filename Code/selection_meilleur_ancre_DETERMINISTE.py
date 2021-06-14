# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:50:37 2021

@author: 33651
"""
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd 

import os
import string 

from nltk import word_tokenize

import copy

os.chdir("C:/Users/33651/Documents/Projet_Anchors/CODE/")
from classifier import *
from perturbation import * 
from couverture import * 

import itertools


 

# =============================================================================
# 
# CODE : Fonction qui renvoie la meilleur ancre avec une méthode de perturbation déterministe.
# 
# 
# =============================================================================

def meilleur_ancre_bis_new(df,phrase,nbre_perturb,st,senti,tau, vectorizer, classifier ):
    
    final = [] #liste qui va contenir le meilleur ancre + sa précision
    label = senti #Sentiment de la phrase qu'on veut expliquer (0 ou 1).
    
# =============================================================================
#     ################ PREPARATION DE LA PHRASE ####################
# =============================================================================

    ph_expli_tok= word_tokenize(phrase.lower()) #Decoupage de la phrase + mise en minuscule(pour eviter les erreurs)


    #Liste des commentaires 
    data_sentences = df['sentence'].values 
    

    les_candidats = ph_expli_tok
# =============================================================================
#     print("les_candidats",les_candidats)
# 
# =============================================================================
    meilleur_prec = 0 #Pour conserver l'ancre ayant la meilleur précision.
    prec_temp = 0 # Précision temporaire -> Pour comparer avec la meilleur.
    best_ancre = [] #On retient l'ancre ayant la meilleur precision.
    
    if len(les_candidats) <=1 : #S'il y a qu'une ancre possible pour être le candidat on le retourne.
        best_candidats = les_candidats
        perturb = generation_perturb_bis4(ph_expli_tok,best_candidats,nbre_perturb,st) #Perturbation déterministe 
        exp2 = np.array(perturb)
        exp3 = list(exp2)
        l =[]
        for j in range(len(exp2)):
            l.append(exp3[j][0])
        resultat = classif2(df,0.2,l) #Application du classifieur sur les perturbations
        meilleur_prec = len(resultat[resultat==label])/len(resultat) #calcul de la précision associé à l'unique candidat
        return(best_candidats, meilleur_prec)
        
    
    for i in range(1,len(les_candidats)): #Taille de la combinaison des ancres
        comb_temp = [] #L'ancre temporaire -> celle qu'on etudie
        
        for c in itertools.combinations(les_candidats,i): #Toutes les combinaisons possibles des candidats de taille i
            x= list(c) #On enregistre toutes les combinaisons possibles
            comb_temp.append(x)
            if meilleur_prec > 0.98 : #Si on est arrive à une précision supérieur à 0.98 = très satisfaisant et on evite de continuer pour rien (gain de temps)
# =============================================================================
#                  print('Anchor:', best_ancre , 'Precision:', meilleur_prec )
# =============================================================================
                 return(best_ancre, meilleur_prec)
                
        for p in range(len(comb_temp)) : #on va etudier la précision associé à chaque combinaison de taille i 
            
          #  print(comb_temp[p], "couverture ", test_cov)


            if p == 0 and len(best_ancre) == 0 : #tout premier pour initialiser les valeurs
                perturb = generation_perturb_bis4(ph_expli_tok,comb_temp[p],nbre_perturb,st) #Perturbation déterministe 
                exp2 = np.array(perturb)
                exp3 = list(exp2)
                l =[]
                for j in range(len(exp2)):
                     l.append(exp3[j][0])
                resultat = classif2(df,0.2,l) #Application du classifieur sur les perturbations
                meilleur_prec = len(resultat[resultat==label])/len(resultat) #calcul de la t
                best_ancre = comb_temp[p]
               
# =============================================================================
# =============================================================================
#                 print("La combinaison qu'on regarde est :", comb_temp[p])
#                 print("La précision associée est :", meilleur_prec)
# # =============================================================================
# =============================================================================
                #print(meilleur_prec)
            else: 
             #   print("com", comb_temp[p])
                perturb = generation_perturb_bis4(ph_expli_tok,comb_temp[p],nbre_perturb,st) #perturbation déterministe
     # transformation pour facilité la manipulation et le calcul de la précision.
                exp2 = np.array(perturb)
                exp3 = list(exp2)
                l =[]
                for j in range(len(exp2)):
                    l.append(exp3[j][0])
                resultat = classif2(df,0.2,l)
               # print(resultat)
                prec_temp = len(resultat[resultat==label])/len(resultat)
                
               
# =============================================================================
# =============================================================================
#                 print("La combinaison qu'on regarde est :",comb_temp[p])
# #                
#                 print("La précision associée est :",prec_temp)
# =============================================================================
#                 
# =============================================================================
                
                if prec_temp == meilleur_prec and prec_temp > tau and len(comb_temp[p]) <= len(best_ancre)  :
                    meilleur_prec = prec_temp
                    best_ancre = comb_temp[p]
                    
                elif prec_temp > meilleur_prec : 
                         meilleur_prec = prec_temp
                         best_ancre = comb_temp[p]
                        
                    
# =============================================================================
#                  
#     print('Anchor:', best_ancre , 'Precision:', meilleur_prec )
#  
# =============================================================================
    final.append(best_ancre)
    final.append(meilleur_prec)
    return final

