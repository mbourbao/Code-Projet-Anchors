# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:52:16 2021

@author: 33651
"""
import numpy as np 
import pandas as pd 
import os
import string 
from nltk import word_tokenize
import copy


# =============================================================================
# 
# CODE : Fonction qui genere des pertubations par méthode déterministe 

# =============================================================================

def generation_perturb_bis4(ma_phrase,mes_mot,nbre_perturb,st): #Prend en parametre la phrase à expliquer, l'ancre et le nbre d'instance qu'on veut generer
    sauv_phrase = [] #liste qui va contenir les instances perturbés
    ph_expli_tok = ma_phrase  #phrase à expliquer celle qui va être perturbée
    
    m=0  #compteur initialiséà 0, qui va permettre de gerer le nombre d'instance à generer
    mes_indice = [] #liste qui va contenir l'indice des mots qu'on fixe : pour fixer
   
    for i in range(len(mes_mot)):
            mes_indice.append(ph_expli_tok.index(mes_mot[i])) #ajout des indices des mots à stabilisé
    #print("mes indices",mes_indice)
           
    while m <= nbre_perturb :  #tant que on est pas arrivé au nbre de perturbation qu'on desire on continue
        
        new_phrase = copy.copy(ph_expli_tok) #on copie la phrase à perturber pour pas modifier la phrase initial
        
        
        #decoupage liste
        list_choix = np.arange(len(new_phrase))
        list_choix = list(list_choix) #liste de tous les indices 
        
        for elem in mes_indice:
            if elem in list_choix : 
                list_choix.remove(elem) #Correspond à I, suppression des indices des mots ancrés
            
       # print("choix = " , list_choix)
       # print("mes indices =" , mes_indice)
       # print("longeur", len(ph_expli_tok))
        num_change = np.random.binomial(len(list_choix),.5) #combien d'indice on modifie
        change = np.random.choice(list_choix, num_change,replace=False)
       # print("change", change)   
        
        for k in change:
            new_phrase[k] = st
         
        
            #print("le mot selectionné est ", ph_expli_tok[i] , "new phrase", new_phrase)
        phr = " ".join(new_phrase) #on reconstitue la phrase
            #print("phrase est ", te)
        #print("phrase num",m,"est",phr)
        sauv_phrase.append(phr)
        
        m+=1
        #print("etape",m)
    
   # print("taille final",len(sauv_phrase))
    data = pd.DataFrame(data=sauv_phrase)
    return(data)

