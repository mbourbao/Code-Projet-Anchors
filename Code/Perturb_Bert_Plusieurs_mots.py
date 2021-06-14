# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:32:22 2021

@author: 33651
"""
from transformers import DistilBertModel, DistilBertConfig
import ipywidgets
import IProgress
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch

import numpy as np 
import pandas as pd 


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV


import string 
from nltk import word_tokenize
import copy


# =============================================================================
# CODE PERTURBATION GENERATIVE - Plusieurs mots cachés 
# =============================================================================


# =============================================================================
# Fonction qui prend en entrée une phrase avec des mots masquées et en sortie nous renvoie une liste contenant 
# les mots qui ont été deviner ainsi que des numéros associées à chaque mot.
# =============================================================================
# =============================================================================
# Cette fonction vient du package Anchors (github Ribeiro)
# 
# =============================================================================


torch = torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def unmask(text_with_mask,torch,bert_tokenizer,bert):
        torch = torch
        tokenizer = bert_tokenizer
        model = bert
        encoded = np.array(tokenizer.encode(text_with_mask, add_special_tokens=True))
        input_ids = torch.tensor(encoded)
        masked = (input_ids == bert_tokenizer.mask_token_id).numpy().nonzero()[0]
        to_pred = torch.tensor([encoded], device=device)
        with torch.no_grad():
            outputs = model(to_pred)[0]
        ret = []
        for i in masked:
            v, top_preds = torch.topk(outputs[0, i], 500)
            words = tokenizer.convert_ids_to_tokens(top_preds)
            v = np.array([float(x) for x in v])
            ret.append((words, v))
        return ret



# =============================================================================
# 
# Fonction qui prend en paramètre la phrase à perturber, l'ancre qu'on veut tester, le nombre de perturbation : N ; qu'on souhaite générer 
# Elle renvoie N instances avec des mots cachés 
# 
# =============================================================================

def generation_perturb_bert(ma_phrase,mes_mot,nbre_perturb,MASK="[MASK]"): #Prend en parametre la phrase à expliquer, l'ancre et le nbre d'instance qu'on veut generer
    sauv_phrase = [] #liste qui va contenir les instances perturbés
    ph_expli_tok =  word_tokenize(ma_phrase)
    les_candidats= copy.copy(ph_expli_tok)
    
    m=0  #compteur initialiséà 0, qui va permettre de gerer le nombre d'instance à generer
    mes_indice = [] #liste qui va contenir l'indice des mots qu'on fixe : pour fixer
    ind_supp = []
    for i in range(len(mes_mot)):
            if mes_mot[i] in les_candidats : 
                while mes_mot[i] in les_candidats : 
                    les_candidats.remove(mes_mot[i])
# =============================================================================
#     print("LES CANDIDATS SONT !!!!!!!!!!!!!!!!",les_candidats)     
# =============================================================================
    while m <= nbre_perturb :  #tant que on est pas arrivé au nbre de perturbation qu'on desire on continue
        mes_indice = []
        new_phrase = copy.copy(ph_expli_tok) #on copie la phrase à perturber pour pas modifier la phrase initial
        
    
        num_change = np.random.binomial(len(les_candidats),.5) #combien d'indice on modifie
        change = np.random.choice(les_candidats, num_change,replace=False)
        for chan in change : 
            
            ind= ph_expli_tok.index(chan)
            mes_indice.append(ind)
            
        for num in mes_indice :  
            new_phrase[num] = str(MASK)
        
# =============================================================================
#         print("!!!!!!!!!!!!!", new_phrase)
# =============================================================================
        
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

# =============================================================================
# 
# FONCTION QUI DEMASQUE LES MOTS MASQUER -> Prend en paramètre une phrase avec des mots cachers, et renvoie une phrase sans mot caché 
# Utilise la fonction de Rebeiro.
# =============================================================================
def gener_perturb_bert(phrase_mask,phrase_entiere,torch,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(phrase_mask, return_tensors="pt")
    labels = tokenizer(phrase_entiere, return_tensors="pt")["input_ids"]

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    torch = torch
    device = device
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')

    bert.to(device)
    bert.eval()
    nbre_mask = phrase_mask.count('[MASK]') #nbre de mot à deviner 
    ind_mask_liste =[]
    ph_expli_tok= word_tokenize(phrase_mask)
# =============================================================================
#     print("LA PHRASE EST", ph_expli_tok)
# =============================================================================
    ind_mask_liste = [idx for idx,e in enumerate(ph_expli_tok) if e == "MASK"]
    
    list_mot = unmask(phrase_mask,torch,bert_tokenizer,bert)
    mot_deviner = []
    for dev_mot in range(len(list_mot)): 
        choice_mot = np.random.choice(list_mot[dev_mot][1],1,replace=False )
        for k in zip(list_mot[dev_mot][0],list_mot[dev_mot][1]):
            if k[1]== choice_mot : 
                mot = k[0]
                mot_deviner.append(mot)
    
# =============================================================================
#     print(" indice mask",ind_mask_liste,"mot deviner et"  , mot_deviner)
# =============================================================================
    for rempl in zip(mot_deviner,ind_mask_liste):
# =============================================================================
#         print("mot et indice", rempl)
# =============================================================================
        ph_expli_tok[rempl[1]] = rempl[0]
    
    ph_expli_tok = [ i for i in ph_expli_tok if i != "[" and i != "]"]

                    
                
            
    phr = " ".join(ph_expli_tok)
    
     
    return(phr)

