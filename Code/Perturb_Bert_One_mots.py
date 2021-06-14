# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:08:47 2021

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



torch = torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## unmask : fonction prise du GitHub de Ribeiro 
## Cette fonction prend en paramètre une phrase avec des mots masqués
## Elle retourne une liste contenant autant de sous-liste que de mot masqué
## Ces sous-liste sont composées d'une liste de mot pouvant remplacer le mot masqué générer par l'algorithme BERT, 
## Chaque mot possède un identifiant numérique. (c'est ce qui permettra le piocher aléatoire un mot de la liste pour remplacer le mot masqué)

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
# Cette fonction est permet de générer N phrases avec un seul mot masqué.
# =============================================================================

def generation_perturb_bert_one(instance,mes_mot,nbre_perturb,MASK="[MASK]"): #Prend en parametre la phrase à expliquer, l'ancre et le nbre d'instance qu'on veut generer
    sauv_phrase = [] #liste qui va contenir les instances perturbés
    
    
    ph_expli_tok = word_tokenize(instance.lower()) #Découpage de la phrase et passage en minuscule (pour éviter les erreurs dans l'application du calcul de la couverture par ex)
    les_candidats= copy.copy(ph_expli_tok) 
    les_candidats = list(set(les_candidats)) #Supprime les doublons 
    m=0  #compteur initialiséà 0, qui va permettre de gerer le nombre d'instance à generer
    mes_indice = [] #liste qui va contenir l'indice des mots qu'on fixe : pour fixer
    ind_supp = []
    for i in range(len(mes_mot)): #mes_mot = c'est la liste de l'ancre qu'on étudie 
            if mes_mot[i] in les_candidats : 
                les_candidats.remove(mes_mot[i]) #On ne peut pas modifier les mots de l'ancre (mot fixe), on les supprime donc des candidats des mots à masquer
# =============================================================================
#     print("LES CANDIDATS SONT !!!!!!!!!!!!!!!!",les_candidats)     
# =============================================================================
    while m <= nbre_perturb :  #tant que on est pas arrivé au nbre de perturbation qu'on desire on continue
        
        new_phrase = copy.copy(ph_expli_tok) #on copie la phrase à perturber pour pas modifier la phrase initial
        
    
        num_change = 1 #combien d'indice on modifie, dans ce cas on masque un seul mot de la phrase
        change = np.random.choice(les_candidats, num_change,replace=False) #on tire au hasard le mot qu'on va masqué
# =============================================================================
#         print("change est !!!!!!!!!!!!!",change)   
#         print(ph_expli_tok)
#         print("LA PHRASE EST ",new_phrase)
# =============================================================================
        indice_change = ph_expli_tok.index(change) #on récupère l'indice du mot qui va être masqué
         
        new_phrase[indice_change] = str(MASK) #on le masque avec l'étiquette [MASK]
        
# =============================================================================
#         print("!!!!!!!!!!!!!", new_phrase)
#         
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

## Fonction qui prend en paramètre l'instance masqué et retourne une phrase avec le mot démasqué (tirer au hasard et génèrer grâce à BERT

def gener_perturb_bert_one(phrase_mask,phrase_entiere,torch):
    ## Préparation de BERT
     # découpage des mots
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Méthode MLM => Bert analyse les mots après et avant le mot masqué pour génèrer un mot qui s'adapte au contexte de la phrase
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(phrase_mask, return_tensors="pt") #On fournit au modèle en entrée le mot masqué
    labels = tokenizer(phrase_entiere, return_tensors="pt")["input_ids"] #Phrase entière pour connaitre le sentiment 

    outputs = model(**inputs, labels=labels) #Pour la sortie de la méthode MLM
    loss = outputs.loss
    logits = outputs.logits
    torch = torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')

    bert.to(device)
    bert.eval()
        #Application de la fonction qui génère une liste de proposition de mot qui est susceptible d'etre le mot masqué 
    list_mot = unmask(phrase_mask,torch,bert_tokenizer,bert)
     # list_mot contient donc une liste de mot susceptible de remplacer le mot masqué et chaque mot est affilié à un numéro (identifiant)
        # list_mot[0][0] = liste contenant le nom des mots et list_mot[0][1] liste des identifiants 
        # On tire au hasard l'identifiant du mot qui va remplacer le mot masqué
    choice_mot = np.random.choice(list_mot[0][1],1,replace=False )
    for k in zip(list_mot[0][0],list_mot[0][1]): #On récupère le nom du mot associé à l'identifiant qui a été tiré au hasard
        if k[1]== choice_mot : 
            mot = k[0]
    ph_expli_tok= word_tokenize(phrase_mask) # On découpe la phrase pour pouvoir faire un remplacement rapide 
# =============================================================================
#     print(ph_expli_tok)
# =============================================================================
    index_mask = ph_expli_tok.index("MASK") #On récupère l'indice du mot masqué pour réaliser le remplacement
# =============================================================================
#     print("le type est !!" ,type(mot))
# =============================================================================
    ph_expli_tok[index_mask] = mot #on remplace
    ph_expli_tok.remove("[") # lorsqu'on a masqué le mot, on a ajouté une etiquette : [MASK] lors du découpage les '[' on était séparer il faut donc les enlevé 
    ph_expli_tok.remove("]")
# =============================================================================
#     print("le mot est",mot)
# =============================================================================
    phr = " ".join(ph_expli_tok) #On recompose la phrase
    
     
    return(phr)

# =============================================================================
# CODE POUR L'EXEMPLE DES PERTURBATIONS The reception has been generally good '
# =============================================================================
# # =============================================================================
# 
# exemple1 = "The [MASK] have been generally good."
# exemple2 = "The reception have [MASK] generally good." 
# exemple3 = "[MASK] reception have been generally good."
# exemple4 = "The reception have been [MASK] good"
# 
# 
# phrase_demask1 = gener_perturb_bert_one("The [MASK] have been generally good." ,"The reception has been generally good." ,torch)
# phrase_demask2 = gener_perturb_bert_one("The reception have [MASK] generally good." ,"The reception has been generally good." ,torch)
# phrase_demask3 = gener_perturb_bert_one("[MASK] reception have been generally good." ,"The reception has been generally good." ,torch)
# phrase_demask4 = gener_perturb_bert_one("The reception have been [MASK] good." ,"The reception has been generally good." ,torch)
# =============================================================================
#generation_perturb_bert_one("the good job",["job"],10,"[MASK]")
