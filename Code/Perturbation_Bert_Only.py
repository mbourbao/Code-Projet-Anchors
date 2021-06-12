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

import os
import string 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
os.chdir("C:/Users/33651/Documents/Projet_Anchors/CODE/")
from classifier import *
from dict_local import * 
from perturbation import * 
from couverture import * 
from select_cov import * 
import itertools
from string import *

import copy
import itertools
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV


torch = torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import numpy as np
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
# Fonction qui génère N = (nbre_perturb) phrases avec un seul mot masqué
# =============================================================================

def generation_perturb_bert_one(instance,mes_mot,nbre_perturb,MASK="[MASK]"): #Prend en parametre la phrase à expliquer, l'ancre et le nbre d'instance qu'on veut generer
    sauv_phrase = [] #liste qui va contenir les instances perturbés
    
    
    ph_expli_tok = word_tokenize(instance.lower())
    les_candidats= copy.copy(ph_expli_tok)
    les_candidats = list(set(les_candidats)) #Supprime les doublons 
    m=0  #compteur initialiséà 0, qui va permettre de gerer le nombre d'instance à generer
    mes_indice = [] #liste qui va contenir l'indice des mots qu'on fixe : pour fixer
    ind_supp = []
    for i in range(len(mes_mot)): #mes_mot c'est la liste de l'ancre qu'on étudie 
            if mes_mot[i] in les_candidats : 
                les_candidats.remove(mes_mot[i]) #On ne peut pas modifier les mots de l'ancre, on les supprime donc des candidats à modifier
# =============================================================================
#     print("LES CANDIDATS SONT !!!!!!!!!!!!!!!!",les_candidats)     
# =============================================================================
    while m <= nbre_perturb :  #tant que on est pas arrivé au nbre de perturbation qu'on desire on continue
        
        new_phrase = copy.copy(ph_expli_tok) #on copie la phrase à perturber pour pas modifier la phrase initial
        
    
        num_change = 1 #combien d'indice on modifie
        change = np.random.choice(les_candidats, num_change,replace=False)
# =============================================================================
#         print("change est !!!!!!!!!!!!!",change)   
#         print(ph_expli_tok)
#         print("LA PHRASE EST ",new_phrase)
# =============================================================================
        indice_change = ph_expli_tok.index(change)
         
        new_phrase[indice_change] = str(MASK)
        
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


def gener_perturb_bert_one(phrase_mask,phrase_entiere,torch):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(phrase_mask, return_tensors="pt")
    labels = tokenizer(phrase_entiere, return_tensors="pt")["input_ids"]

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    torch = torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')

    bert.to(device)
    bert.eval()

    list_mot = unmask(phrase_mask,torch,bert_tokenizer,bert)
    
    choice_mot = np.random.choice(list_mot[0][1],1,replace=False )
    for k in zip(list_mot[0][0],list_mot[0][1]):
        if k[1]== choice_mot : 
            mot = k[0]
    ph_expli_tok= word_tokenize(phrase_mask)
# =============================================================================
#     print(ph_expli_tok)
# =============================================================================
    index_mask = ph_expli_tok.index("MASK")
# =============================================================================
#     print("le type est !!" ,type(mot))
# =============================================================================
    ph_expli_tok[index_mask] = mot
    ph_expli_tok.remove("[")
    ph_expli_tok.remove("]")
# =============================================================================
#     print("le mot est",mot)
# =============================================================================
    phr = " ".join(ph_expli_tok)
    
     
    return(phr)
