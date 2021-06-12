############## BERT ##############
from transformers import DistilBertModel, DistilBertConfig
import ipywidgets
import IProgress
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch

import numpy as np 
import pandas as pd 

import string 
from nltk import word_tokenize
import copy



torch = torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Fonction GitHub _ Ribeiro 
## Cette fonction prend en entrée une phrase masqué (avec ou plusieurs mots cachés) 
## Elle renvoie une liste contenant autant de sous liste que de mot caché
## Ces sous liste contienne une liste de candidat pour remplacer le mot caché (mot qui convient deviner par BERT)
## Les sous liste contienne une autre liste contenant les identifiants des mots candidats
# Par exemple : Pour une phrase avec un mot masqué la fonction renvoie [["good","like"],[1.2,1.3]] 
# C'est à dire que le mot masqué peut-etre remplacer soit par good soit par like, et pour faciliter l'accés à ces données
# Respectivement good a pour identifiant 1.2 et like a pour identifiant 1.3 


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
    
    
    ph_expli_tok = word_tokenize(instance.lower()) #Séparation des mots de la phrase + mise en minuscule pour éviter les erreurs 
    les_candidats= copy.copy(ph_expli_tok) #On realise une copie pour éviter de modifier la base
    les_candidats = list(set(les_candidats)) #Supprime les doublons 
        
    m=0  #compteur initialisé à 0, qui va permettre de gerer le nombre d'instance à generer
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
        change = np.random.choice(les_candidats, num_change,replace=False) #On choisit le mot qui va être masquer au hasard => Tirage uniforme
# =============================================================================
#         print("change est !!!!!!!!!!!!!",change)   
#         print(ph_expli_tok)
#         print("LA PHRASE EST ",new_phrase)
# =============================================================================
        indice_change = ph_expli_tok.index(change) #On récupère l'indice du mot qui a été séléctionné pour être modifier
         
        new_phrase[indice_change] = str(MASK) # On le masque

        phr = " ".join(new_phrase) #on reconstitue la phrase
     
        sauv_phrase.append(phr) #On sauvegarde la phrase masquer 
        
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
    # Décomposition 
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    #Bert Mask => Pour faire deviner les mots masquer 
    bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')

    bert.to(device)
    bert.eval()
    #Genere une liste de choix de mot qu'on peut remplacer - deviner par BERT 
    list_mot = unmask(phrase_mask,torch,bert_tokenizer,bert)
    
    #On choisit le mot au hasard => Tirage uniforme parmi tous les mots deviner par BERT (récupération par identifiant)
    choice_mot = np.random.choice(list_mot[0][1],1,replace=False )
    #On récupère le nom du mot associé à l'identifiant tiré au sors 

    for k in zip(list_mot[0][0],list_mot[0][1]):
        if k[1]== choice_mot : 
            mot = k[0]
    #On décompose pour pouvoir remplacer plus facilement 
    ph_expli_tok= word_tokenize(phrase_mask)

#On récupère l'indice du mot masquer pour qu'on puisse le modifier 
    index_mask = ph_expli_tok.index("MASK")

#On remplace par le mot qu'on a tirer au sors parmi les mots proposés par BERT
    ph_expli_tok[index_mask] = mot
    ph_expli_tok.remove("[")
    ph_expli_tok.remove("]") #Quand on a masquer on a mit sous forme de liste - Le tokenizer a decomposer les crochets de la liste on les supprime pour pas qu'il apparaisse à la fin.

    phr = " ".join(ph_expli_tok) #On recontruie la phrase
    
     
    return(phr)
