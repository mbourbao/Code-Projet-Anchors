import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
import os
import string 
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import copy
os.chdir("C:/Users/33651/Documents/Projet_Anchors/CODE/")
from classifier import *
from perturbation import * 
from couverture import * 
from select_cov import * 
import itertools
import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble



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

# =============================================================================
# 
# CODE : Fonction qui renvoie la meilleur ancre (méthode perturbation déterministe )
# 
# 
# =============================================================================

def meilleur_ancre_bis_new(df,phrase,nbre_perturb,st,senti,tau,vectorizer, classifier ):
    
    
    label = senti #Sentiment postif = 1 ou sentiment negatif = 0 
    
# =============================================================================
#     ################ PREPARATION DE LA PHRASE ####################
# =============================================================================

    ph_expli_tok= word_tokenize(phrase.lower()) #Decoupage de la phrase + mise en minuscule(pour eviter les erreurs)


    #Liste des commentaires 
    data_sentences = df['sentence'].values 
    

    les_candidats = ph_expli_tok
    print("les_candidats",les_candidats)

    meilleur_prec = 0 #Pour conserver l'ancre ayant la meilleur précision.
    prec_temp = 0 # Précision temporaire -> Pour comparer avec la meilleur.
    best_ancre = [] #On retient l'ancre ayant la meilleur precision.
    
    if len(les_candidats) <=1 : #S'il y a qu'une ancre possible pour être le candidat on le retourne.
        best_candidats = les_candidats
        return(best_candidats)
        
    
    for i in range(1,len(les_candidats)): #Taille de la combinaison des ancres
        comb_temp = [] #L'ancre temporaire -> celle qu'on etudie
        
        for c in itertools.combinations(les_candidats,i): #Toutes les combinaisons possibles des candidats de taille i
            x= list(c) #On enregistre toutes les combinaisons possibles
            comb_temp.append(x)
            if meilleur_prec > 0.98 : #Si on est arrivé au maximum de la précsion (on ne peut pas aller + haut que 1) evite de continuer pour rien
                 print('Anchor:', best_ancre , 'Precision:', meilleur_prec )
                 return(best_ancre)
                
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
                perturb = generation_perturb_bis4(ph_expli_tok,comb_temp[p],nbre_perturb,st)
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
                        
                    
                 
    print('Anchor:', best_ancre , 'Precision:', meilleur_prec )
 
    
    return(best_ancre)
