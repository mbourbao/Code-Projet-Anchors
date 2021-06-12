
import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
import time
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
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import copy
os.chdir("C:/Users/33651/Documents/Projet_Anchors/CODE/")
from classifier import *
from dict_local import * 
from perturbation import * 
from couverture import * 
import itertools
from string import *
from select_cov import * 
import itertools
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
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import copy
os.chdir("C:/Users/33651/Documents/Projet_Anchors/CODE/")
from classifier import *
from dict_local import * 
from perturbation import * 
from couverture import * 
import itertools
import matplotlib.patches as mpatches

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from meilleur_ancre_general import*
nlp = spacy.load('en_core_web_sm')
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

# =============================================================================
# 
# Separation en matrice entrainement - test - validation
# =============================================================================

train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(sentences, y, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)


# =============================================================================
# Vectorisation des données textuelles 
# 
# =============================================================================
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)

# =============================================================================
# Entrainement du classifier
# =============================================================================


c = sklearn.linear_model.LogisticRegression()
# c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))



explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True) #PERTURBATION AVEC BERT

np.random.seed(1)

# =============================================================================
# Recuperation des commentaires 
# =============================================================================
sen = df["sentence"]
sen_a = np.array(sen)

## On prend des phrases pas trop grande - Pour avoir des commentaires qui ont du sens - 
phrase_expl = []
for i in range(len(sen_a)):
    if len(sen_a[i]) <= 40: 
        phrase_expl.append(sen_a[i])
    

#indice_phrase =  np.random.choice(len(phrase_expl), 100 ,replace=False)



list_performance =[]


perf = 0

prec = []
res= 0
etap = 1
phrase_l= []
ancre_faux_p = []
ancre_faux_moi = []


# =============================================================================
# On va répéter 10 fois les mêmes étapes 
# =============================================================================
while perf < 10 : 
     g=0
     indice_phrase =[]
     while g < 100 :
         precis = 0 #Initialisation de la précision à chaque fois on la réinitialise à 0
# =============================================================================
#          Pour faire le test, on prend des bons commentaires c'est à dire que le package trouve bien l'ancre (précision 1.0) pour eviter les erreurs 
#          due à des commentaires mal rédigés qu'on arrive pas à classer '
# =============================================================================
         while precis != 1 : #On récupère les indices des phrases qui sont bien 
             indice=  np.random.choice(len(phrase_expl), 1 ,replace=False)
# =============================================================================
#              print(indice)
#              print(phrase_expl[indice[0]])
# =============================================================================
             #Application du package Anchors
             exp3 = explainer.explain_instance(phrase_expl[indice[0]], predict_lr, threshold=0.95)
             #précision obtenue avec le package
             precis = exp3.precision()
         # Si l'indice n'est pas déjà dans les indices enregistrer => Evite d'avoir des phrases en double à expliquer    
         if indice[0] not in indice_phrase :
             indice_phrase.append(indice[0]) 
             g=g+1

         for i in indice_phrase:
# =============================================================================
#          print(phrase_expl[i])
# =============================================================================
            label = predict_lr([phrase_expl[i]])[0]
# =============================================================================
#         print("Le label associé est", label)
# =============================================================================
        
        
# =============================================================================
#         print("La phrase est", phrase_expl[i])
# =============================================================================
            phrase_l.append(phrase_expl[i])
        
            moi = meilleur_ancre_bis_new(df,phrase_expl[i],500,"UNK",label,0.15,vectorizer,c)
            exp3 = explainer.explain_instance(phrase_expl[i], predict_lr, threshold=0.95)
            ancre = exp3.names()
            precis = exp3.precision()
            for k in range(len(ancre)):
                ancre[k] = ancre[k].lower()
        
            ancre_moi = sorted(moi)
            ancre_pack = sorted(ancre)
            print("COMPARAISON : MON ANCRE ", ancre_moi ,"VS" , ancre_pack )
        
            if ancre_moi == ancre_pack : 
            
                res = res + 1 
                print("Résultat identique")
                print('Precision_p: %.2f' % exp3.precision())
            
            else : 
                prec.append(precis)
                ancre_faux_p.append(ancre_pack)
                ancre_faux_moi.append(ancre_moi)
                print("Résultat différent")
                print('Precision_p: %.2f' % exp3.precision())
        
            print("etape",etap,"/100")
            etap = etap+1
     perf = perf +1 
     list_performance.append(res)


