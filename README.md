#### Morgane Bourbao 
# Projet Anchors                                                                                                                            
## Les exemples et les codes 


Ce GitHub contient les codes utilisés dans l'article. 
Le but de ce mémoire est de comprendre le principe de la méthode Anchors et de comprendre en détail son fonctionnement. Afin de comprendre cette méthode, nous avons réaliser une nouvelle implémentation de la méthode en utilisant de langage `Python`. On s'est aidé des codes de Rebeiro présent du son GitHub (en référence dans l'article).

Plusieurs JupyterNotebook sont à disposition afin de visualiser les différents exemples du mémoire.
Les codes de la nouvelle implémentation de Anchors se trouve dans le dossier `Code`.

Regardons un premier exemple d'application de la méthode Anchors avec perturbation déterministe sur un exemple simple pour se donner une idée de ce qu'est une ancre. 

[Premier exemple d'application](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Application-sur-la-phrase-The-reception-have-been-generally-good.ipynb).

Dans le mémoire, nous utilisons des données textuelles, la machine doit donc vectoriser ces données afin de les manipuler. 
Nous avons parler dans l'article d'une méthode en particulier pour la vectorisation que nous avons illustré sur un exemple. Regardons le résultat obtenue par codage.

[Vectorisation des données avec CountVectorizer de SkLearn](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Vectorisation_Count_Vect.ipynb
).

La méthode Anchors est local, elle explique une instance en particulier. Pour se faire, la méthode utilise des perturbations de l'instance que nous expliquons. Il existe plusieurs méthodes pour générer des perturbations. Dans ce travail, nous étudions deux types de méthode : La méthode dite déterministe et la méthode dite générative. Visualisons ces méthodes à travers des exemples. 

  - [Exemple de perturbation déterministe](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Exemple%20perturbation%20d%C3%A9terministe.ipynb).
  - [Exemple de perturbation générative](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Perturbation_Bert.ipynb).


L'objectif du travail est de comprendre le fonctionnement d'Anchors, nous avons vu dans le mémoire, que cette méthode utilise plusieurs notions : 
  - La couverture,
  - La précision, 
  - Des perturbations. 
 
Regardons sur un exemple, l'evolution de ses valeurs et essayons de comprendre intuitivement comment Anchors raisonne pour choisir le point d'ancrage. 


Afin de vérifier la cohérence de nos codes utilisés dans ce projet, nous réalisons une expérience permettant de mesurer la performance et de valider notre implémentation. 
Comparons dans un premier temps les deux implémentations sur quelques phrases (positives ou négatives) : 

-  [Explication donnée par Anchors sur plusieurs commentaires](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Application_sur_plusieurs_phrase.ipynb) .

-  [Experience pour mesurer la performance de la nouvelle implémentation.](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Comparaison-performance.ipynb)

