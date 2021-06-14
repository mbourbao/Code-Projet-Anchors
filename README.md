#### Morgane Bourbao 
# Projet Anchors                                                                                                                            
## Les exemples et les codes 


Ce GitHub contient les codes utilisés dans l'article. 
Le but de ce mémoire est de comprendre le but de la méthode Anchors, de comprendre en détail son fonctionnement et de vérifier la cohérence des résultats obtenus en l'appliquant. Afin de comprendre son fonctionnement et de présenter son mécanisme de fonctionnement, nous avons réaliser une nouvelle implémentation en se basant sur les codes du GitHub de Rebeiro. 

Plusieurs JupyterNotebook sont à disposition afin de visualiser les différents exemples de l'article.
Les codes de la nouvelle implémentation de Anchors se trouve dans le dossier `Code`.

Regardons un premier exemple d'application de la méthode Anchors avec perturbation déterministe sur un exemple simple pour se donner une idée de ce qu'est une ancre. 

[Premier exemple d'application](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Application-sur-la-phrase-The-reception-have-been-generally-good.ipynb).

Dans le mémoire, nous utilisons des données textuelles, la machine doit donc vectoriser ces données afin de les manipuler. 
Nous avons parler dans l'article d'une méthode en particulier pour la vectorisation que nous avons illustré sur un exemple. Regardons le résultat obtenue par codage.

[Vectorisation des données avec CountVectorizer de SkLearn](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Vectorisation_Count_Vect.ipynb
).

La méthode Anchors est local, elle explique une instance en particulier. Pour se faire, la méthode utilise des perturbations de l'instance que nous expliquons. Il existe plusieurs méthodes pour générer des perturbations. Dans ce travail, nous étudions deux types de méthode : La méthode dite déterministe et la méthode dite générative. Visualisons ces méthodes à travers des exemples. 

  - [Exemple de perturbation déterministe](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Exemple%20perturbation%20d%C3%A9terministe.ipynb).
  - [Exemple de perturbation générative](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Perturbation_Bert.ipynb)

Afin de vérifier la cohérence de nos codes utilisés dans ce projet, nous réalisons une expérience permettant de mesurer la performance et de valider notre implémentation. 

- [Experience pour mesurer la performance de la nouvelle implémentation.](https://github.com/mbourbao/Code-Projet-Anchors/blob/main/Notebook/Comparaison-performance.ipynb)

  
