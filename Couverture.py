
# ============================================================================#
#                                                                             #
# CODE POUR LE CALCUL DE LA COUVERTURE D UNE ANCRE                            #
#                                                                             #
# ============================================================================#



def cov_bis(ancre,data_sentence):
# =============================================================================
#     Pour calculer la couverture d une ancre, il faut compter combien de fois ce mot apparait dans nos phrases 
# =============================================================================
# =============================================================================
# 
# Toutes nos phrases pour pouvoir compter 
# =============================================================================
    cov_testl = list(data_sentence)
    
    cov = 0 #initialisation
    total = len(data_sentence) #nbre total d'instance
    present = [0]*len(ancre) #liste remplie de 0 de la taille de notre "ancre" exemple: {chien,chat} : present =[0,0]
    
    for j in range(len(data_sentence)): #on parcourt toute les instances
        present = [0]*len(ancre)  #liste remplie de 0 de la taille de notre "ancre" exemple: {chien,chat} : present =[0,0]
        for i in range(len(ancre)): #on parcourt l'ancre
            
            if ancre[i] in cov_testl[j] : #si le mot 1 et le mot 2 sont présent dans la phrase cov_testl[j] alors on aurait [1,1]
                present[i]=1 #L'instance remplie les conditions de la regle alors on a un vecteur remplie de 1
   
# =============================================================================
#     Si tous les mots de l'ancre sont présents dans la phrase cov_testl[j] alors 
#     l'indicateur "present" contient autant de 1 que de mot dans l'ancre 
# =============================================================================
      
        if present.count(1) == len(ancre): 
            cov = cov + 1 # +1 à chaque fois qu'on a une phrase avec les mots de l'ancre (on compte le nombre de phrase remplissant les cond de l'ancre)
    res = (cov/total)*100
    return(res)

