# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 22:46:13 2021

@author: gabriel & alexandre 
"""

# Ajout des librairies
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import sys
# Ajout des chemins d'accès des scripts pythons existants
sys.path.append('C:/Users/gabriel/Desktop/M2_SIGMA/Ateliers/script/atelier_sigma_2021')

# Ajout des librairies personnelles
import read_and_write as rw
import classification as cla


def tune_hyperparametre (X, Y, param_dict, cv=10):
    """
    Réitère sur toutes les combinaisons possibles d'hyperparmaètres fournis en entrée.

    Parameters
    ----------
    X : array des pixels
    Y : labels des échantillons
    param_dict : Dictionnaire
        liste et amplitudes des hyperparamètres.
    cv : nombre de folds
    
    Return
    -------
    best_params : meilleurs paramètres pour le modèle
    best_estimator : meilleur estimateur pour le modèle
    best_score : meilleur score pour le modèle
    param_max : paramètres du meilleur modèle selon le principe de parcimonie 
    acc_max : accuracy du meilleur modèle selon le principe de parcimonie

    """
    
    clf = tree.DecisionTreeClassifier()
    
    grid = GridSearchCV(clf, param_grid = param_dict, cv = cv, verbose = 1, n_jobs = -1)
    grid.fit(X,Y)
    
    # Meilleur modèle
    best_param = grid.best_params_
    best_estimator = grid.best_estimator_
    best_score = grid.best_score_
    
    # On sort l'index du meilleur modèle
    index = grid.best_index_
    
    # Array bool des modèles dont le score est à moins d'un écart type du score du meilleur modèle
    results = grid.cv_results_
    array_parsimony = results["mean_test_score"] > best_score - results["std_test_score"][index]
    
    # Array des index des modèles
    array_index = np.arange(len(array_parsimony))
    
    # On mets une valeur particulière aux modèles à garder 
    # (on passe par les index et non bool pour prendre en compte les NaN)
    array_index[array_parsimony] = -9999
    
    # Suppression des modèles peu performants
    array_index = array_index[array_index != -9999]
    
    # Création de l'array des paramètres et des accuracy des modèles
    array_temp = results["params"]
    array_acc_temp = results["mean_test_score"]
    
    # Suppression des paramètres des modèles qui ne nous intéresse pas.
    array_param = np.delete(array_temp, array_index)
    array_acc = np.delete(array_acc_temp, array_index)
    
    # On stocke l'accuracy maximale des modèles avec le moins de leaf_nodes
    # (corrélée aux nombres de changements et donc à la parcimonie) et son index
    i = 0
    acc_max = array_acc[array_param[i]["max_leaf_nodes"]]
    min_leaf = array_param[0]["max_leaf_nodes"]
    
    # On boucle parmis les modèles ayant le moins de leaf pour chercher l'accuracy maximale
    while array_param[i]["max_leaf_nodes"] == min_leaf:

        acc_temp = array_acc[array_param[i]["max_leaf_nodes"]]

        if array_param[i+1]["max_leaf_nodes"] == min_leaf:

            acc_temp1 = array_acc[array_param[i+1]["max_leaf_nodes"]]

            if acc_temp < acc_temp1:

                acc_max = acc_temp1

        else:

            break

        i += 1
        
    # Stockage de l'index de l'accuracy maximale
    acc_where = np.where(acc_max)  
       
    # On récupère les paramètres du modèle retenu
    param_max = array_param[acc_where]
        
    return best_param, best_estimator, best_score, param_max, acc_max

def train_and_predict (X, Y, cv=10,\
                       test_size=None, \
                       method="split", nb_iter=30, comparison=True):
    """
    Entrainement et test de l'arbre de décision (CART optimisé)

    Parameters
    ----------
    if "split" :
        X : array des pixels
        Y : labels des échantillons
    method : k-fold ou split (défaut),
    cv : nombre de folds à créer pour le test des hyperparmaètres
    nb_iter : nombre d'itération pour l'entrainement (défaut : 30) si method = k-fold : nombre de folds.
    test_size : Porportions de l'échantillon test (à remplir si X est renseigné)
    comparison : if True --> fournit les indices de qualité comparant les modèles parcimonieux et non-parcimonieux
                             en plus de ceux du modèle parcimonieux
                 if Falste --> fournit les indices de qualité du modèle parcimonieux

    Returns
    -------

    mean_cm : Moyenne des matrices de confusions du modèle parcimonieux
    mean_report : Moyenne des rapports des différentes métriques du modèle parcimonieux
    mean_acc : Moyenne de l'OA des tests du modèle parcimonieux
    
    Options : 
    mean_cm_comp : Moyenne des matrices de confusions du modèle parcimonieux versus modèle non parcimonieux
    mean_report_comp : Moyenne des rapports des différentes métriques du modèle parcimonieux versus modèle non parcimonieux
    mean_acc_comp : Moyenne de l'OA des tests du modèle parcimonieux versus modèle non parcimonieux
    """
    
    ## Initialisation des liste de stockage des métrique : 
        
    # Modèle parcimonieux
    list_cm = []
    list_acc = []
    list_report = []
    
    # Comparaison des modèles parcimonieux et non-parcimonieux
    list_cm_comp = []
    list_acc_comp = []
    list_report_comp = []
    
    # Définition des hyperparamètres à tester
    param_dict = {"max_leaf_nodes" : range(1, 10), 
                  "min_samples_split" : range(1, 40), 
                  "min_samples_leaf" : range(1, 20)}
    
    if method == "split":
            
        for ite in range(nb_iter): 
            
            ## Split des échantillons
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)
                
            ## Détermination des meilleurs hyperparamètres et du modèle parcimonieux
            best_param, best_estimator, best_score, param_max, acc_max = \
                tune_hyperparametre(X = X_train, Y = Y_train, param_dict = param_dict, cv = cv)
            
            ## Extraction des hyperparamètres :
                
            # Modèle parcimonieux
            max_leaf_nodes_parci = param_max[0]["max_leaf_nodes"]
            min_samples_leaf_parci = param_max[0]["min_samples_leaf"]
            min_samples_split_parci = param_max[0]["min_samples_split"]
            
            # Modèle non-parcimonieux
            max_leaf_nodes = best_param["max_leaf_nodes"]
            min_samples_leaf = best_param["min_samples_leaf"]
            min_samples_split = best_param["min_samples_split"]
            
            ## Entrainement :
                
            # Modèle parcimonieux
            clf_parci = tree.DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes_parci, \
                                                    min_samples_split = min_samples_split_parci,\
                                                    min_samples_leaf = min_samples_leaf_parci)
                
            clf_parci.fit(X_train, Y_train)
            
            # Modèle non-parcimonieux
            clf = tree.DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, \
                                              min_samples_split = min_samples_split,\
                                              min_samples_leaf = min_samples_leaf)
                
            clf.fit(X_train, Y_train)
                
            ## Test :
                
            # Modèle parcimonieux
            Y_predict_parci = clf_parci.predict(X_test)
            
            # Modèle non-parcimonieux
            Y_predict = clf.predict(X_test)
            
            ## Calcul des métriques :
            # Modèle parcimonieux
            cm_parci = confusion_matrix(Y_test, Y_predict_parci)
            report_parci = classification_report(Y_test, Y_predict_parci, labels=np.unique(Y_predict_parci),output_dict=True)
            accuracy_parci = accuracy_score(Y_test, Y_predict_parci)
        
            list_cm.append(cm_parci)
            list_acc.append(accuracy_parci)
            list_report.append(rw.report_from_dict_to_df(report_parci))
            
            if comparison == True:
                # Métriques de comparaison des modèles parcimonieux et non-parcimonieux
                cm_comp = confusion_matrix(Y_predict, Y_predict_parci)
                report_comp = classification_report(Y_predict, Y_predict_parci, labels=np.unique(Y_predict_parci), output_dict=True)
                accuracy_comp = accuracy_score(Y_predict, Y_predict_parci)
            
                list_cm_comp.append(cm_comp)
                list_acc_comp.append(accuracy_comp)
                list_report_comp.append(rw.report_from_dict_to_df(report_comp))
            
        """
    elif method == "k-fold":
        
        skf = StratifiedKFold(n_splits=nb_iter)

        # Iterations sur skf
        for index_train, index_test in skf.split(X, Y):
            X_train, X_test = X[index_train], X[index_test]
            Y_train, Y_test = Y[index_train], Y[index_test]
            
            # Entrainement
            clf = tree.DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, \
                  min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
            clf.fit(X_train, Y_train)
            
            # Test
            Y_predict = clf.predict(X_test)
            
            # Métriques
            cm = confusion_matrix(Y_test, Y_predict)
            report = classification_report(Y_test, Y_predict, labels=np.unique(Y_predict), output_dict=True)
            accuracy = accuracy_score(Y_test, Y_predict)
                    
            list_cm.append(cm)
            list_acc.append(accuracy)
            list_report.append(rw.report_from_dict_to_df(report))    
            """
    
    ## Calcul de la moyenne : 
        
    # Modèle parcimonieux
    
    ## CM moyenne
    
    array_cm = np.array(list_cm)
    mean_cm = array_cm.mean(axis=0)
    
    ## OA moyen : 
    
    array_accuracy = np.array(list_acc)
    mean_acc = array_accuracy.mean()
    
    ## Rapports de classification moyen : 

    array_report = np.array(list_report)
    mean_report = array_report.mean(axis=0)
    
    if comparison == True:
        
        ## CM moyenne

        array_cm_comp = np.array(list_cm_comp)
        mean_cm_comp = array_cm_comp.mean(axis=0)

        ## OA moyenne
            
        array_accuracy_comp = np.array(list_acc_comp)
        mean_acc_comp = array_accuracy_comp.mean()
        
        ## Rapport moyen
        
        array_report_comp = np.array(list_report_comp)
        mean_report_comp = array_report_comp.mean(axis=0)

        return mean_cm_comp, mean_report_comp, mean_acc_comp, mean_cm, mean_report, mean_acc
   
    else :
        
        return mean_cm, mean_report, mean_acc


def predict_decisionTree(cv=10,\
                         X=None, Y=None, X_img=None, t_img=None, \
                         image_filename=None, sample_filename=None, \
                         parcimonie=True, field_name='label'):
    """
    Lance un arbre de décision (CART optimisé) avec la librairie scikit.learn

    Parameters
    ----------
    cv : nombre de folds à créer pour le test des hyperparamètres
    Si data_set en entrée :
        sample_filename : couche vecteur des échantillons.
    Si array en entrée :
        X : pixels des échantillons
        Y : Labels des échantillons
        X_img : pixels de l'image à classifier.
        t_img : coordonnées de l'image à classifier.
    image_filename : data_set GDAL (raster) de la zone d'étude.
    parcimonie : True - prise en compte du modèle parcimonieux
                 False - prise en compte du modèle non-parcimonieux 
    field_name : nom du champ contenant les labels (défaut : label)
    
    Return
    -------
    img : numpy array 
            Contient les labels prédit.
    clf : classifieur du modèle entraîné
    max_leaf_nodes : nombres de feuilles maximum
    min_samples_split : nombre d'échantillons minimumu pour autoriser un choix
    min_samples_leaf : nombre d'échantillons minimum dans une feuille
    
    """
    
    
    # Extraction des échantillons 
    if sample_filename is not None:
        
        X, Y = rw.get_data_for_scikit (sample_filename, \
                                       image_filename,\
                                       field_name = field_name)
        
        X_img, _, t_img = cla.get_samples_from_roi(image_filename, image_filename)
    
    # Définition des hyperparamètres à tester
    param_dict = {"max_leaf_nodes" : range(1, 10), 
                  "min_samples_split" : range(1, 40), 
                  "min_samples_leaf" : range(1, 20)}    
    
    # Détermination des meilleurs hyperparamètres du modèle parcimonieux    
    best_param, best_estimator, best_score, param_max, acc_max = \
        tune_hyperparametre(X = X, Y = Y, param_dict = param_dict, cv = cv)
     
        
    # Choix du modèle : parcimonieux ou non-parcimonieux

    if  parcimonie == True:
        max_leaf_nodes = param_max[0]["max_leaf_nodes"]
        min_samples_leaf = param_max[0]["min_samples_leaf"]
        min_samples_split = param_max[0]["min_samples_split"]
    else :
        max_leaf_nodes = best_param["max_leaf_nodes"]
        min_samples_leaf = best_param["min_samples_leaf"]
        min_samples_split = best_param["min_samples_split"]
            
    # Entrainement sur tout le jeu d'échantillons
    clf = tree.DecisionTreeClassifier(max_leaf_nodes = max_leaf_nodes, \
                                      min_samples_split = min_samples_split,\
                                      min_samples_leaf = min_samples_leaf)
    clf.fit(X, Y)
    
    # Prédiction       
    Y_img_predict = clf.predict(X_img)
    
    # Création de l'array d'accueil
    ds = rw.open_image(image_filename)
    nb_row, nb_col, nb_band = rw.get_image_dimension(ds) 
    
    img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
    img[t_img[0], t_img[1], 0] = Y_img_predict
    
    return img, clf, max_leaf_nodes, min_samples_leaf, min_samples_split
                

