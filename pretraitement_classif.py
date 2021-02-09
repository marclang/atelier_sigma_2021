# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:11:57 2021

@author: gabriel & alexandre
"""

import geopandas as gpd
import numpy as np
import os


def resample_sentinel(array, nb_col, nb_ligne):
    """
    Modifie la résolution des bandes 20m² des images Sentinel en 10m² par
    réechantillonage "nearest neighbour"
    
    Parameters
    ----------
    array : array numpy
        Array de l'image à réechantillonner'
    nb_col, nb_ligne : int
        Nombres de colonnes et de lignes de l'image à réechantillonner
        
    Returns
    -------
    Array numpy
    """
    #Initialisation des paramètres et de l'array d'accueil
    l, c, x, y = 0, 0, 0, 0
    nb_ligne20 = nb_ligne / 2
    nb_col20 = nb_col / 2
    array_Rech = np.empty((nb_ligne,nb_col, 1)) 
    
    # On attribue chaque pixel d'array à 4 pixel de array_Rech
    while x < nb_ligne20: 
        
        array_Rech[l:l+2,c:c+2] = array[x,y]
        
        # Si on arrive en bout de ligne on reset les itérateurs
        if x == nb_ligne20-1:
            
            # Si on arrive en bout de colonnes on sort de la boucle
            if y == nb_col20-1:
                break
            
            # Incrémentation en fin de ligne
            l = -2
            c += 2
            x = -1
            y += 1
            
            
        
        # Incrémentation pour chaque "kernel"
        x += 1
        l += 2
        
    return array_Rech

def mask (array, mask, no_data, value, group=True):
    """
    Permets de retirer les élèments superflus des couches nécessaires à la classification.
    
    Parameters
    ----------
    array : numpy array
        Image à classifier
    mask : numpy array
        couche de masque
    noData : int
        valeur de noData
    value : int
        valeur à masquer
    group : bool
        True (défaut) : Toutes les valeurs strictement supérieur à la valeur indiquée sont masquées
        False : Seul la valeur à masquer est masquée
        
    Returns
    -------
    array : numpy array
        Array numpy sans les valeurs du mask.
    """

    if group == True:
        
        mask_bool = mask > value
        array[mask_bool] = no_data
        
    else:
        
        mask_bool = mask == value
        array[mask_bool] = no_data
        
    return array

def prepare_for_lvl2 (array, array_classif1, no_data, other_value = 0):
    """
    Permets d'isoler les pixels ou de la végétation a été prédite lors de la première classification

    Parameters
    ----------
    array : numpy array
        Image à classifier
    array_classif1 : numpy array
        Array issue de la classification niveau 1
    noData : int
        valeur de noData
    other_value : int
        valeur de la classe à masquer (autres) dans array_classif1 (défaut 0)
    Returns
    -------
    array : numpy array
        Array de la zone d'étude sans la classe à retirer (defaut : "autres")
    """
    # Récupération des dimensions de de l'image à masquer
    
    nb_band = array.shape[2]
    nb_col = array.shape[1]
    nb_ligne = array.shape[0]
    
    # Création de l'array d'accueil
    
    array_lvl2 = np.zeros((nb_ligne, nb_col, nb_band)).astype('int16')
        
    # Création du masque pour la classe "autres" du premier niveau de la classification
    
    
    # Masquage "autres" sur toutes les bandes
    for i in range(0,nb_band):
        
        array_temp = np.zeros((nb_ligne, nb_col, 1)).astype('int16')
        array_temp[:,:,0] = array[:,:,i]
        array_temp_masked = mask(array = array_temp, mask = array_classif1, \
                                 no_data = no_data, value = other_value, group = False)
        array_lvl2[:,:,i] = array_temp_masked[:,:,0] 
        
    return array_lvl2

def polygon_to_point (vector_filename, out_filename):
    """
    Calcul du centroide pour chaque polygones en entrée.

    Parameters
    ----------
    vector_filename : couche vecteur (gpd).
    out_filename : nom du fichier en sortie
    Returns
    -------
    Couche points (gpd)
    """
    
    polygon = gpd.read_file(vector_filename)
    temp = polygon.centroid
    
    centroid_bool = polygon.contains(temp) 
    
    for i in range(0,len(centroid_bool)): 
        
        # Si centroid n'est pas contenu dans polygone on prends un point au hasard dans le polygone
        if not centroid_bool[i]:
        
            temp[i] = polygon.loc[i,'geometry'].representative_point()
    
    point = polygon.copy()
    point.loc[:,'geometry'] = temp
    
    head_tail = os.path.split(vector_filename) 
    path = head_tail[0]
    
    point.to_file(os.path.join(path + "/" + out_filename))