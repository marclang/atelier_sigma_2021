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

def mask_rpg (array, mask, no_data):
    """
    Permets de retirer les élèments superflus des couches nécessaires à la classification.
    
    Parameters
    ----------
    array : numpy array
        Image à classifier
    mask : numpy array
        couche de masque
    noData : valeur de noData
        
    Returns
    -------
    outArray : numpy array
        Array numpy sans les valeurs du mask.
    """
    mask_bool = mask > 0
    array[mask_bool] = no_data
    
    return array

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