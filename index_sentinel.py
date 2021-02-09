# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:24:48 2021

@author: gabriel & alexandre
"""

import numpy as np
import os
import sys
# Ajout des chemins d'accès des scripts pythons existants
sys.path.append('C:/Users/gabriel/Desktop/M2_SIGMA/Ateliers/script')

# Ajout des librairies personnelles
import read_and_write as rw
import pretraitement_classif as pc

def calcul_index (mask, in_Folder=None, nb_ligne=None, nb_col=None, 
                  nb_Out_Img=1, index="NDVI"):
    """
    Charge des images monobandes sentinel 2 de même dimensions et les concatène

    Parameters
    ----------
    mask : numpy array
        Masque des élèments superflus.
    inFolder : str
        Chemin d'accès du dossier parent si on veut boucler le processus. 
        En cas de traitement en batch vos fichiers doivent être contenu dans des sous dossiers
        et doivent s'appeller :
            - NDVI : B4 pour la bande rouge et B8 pour l'IR.
    nb_Out_Img : int
        Nombre d'image souhaité en sortie. Une par sous dossier dans le 
        répertoire parent (int quelconque) ou une contenant toutes les bandes (défaut)
    nb_col : int
        Nombre de colonne de l'image
    nb_ligne : int
        Nombre de ligne de l'image
    index : str
        NDVI (défaut)
        NDWI11
        NDWI12
        
    Returns
    -------
    Image multibandes (concaténation des index)
    Dictionnaire de ndvi (une bande par dossier)
    Pour les 2 choix --> NDVI * 1000
    """
                
    # Initialisation des variables
    p_Directory = os.listdir(in_Folder)
    temp_List = []
    
    # Sélection d'une image de référence pour extraire
    # les dimensions de l'image.
    
    if nb_col == None and nb_ligne == None:
        
        temp = os.path.join(in_Folder + p_Directory[0] + p_Directory[0])
        temp_files = os.listdir(temp)
        
        for f in temp_files: 
                 
            # Sélection d'une image de référence
            if "B4" in f:
                
                ref_Path = os.path.join(temp + "/" + f)                    
                ref_Image = rw.open_image(ref_Path)
                nb_col, nb_ligne, _  = rw.get_image_dimension(ref_Image)
                end_Img = np.empty((nb_ligne, nb_col, len(p_Directory))).astype('int16')
                break
                
    else:
            
        end_Img = np.empty((nb_ligne, nb_col, len(p_Directory))).astype('int16')

    # On parcourt le répertoire parent            
    for p in range(0, len(p_Directory)):
        
        # Création des clefs du dictionnaire      
        temp_List.append(p)  
        dict_index = dict.fromkeys(temp_List)
        # Définition du chemin d'accès des sous répertoires
        e_Directory = os.path.join(in_Folder + p_Directory[p] + p_Directory[p])
        # Liste des fichiers du sous répertoire
        files = os.listdir(e_Directory)
        
        if index == "NDVI": 
            
            # On parcours les fichiers 
            for f in files: 
                
                # Sélection des fichiers d'intérêt
                if "B4" in f:
                    
                    my_Image = os.path.join(e_Directory + "/" + f)                    
                    red = rw.load_img_as_array(my_Image)
                
                elif "B8" in f and not "B8A" in f: 
                    
                    my_Image = os.path.join(e_Directory + "/" + f)
                    ir = rw.load_img_as_array(my_Image)
                    
            # Calcul NDVI (ACORVI (0.05))
            ndvi = ((ir - (red + 0.05)) / (ir + (red + 0.05))) * 1000
            ndvi = ndvi.astype('int16')
            ndvi_mask = pc.mask(ndvi, mask, no_data = -32000, value = 0).astype('int16')
            
            # Si on veut une seule image multibandes en sortie
            if nb_Out_Img == 1:
                
                # Stockage dans une image (nb_band = nb sous dossier)
                end_Img[:,:,[p]] = ndvi_mask
            
            # Une image par dossier : stockage dans un dictionnaire - 1 : premier dossier
            else:
                
                dict_index[p] = ndvi_mask
                
        if index == "NDWI11" or index == "NDWI12": 
    
            # On parcours les fichiers 
            for f in files: 
                
                # Sélection des fichiers d'intérêt
                if "B8A" in f:
                    
                    my_Image = os.path.join(e_Directory + "/" + f)                    
                    nir = rw.load_img_as_array(my_Image)
                
                elif "11" in index and "B11" in f: 
                    
                    my_Image = os.path.join(e_Directory + "/" + f)
                    swir = rw.load_img_as_array(my_Image)
                
                elif "12" in index and "B12" in f: 
                    
                    my_Image = os.path.join(e_Directory + "/" + f)
                    swir = rw.load_img_as_array(my_Image)
                    
            # Calcul NDWI
            ndwi20 = ((nir - swir) / (nir + swir)) * 1000
            ndwi20 = ndwi20.astype('int16')         
            ndwi = pc.resample_sentinel(ndwi20, nb_col, nb_ligne).astype('int16')
            ndwi_mask = pc.mask(ndwi, mask, no_data = -32000, value = 0).astype('int16')
            # Si on veut une seule image multibandes en sortie
            if nb_Out_Img == 1:
                
                # Stockage dans une image (nb_band = nb sous dossier)
                end_Img[:,:,[p]] = ndwi_mask
            
            # Une image par dossier : stockage dans un dictionnaire - 1 : premier dossier
            else :
                
                dict_index[p] = ndwi_mask       
            
    if dict_index[1] is None:
        
        return end_Img
    
    else : 
        
        return dict_index