# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:32:31 2021

@author: gabriel & alexandre 
"""

import matplotlib.pyplot as plt
from sklearn import tree
from joblib import dump, load
import sys
# Ajout des chemins d'accès des scripts pythons existants
sys.path.append('C:/Users/gabriel/Desktop/M2_SIGMA/Ateliers/script/atelier_sigma_2021')

# Ajout des librairies personnelles
import read_and_write as rw
import index_sentinel as ind
import pretraitement_classif as pc
import decisionTree as dt


###Test###

# Définition des chemins d'accès et des noms des fichiers en sortie

#outPoint = "sample2_point.shp"
outFilename1 = "D:/Ateliers_data/acorvi_lvl1.tif"
treeFile1 = "D:/Ateliers_data/Niveau_1/model_NDVI_lvl1.pkl"
outClassif1 = "D:/Ateliers_data/Niveau_1/classif_lvl1.tif"
outFilename2 = "D:/Ateliers_data/acorvi_lvl2.tif"
outClassif2 = "D:/Ateliers_data/Niveau_2/classif_lvl2.tif"
treeFile2 = "D:/Ateliers_data/Niveau_2/model_NDVI_lvl2.pkl"

# Définition des chemin d'accès des fichiers en entrée

myFolder = "D:/Ateliers_data/Sentinel/"
red = "D:/Ateliers_data/Sentinel/SENTINEL2X_20200215-000000-000_L3A_T31TCJ_C_V2-0/SENTINEL2X_20200215-000000-000_L3A_T31TCJ_C_V2-0_FRC_B4.tif"
dataSet = rw.open_image(red)
mask = rw.load_img_as_array('D:/Ateliers_data/RPG/rpg.tif')
#polygonSample = "D:/Ateliers_data/sample/Level2/sample_lvl2.shp"
imageFilename1 = outFilename1
imageFilename2 = outFilename2
sampleFilename1 = "D:/Ateliers_data/sample/sample_pur_point.shp"
sampleFilename2 = "D:/Ateliers_data/sample/Level2/sample_pur_point_lvl2.shp"
arrayZe = rw.load_img_as_array(imageFilename1)



######################################################
# Ajout d'une fonction de rasterisation pour le rpg !#
###################################################### 
   
# NDVI

test = ind.calcul_index (mask, myFolder)

# NDWI

# test_NDWI = ind.calcul_index (mask, myFolder, index = "NDWI11")

# Ecriture

rw.write_image(outFilename1, test, dataSet, gdal_dtype=None,
               transform=None, projection=None, driver_name=None,
               nb_col=None, nb_ligne=None, nb_band=None, no_data = -32000)                     

# Polygone to point :

# pc.polygon_to_point(polygonSample, outPoint)

#############
### LVl 1 ###
#############
       
# Extraction des échantillons

X, Y = rw.get_data_for_scikit (sampleFilename1, imageFilename1, field_name = "label")
                
# 1 - Entrainement 

mean_cm_comp, mean_report_comp, mean_acc_comp, mean_cm, mean_report, mean_acc = \
        dt.train_and_predict (X = X, Y = Y, test_size = 0.33, method = "split", \
                              nb_iter = 30, comparison = True)

print("Matrice de confusion moyenne: \n", mean_cm)
print("Rapport moyen : \n", mean_report)
print("Accuracy moyenne : \n", mean_acc) 


# 2 - Prédiction

img, classif, max_leaf_nodes, min_samples_leaf, min_samples_split = \
    dt.predict_decisionTree(sample_filename = sampleFilename1, \
                            image_filename = imageFilename1, \
                            parcimonie=True, field_name = "label")

#################################Persistance##################################

## ATTENTION ## Problème potentiel de comptabilité entre les versions de python

## Sauvegarder l'arbre avec joblib pour export :
    
dump(classif, treeFile1) 

## Chargement à partir du fichier

# classif = load(joblib_file)

##############################################################################

# Plot tree

fig, ax = plt.subplots(figsize=(20,10))
plot_tree = tree.plot_tree(classif, ax=ax, fontsize=14,
                     feature_names= \
['NDVI_Janvier', 'NDVI_Février','NDVI_Mars','NDVI_Avril','NDVI_Mai', 'NDVI_Juin', 'NDVI_Juillet'\
 ,'NDVI_01Aout', 'NDVI_15Aout', 'NDVI_Septembre','NDVI_Octobre', 'NDVI_Novembre', 'NDVI_Décembre'],
                     class_names=['Autres', 'Végétation'],
                     label='root', impurity=False, proportion=True,
                     precision=2)   


# On retire les pixels du rpg de la classif
    
imgClassifie1 = pc.mask (img, mask, no_data = 255, value = 0) 
    
# 3 - Ecriture
 
rw.write_image(outClassif1, imgClassifie1, dataSet, gdal_dtype=None,
               transform=None, projection=None, driver_name=None,
               nb_col=None, nb_ligne=None, nb_band=1, no_data = 255)

#############
### LVl 2 ###
#############

# 1 - Préparation de l'image pour le lvl 2 

# Masque des zones autres et rpg

array_lvl2 = pc.prepare_for_lvl2(arrayZe, imgClassifie1, -32000)

# Ecriture de l'image

rw.write_image(outFilename2, array_lvl2, dataSet, gdal_dtype=None,
               transform=None, projection=None, driver_name=None,
               nb_col=None, nb_ligne=None, nb_band=None, no_data = -32000)                     

# Extraction des échantillons

X, Y = rw.get_data_for_scikit (sampleFilename2, imageFilename2, field_name = "label2")
                
# 3 - Entrainement 

mean_cm_comp, mean_report_comp, mean_acc_comp, mean_cm, mean_report, mean_acc = \
        dt.train_and_predict (X = X, Y = Y, test_size = 0.33, method = "split", \
                              nb_iter = 30, comparison = True)

print("Matrice de confusion moyenne: \n", mean_cm)
print("Rapport moyen : \n", mean_report)
print("Accuracy moyenne : \n", mean_acc) 


# 4 - Prédiction

img, classif, max_leaf_nodes, min_samples_leaf, min_samples_split = \
    dt.predict_decisionTree(sample_filename = sampleFilename2, \
                            image_filename = imageFilename2, \
                            parcimonie=True, field_name = "label2")

#################################Persistance##################################

## ATTENTION ## Problème potentiel de comptabilité entre les versions de python

## Sauvegarder l'arbre avec joblib pour export :
    
dump(classif, treeFile2) 

## Chargement à partir du fichier

# classif = load(joblib_file)

##############################################################################

# Plot tree

fig, ax = plt.subplots(figsize=(20,10))
plot_tree = tree.plot_tree(classif, ax=ax, fontsize=14,
                     feature_names= \
['NDVI_Janvier', 'NDVI_Février','NDVI_Mars','NDVI_Avril','NDVI_Mai', 'NDVI_Juin', 'NDVI_Juillet'\
 ,'NDVI_01Aout', 'NDVI_15Aout', 'NDVI_Septembre','NDVI_Octobre', 'NDVI_Novembre', 'NDVI_Décembre'],
                     class_names=['Autres', 'Veg_haute', 'Veg_basse'],
                     label='root', impurity=False, proportion=True,
                     precision=2)   


# On retire les pixels du rpg et "autres" de la classif
    
img_temp = pc.mask (img, mask, value = 0, no_data = 255) 
imgClassifie2 = pc.mask (img_temp, imgClassifie1, no_data = 255, value = 0, group = False)
    
# 5 - Ecriture
 
rw.write_image(outClassif1, imgClassifie2, dataSet, gdal_dtype=None,
               transform=None, projection=None, driver_name=None,
               nb_col=None, nb_ligne=None, nb_band=1, no_data = 255)