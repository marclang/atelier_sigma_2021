# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:32:31 2021

@author: gabriel & alexandre 
"""

import decisionTree as dT
import matplotlib.pyplot as plt
from sklearn import tree
import sys
# Ajout des chemins d'accès des scripts pythons existants
sys.path.append('C:/Users/gabriel/Desktop/M2_SIGMA/Ateliers/script/atelier_sigma_2021')

# Ajout des librairies personnelles
import os
import read_and_write as rw




###Test###

# define parameters
myFolder = 'D:/Ateliers_data/Sentinel/'
red = 'D:/Ateliers_data/Sentinel/SENTINEL2X_20200215-000000-000_L3A_T31TCJ_C_V2-0/SENTINEL2X_20200215-000000-000_L3A_T31TCJ_C_V2-0_FRC_B4.tif'
dataSet = rw.open_image(red)
outFilename = os.path.join('D:/Ateliers_data/test_mask2.tif')
mask = rw.load_img_as_array('D:/Ateliers_data/RPG/rpg.tif')
imageFilename = 'D:/Ateliers_data/test_acorvi.tif'
sampleFilename = 'D:/Ateliers_data/sample/Level2/sample_pur_point.shp'

# NDVI

test = dT.calcul_index (mask, myFolder)

# NDWI

test_NDWI = dT.calcul_index (mask, myFolder, index = "NDWI11")

# Ecriture

rw.write_image(outFilename, test, dataSet, gdal_dtype=None,
               transform=None, projection=None, driver_name=None,
               nb_col=None, nb_ligne=None, nb_band=None)                     

# Polygone to point :

dT.polygon_to_point('D:/Ateliers_data/sample/Level2/sample_lvl2.shp', "sample2_point.shp")
                 
# Extraction des échantillons

X, Y = dT.extract_sample (imageFilename, sampleFilename, is_point = True, field_name = "label")

# 1 - Choix des hyperparamètres

# 2/3 des échantillons en entrainement puis 90% de ses éch en fold de test
param_dict = { 
    "max_leaf_nodes" : range(1, 10), 
    "min_samples_split" : range(1, 40), 
    "min_samples_leaf" : range(1, 20)}

X_train, X_test , Y_train, Y_test = dT.train_test_split(X, Y, test_size = 0.33)

best_param, best_estimator, best_score, param_max, acc_max \
    = dT.tune_hyperparametre (X_train, Y_train, param_dict, cv = 10)

print("Meilleur paramètre : ", best_param)
print("Meilleur estimateur : ", best_estimator)
print("Meilleur accuracy : ", best_score)
print("Meilleur estimateur (parcimonie) : ", param_max)
print("Meilleur accuracy (parcimonie) : ", acc_max)
    
## Lvl 1 : ## 300 ech : 
           ## {'max_leaf_nodes': 4, 'min_samples_leaf': 2, 'min_samples_split': 2} 
           ## acc = 0.9631578947368421
           
           ## 600 ech : 
           ## {'max_leaf_nodes': 6, 'min_samples_leaf': 2, 'min_samples_split': 2}
           ## acc = 0.9596464646464646
           
           ## 600 ech - parcimonie :
           ## {'max_leaf_nodes': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
           ## acc = 0.946111111111111
        
        
## Lvl 2 : ## 130 ech :
           ## {'max_leaf_nodes': 5, 'min_samples_leaf': 3, 'min_samples_split': 19}
           ## acc = 0.9112903225806452
                
# 2 - Entrainement 

mean_cm, mean_report, mean_acc = dT.train_and_predict (X_train = X_train, \
                        X_test = X_test , Y_train = Y_train, Y_test = Y_test, \
                        max_leaf_nodes = 3, min_samples_leaf = 1, \
                        min_samples_split = 2, method = "split", nb_iter = 30)

print("Matrice de confusion moyenne: \n", mean_cm)
print("Rapport moyen : \n", mean_report)
print("Accuracy moyenne : \n", mean_acc) 

## Lvl 1 : ## 600 ech : 
           ## {'max_leaf_nodes': 6, 'min_samples_leaf': 2, 'min_samples_split': 2}
           ## Accuracy moyenne : 0.9435897435897436 
           
           ## 600 ech - parcimonie :
           ## {'max_leaf_nodes': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
           ## Accuracy moyenne : 0.9321266968325792

# 3 - Prédiction

# X_img = np.load("D:/Ateliers_data/X_img.npy")
# t_img = np.load("D:/Ateliers_data/t_img.npy")
# image = np.load("D:/Ateliers_data/acorvi_array.npy")

#  # get X
# list_row, list_col = rw.get_row_col_from_file(sample_filename, image_filename)
# X = image[(list_row, list_col)]

# # get Y
# gdf = gpd.read_file(sample_filename)
# Y = gdf.loc[:, "label2"].values
# Y = np.atleast_2d(Y).T

img, classif = dT.predict_decisionTree(sample_filename = sampleFilename,\
                                       image_filename = imageFilename, \
                                       max_leaf_nodes = 6, min_samples_leaf = 2, \
                                       min_samples_split = 2, field_name = "label")

# Plot tree

fig, ax = plt.subplots(figsize=(20,10))
plot_tree = tree.plot_tree(classif, ax=ax, fontsize=14,
                     feature_names= \
['NDVI_Janvier', 'NDVI_Février','NDVI_Mars','NDVI_Avril','NDVI_Mai', 'NDVI_Juin', 'NDVI_Juillet'\
 ,'NDVI_01Aout', 'NDVI_15Aout', 'NDVI_Septembre','NDVI_Octobre', 'NDVI_Novembre', 'NDVI_Décembre'],
                     # class_names=['Autres', 'Végétation'],  # Lvl 1
                     class_names=['Autres', 'Veg_haute', 'Veg_basse'], # Lvl 2
                     label='root', impurity=False, proportion=True,
                     precision=2)   


# On retire les pixels du rpg de la classif
    
imgClassifie = dT.mask_rpg (img, mask, no_data = 255) 
    
# 4 - Ecriture

outClassif = os.path.join('D:/Ateliers_data/classif_lvl2.tif')
 
rw.write_image(outClassif, imgClassifie, dataSet, gdal_dtype=None,
               transform=None, projection=None, driver_name=None,
               nb_col=None, nb_ligne=None, nb_band=1, no_data = 255)
