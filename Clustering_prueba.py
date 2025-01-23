# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:46:59 2020

@author: juanz
"""
from scipy.io import loadmat
import Neuro as neuro
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from sklearn.datasets import *

#P1 cargar datos
cat=loadmat("cat.mat")
macaco=loadmat("macaque47.mat")
conectoma_gato=cat["CIJctx"]
conectoma_macaco=macaco["CIJ"]

#P2 Matriz adyacentes no dirigidas y binarias 
conectoma_cat=np.copy(conectoma_gato)
conectoma_monkey=np.copy(conectoma_macaco)

k,_=conectoma_cat.shape

for i in range(k):
    for j in range(k):
        if conectoma_cat[i,j]!=0:
            conectoma_cat[i,j]=1
            conectoma_cat[j,i]=1

l,_=conectoma_monkey.shape

for i in range(l):
    for j in range(l):
        if conectoma_cat[i,j]!=0:
            conectoma_cat[i,j]=1
            conectoma_cat[j,i]=1
            
md_cat=neuro.calcularMatdeDistancias(conectoma_cat)
#------Agrupación-------
grado_cat=np.sum(conectoma_cat,axis=0).T
coef_al_cat=neuro.coeficienteAgrupamiento(conectoma_cat).T
distancias=np.mean(md_cat,axis=0)
parametros=np.vstack((distancias,grado_cat)).T         

#Agrupamiento 
#------MeanShift-------
#matplotlib inline
from sklearn.cluster import MeanShift
from matplotlib import style
style.use("ggplot")
plt.scatter(parametros[:,0],parametros[:,1])

ms = MeanShift()
ms.fit(parametros)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
color= 10*['r.','g.','b.','c.','k.','y.','m.']

for i in range(len(parametros)):
    plt.plot(parametros[i][0], parametros[i][1], color[labels[i]])
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
            marker = "*", s = 40, linewidth=5,zorder = 10)
plt.show()





        