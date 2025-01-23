# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:19:12 2020

@author: juanz
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib import style
from scipy.io import loadmat
import Neuro as neuro
#style.available
style.use('ggplot')

#P1 cargar datos
cat=loadmat("cat.mat")
conectoma_gato=cat["CIJctx"]
nombres_regiones_cat=cat["Names"].tolist()
macaco=loadmat("macaque47.mat")
conectoma_macaco=macaco["CIJ"]
nombres_regiones_monkey=macaco["Names"].tolist()

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
        if conectoma_monkey[i,j]!=0:
            conectoma_monkey[i,j]=1
            conectoma_monkey[j,i]=1


#P3 Gráficas de la matriz adyacente 
plt.figure('Conectoma del gato')
plt.imshow(conectoma_cat,aspect='equal',cmap='cividis')
plt.yticks(np.arange(len(nombres_regiones_cat)),nombres_regiones_cat)
plt.xticks(np.arange(len(nombres_regiones_cat)),nombres_regiones_cat,rotation=90)
plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.colorbar()
plt.show()

plt.figure('Conectoma del macaco')                           
plt.imshow(conectoma_monkey,aspect='equal',cmap='cividis')
plt.yticks(np.arange(len(nombres_regiones_monkey)),nombres_regiones_monkey)
plt.xticks(np.arange(len(nombres_regiones_monkey)),nombres_regiones_monkey,rotation=90)
plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.colorbar()
plt.show()

#P4 Convertir a .net para hacer redes
gephi_cat=neuro.guardarGrafo("Red del conectoma del Gato.net",conectoma_cat)
gephi_monkey=neuro.guardarGrafo("Red del conectoma del Macaco.net",conectoma_monkey)

#P5 Distribuciones de grado 
grado_cat=np.sum(conectoma_cat,axis=0)
np.mean(grado_cat)
grado_monkey=np.sum(conectoma_monkey, axis=1)
np.mean(grado_monkey)

plt.figure('Distribución de grado del gato')
ejex=np.arange(0,len(grado_cat),1)
plt.bar(ejex,grado_cat,edgecolor='black',color='seagreen')
plt.title('Distribución de grado del gato')
plt.xticks(np.arange(len(nombres_regiones_cat)),nombres_regiones_cat,rotation=90)
plt.ylabel('Grado')
plt.show()


plt.figure('Distrubución de grado del macaco')
ejex=np.arange(0,len(grado_monkey),1)
plt.bar(ejex,grado_monkey,edgecolor='black',color='mediumseagreen')
plt.title('Distribución de grado del macaco')
plt.xticks(np.arange(len(nombres_regiones_monkey)),nombres_regiones_monkey,rotation=90)
plt.ylabel('Grado')
plt.show()


#P6 Matriz de distancias 
plt.figure('Matriz de distancias del gato')
md_cat=neuro.calcularMatdeDistancias(conectoma_cat)
plt.imshow(md_cat,aspect='equal',cmap='PuBuGn')
plt.yticks(np.arange(len(nombres_regiones_cat)),nombres_regiones_cat)
plt.xticks(np.arange(len(nombres_regiones_cat)),nombres_regiones_cat,rotation=90)
plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.colorbar()
plt.show()
     
plt.figure('Matriz de distancias del macaco')
md_monkey=neuro.calcularMatdeDistancias(conectoma_monkey)
plt.imshow(md_monkey,aspect='equal',cmap='PuBuGn')
plt.yticks(np.arange(len(nombres_regiones_monkey)),nombres_regiones_monkey)
plt.xticks(np.arange(len(nombres_regiones_monkey)),nombres_regiones_monkey,rotation=90)
plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.colorbar()
plt.show()

#P7 Parametros 
#Densidad
k_cat=len(np.where(conectoma_cat==1)[0])/2  
k_max_cat=(len(conectoma_cat)*(len(conectoma_cat)-1))/2
rho_cat=k_cat/k_max_cat
k_monkey=len(np.where(conectoma_monkey==1)[0])/2
k_max_monkey=(len(conectoma_monkey)*(len(conectoma_monkey)-1))/2
rho_monkey=k_monkey/k_max_monkey

#Eficiencia 
eficiencia_cat=neuro.eficiencia(md_cat)
eficiencia_monkey=neuro.eficiencia(md_monkey)
#Coeficiente de agrupamiento local
coef_al_cat=neuro.coeficienteAgrupamiento(conectoma_cat).flatten()
coef_al_monkey=neuro.coeficienteAgrupamiento(conectoma_monkey).flatten()
#Coeficiente de agrupamiento global 
coef_gl_cat=np.mean(coef_al_cat)
coef_gl_monkey=np.mean(coef_al_monkey) 


#Histograma del coeficiente de agrupamiento local
plt.figure('Distrubución del coeficiente de agrupamiento local del gato')
ejex=np.arange(0,len(coef_al_cat),1)
plt.bar(ejex,coef_al_cat,edgecolor='black',color='olive')
plt.title('Coeficiente de agrupamiento local del gato')
plt.xticks(np.arange(len(nombres_regiones_cat)),nombres_regiones_cat,rotation=90)
plt.ylabel('Coeficiente de agrupamiento local')
plt.show()


plt.figure('Distrubución del coneficiente de agrupamiento local del macaco')
ejex=np.arange(0,len(coef_al_monkey),1)
plt.bar(ejex,coef_al_monkey,edgecolor='black',color='olivedrab')
plt.title('Coeficiente de agrupamiento local del macaco')
plt.xticks(np.arange(len(nombres_regiones_monkey)),nombres_regiones_monkey,rotation=90)
plt.ylabel('Coeficiente de agrupamiento local')
plt.show()


#P8 Algoritmo de agrupamiento-----MeanShift
#GATO
from sklearn.cluster import MeanShift
distancias=np.mean(md_cat,axis=0)
parametros=np.vstack((distancias,grado_cat)).T   

plt.figure('Agrupamiento por "MeanShift" para gato')
plt.ylabel('Grado')
plt.xlabel('Distancia Promedio')
ms = MeanShift()
ms.fit(parametros)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Grupos estimados:", n_clusters_)
color= 10*['g.','b.','c.','y.','m.']

for i in range(len(parametros)):
    plt.plot(parametros[i][0], parametros[i][1], color[labels[i]])
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker = "*",color='darkred', s = 40, linewidth=5,zorder = 10)
plt.show()

#MACACO
distancias=np.mean(md_monkey,axis=0)
parametros=np.vstack((distancias,grado_monkey)).T   

plt.figure('Agrupamiento por "MeanShift" para macaco')
plt.ylabel('Grado')
plt.xlabel('Distancia Promedio')
ms = MeanShift()
ms.fit(parametros)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Grupos estimados:", n_clusters_)
color= 10*['g.','b.','c.','y.','m.']

for i in range(len(parametros)):
    plt.plot(parametros[i][0], parametros[i][1], color[labels[i]])
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
            marker = "*",color='darkred', s = 40, linewidth=5,zorder = 10)
plt.show()

#-----Para obtener 9------

#P1 simulaciones aleatorias
#GATO
pCat=0.19045858  #No. de conexiones existentes/total de conexiones para ser un grafo completo
n=52
nsim=1000
#Completamente aleatorio
mediasCat=[]
ensayos=[]
stdCat=[]
for i in range(nsim):   
    grafo_rnd_cat=neuro.crearGrafoErdosRenyi(n,pCat)
    mediasCat.append(np.mean(np.sum(grafo_rnd_cat,axis=0)))
    ensayos.append(i)
      
long_ensayos=len(ensayos)
stdCat=np.std(mediasCat)
stdmnCat=stdCat/np.sqrt((long_ensayos))

plt.figure('Intervalos de confianza gato (1)')
plt.title('Simulaciones completamente aleatorias')
plt.ylabel('Variación de la media de los grados')
plt.xlabel('Simulaciones')
plt.plot(ensayos,mediasCat,lw=1,color='red')
plt.plot(ensayos,mediasCat+2*stdmnCat,lw=0.3,color='blue')
plt.plot(ensayos,np.ones(long_ensayos)*np.mean(mediasCat+2*stdmnCat),color='gold')
plt.plot(ensayos,mediasCat-2*stdmnCat,lw=0.3,color='blue')
plt.plot(ensayos,np.ones(long_ensayos)*np.mean(mediasCat-2*stdmnCat),color='green')
plt.plot(ensayos,np.ones(long_ensayos)*np.mean(mediasCat),lw=1,color='black')
mn_disrtibuciones_cat = mpatches.Patch(color='red', label='Media de distribuciones',linewidth=6)
plt.show()

#Preservando el grado 
distribucionesCat_rndg=[]
for i in range(nsim):
    grafo_rndg_cat=neuro.rasterPermutado(conectoma_cat,2)
    distribucionesCat_rndg.append(np.sum(grafo_rndg_cat,axis=1))

mediasCat_rndg=np.mean(distribucionesCat_rndg,axis=0)
stdCat_rndg=np.std(mediasCat_rndg)
stdmnCat_rndg=stdCat_rndg/np.sqrt((long_ensayos))

plt.figure('Intervalos de confianza gato (2)')
plt.title('Simulaciones aleatorias conservando el grado')
plt.ylabel('Grados')
plt.xlabel('Nodos')
y=np.arange(1,len(mediasCat_rndg)+1,1)
plt.plot(y,mediasCat_rndg,lw=0.7,color='red')
plt.plot(y,mediasCat_rndg+2*stdmnCat_rndg,lw=0.2,color='blue')
plt.plot(y,np.ones(len(mediasCat_rndg))*np.mean(mediasCat_rndg+2*stdmnCat_rndg))
plt.plot(y,mediasCat_rndg-2*stdmnCat_rndg,lw=0.2,color='blue')
plt.plot(y,np.ones(len(mediasCat_rndg))*np.mean(mediasCat_rndg-2*stdmnCat_rndg))
plt.plot(y,np.ones(len(mediasCat_rndg))*np.mean(mediasCat_rndg),lw=1,color='black')
plt.show()

#MACACO
pMonkey=0.1186057
m=47

#Completamente aleatoria
mediasMonkey=[]
ensayos=[]
stdMonkey=[]
for i in range(nsim):   
    grafo_rnd_monkey=neuro.crearGrafoErdosRenyi(m,pMonkey)
    mediasMonkey.append(np.mean(np.sum(grafo_rnd_monkey,axis=0)))
    ensayos.append(i)
    
long_ensayos=len(ensayos)
stdMonkey=np.std(mediasMonkey)
stdmnMonkey=stdMonkey/np.sqrt(long_ensayos)
linea0=np.mean(mediasMonkey)
plt.figure('Intervalos de confianza macaco (1)')
plt.title('Simulaciones completamente aleatorias')
plt.ylabel('Variación de la media de los grados')
plt.xlabel('Simulaciones')
plt.plot(ensayos,mediasMonkey,lw=0.7,color='red')
plt.plot(ensayos,mediasMonkey+2*stdmnMonkey,lw=0.2,color='blue')
plt.plot(ensayos,np.ones(long_ensayos)*np.mean(mediasMonkey+2*stdmnMonkey),color='gold')
plt.plot(ensayos,mediasMonkey-2*stdmnMonkey,lw=0.2,color='blue')
plt.plot(ensayos,np.ones(long_ensayos)*np.mean(mediasMonkey-2*stdmnMonkey),color='green')
plt.plot(ensayos,np.ones(long_ensayos)*np.mean(mediasMonkey),lw=1,color='black')

mn_disrtibuciones_monkey = mpatches.Patch(color='red', label='Media de distribuciones',linewidth=6)
ic_monkey=mpatches.Patch(color='blue',label='Intervalos de confianza',linewidth=6)
plt.legend(handles=[mn_disrtibuciones_monkey,ic_monkey])
plt.show()

#Preservando el grado 
distribucionesMonkey_rndg=[]
for i in range(nsim):
    grafo_rndg_Monkey=neuro.rasterPermutado(conectoma_monkey,2)
    distribucionesMonkey_rndg.append(np.sum(grafo_rndg_Monkey,axis=1))
    
mediasMonkey_rndg=np.mean(distribucionesMonkey_rndg,axis=0)
stdMonkey_rndg=np.std(mediasMonkey_rndg)
stdmnMonkey_rndg=stdMonkey_rndg/np.sqrt((long_ensayos))
y=np.arange(1,len(mediasMonkey_rndg)+1,1)
plt.figure('Intervalos de confianza macaco (2)')
plt.title('Simulaciones conservando el grado')
plt.ylabel('Grados')
plt.xlabel('Nodos')
plt.plot(y,mediasMonkey_rndg,lw=0.7,color='red')
plt.plot(y,mediasMonkey_rndg+2*stdmnMonkey_rndg,lw=0.2,color='blue')
plt.plot(y,np.ones(len(mediasMonkey_rndg))*np.mean(mediasMonkey_rndg+2*stdmnMonkey_rndg))
plt.plot(y,mediasMonkey_rndg-2*stdmnMonkey_rndg,lw=0.2,color='blue')
plt.plot(y,np.ones(len(mediasMonkey_rndg))*np.mean(mediasMonkey_rndg-2*stdmnMonkey_rndg))
plt.plot(y,np.ones(len(mediasMonkey_rndg))*np.mean(mediasMonkey_rndg),lw=1,color='black')
plt.show()

#----Para obtener el 10----
#Primera parte: Resiliencia de conectomas 
# GATO:
conectoma_cat_dummy=np.copy(conectoma_cat)
Tcg_cat_fallas,E_cat_fallas,L_cat_fallas,C_cat_fallas=neuro.fallasAleatorias(conectoma_cat_dummy)
Tcg_cat_ataques,E_cat_ataques,L_cat_ataques,C_cat_ataques=neuro.ataquesDirigidos(conectoma_cat_dummy)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Resiliencia del conectoma del Gato')

axs[0, 0].set(ylabel='Tamaño del componente gigante')
axs[0, 0].plot(Tcg_cat_fallas,color='g')
axs[0, 0].plot(Tcg_cat_ataques,color='r')

axs[0, 1].set(ylabel='Eficiencia')
axs[0, 1].plot(E_cat_fallas,color='g')
axs[0, 1].plot(E_cat_ataques,color='r')

axs[1, 0].set(ylabel='Longitud característica')
axs[1, 0].plot(L_cat_fallas,color='g')
axs[1, 0].plot(L_cat_ataques,color='r')

axs[1, 1].set(ylabel='Coeficiente de agrupamiento global')
axs[1, 1].plot(C_cat_fallas,color='g')
axs[1, 1].plot(C_cat_ataques,color='r')

for ax in axs.flat:
    ax.set(xlabel='Número de nodos eliminados')

fallas = mpatches.Patch(color='red', label='Atques',linewidth=6)
ataques=mpatches.Patch(color='green',label='Fallas',linewidth=6)
fig.legend(handles=[fallas,ataques],loc='center')

#MACACO
conectoma_monkey_dummy=np.copy(conectoma_monkey)
Tcg_monkey_fallas,E_monkey_fallas,L_monkey_fallas,C_monkey_fallas=neuro.fallasAleatorias(conectoma_monkey_dummy)
Tcg_monkey_ataques,E_monkey_ataques,L_monkey_ataques,C_monkey_ataques=neuro.ataquesDirigidos(conectoma_monkey_dummy)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Resiliencia del conectoma del macaco')

axs[0, 0].set(ylabel='Tamaño del componente gigante')
axs[0, 0].plot(Tcg_monkey_fallas,color='navy')
axs[0, 0].plot(Tcg_monkey_ataques,color='gold')

axs[0, 1].set(ylabel='Eficiencia')
axs[0, 1].plot(E_monkey_fallas,color='navy')
axs[0, 1].plot(E_monkey_ataques,color='gold')

axs[1, 0].set(ylabel='Longitud característica')
axs[1, 0].plot(L_monkey_fallas,color='navy')
axs[1, 0].plot(L_monkey_ataques,color='gold')

axs[1, 1].set(ylabel='Coeficiente de agrupamiento global')
axs[1, 1].plot(C_monkey_fallas,color='navy')
axs[1, 1].plot(C_monkey_ataques,color='gold')

for ax in axs.flat:
    ax.set(xlabel='Número de nodos eliminados')

fallas = mpatches.Patch(color='navy', label='Fallas',linewidth=6)
ataques=mpatches.Patch(color='gold',label='Ataques',linewidth=6)
fig.legend(handles=[fallas,ataques],loc='center')