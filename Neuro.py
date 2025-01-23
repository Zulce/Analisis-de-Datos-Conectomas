# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:37:19 2020

@author: JM the 3rd
"""
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.transforms as tfs

def fallasAleatorias(grafo):
    E=[]
    L=[]
    C=[]
    Tcg=[]
    for i in range(len(grafo)-1):
        nodos=range(len(grafo))
        quitar=np.random.choice(nodos)
        grafo=np.delete(grafo,quitar,0)
        grafo=np.delete(grafo,quitar,1)
        Tcg.append(len(extraerComponenteGigante(grafo)))
        md=calcularMatdeDistancias(grafo)
        E.append(eficiencia(md))
        L.append(longitudCaracteristica(md))
        C.append(np.mean(coeficienteAgrupamiento(grafo)))

    return Tcg,E,L,C

def ataquesDirigidos(grafo):
    a=[]
    E=[]
    L=[]
    C=[]
    Tcg=[]
    for i in range(len(grafo)-1):
        grados=np.sum(grafo,axis=0)
        quitar=np.where(max(grados)==grados)[0]
        grafo=np.delete(grafo,quitar[0],0)
        grafo=np.delete(grafo,quitar[0],1)
        a.append(len(extraerComponenteGigante(grafo)))  
        md=calcularMatdeDistancias(grafo)
        E.append(eficiencia(md))
        L.append(longitudCaracteristica(md))
        C.append(np.mean(coeficienteAgrupamiento(grafo)))
        cg=extraerComponenteGigante(grafo)
        Tcg.append(cg.shape[0])
        
    return Tcg,E,L,C
        

def eficiencia (distancias):
    Dv=distancias[np.logical_not(distancias==0)].ravel()   #Diagonal principal
    return np.mean(1/Dv)

def longitudCaracteristica(D):
    Dv=D[np.logical_not(D==0)].ravel() 
    return np.mean(Dv)

def extraerComponenteGigante(grafo):
    comps=buscarComponentes(grafo)
    compg=comps[0]
    mayor_valor=len(comps[0])
    for comp in comps:
        if len(comp)>mayor_valor:
            compg=comp
            mayor_valor=len(comp)
    nodos_cg=np.sort(np.asarray(compg))
    return grafo[nodos_cg[:,np.newaxis],nodos_cg]
    

def calcularMatdeDistancias(grafo):
    distancias=np.eye(len(grafo))  #Matriz identidad (eye)
    n=1
    nCaminos=grafo.copy()
    caminosDirectos=(nCaminos!=0)    #Booleano
    
    while np.any(caminosDirectos):  #Any: devuelve T o F; aquí se seguirá ejecutando si encuentra algún True 
        distancias += n*caminosDirectos  #El miembro derecho al mutltiplicar un T por 1, devuelve 1 o 0 en caso de ser un F 
        n+=1
        nCaminos=np.dot(nCaminos,grafo)  #Producto punto: informa si es que hay una ruta directa y cuántos nodos debe para para llegar del nodo i al j.  
        caminosDirectos=np.logical_and((nCaminos!=0),(distancias==0)) #Corrobora los contados y cuenta los aún no considerados
        
    distancias[distancias==0]=np.inf
    np.fill_diagonal(distancias,0)

    return distancias

def coeficienteAgrupamiento(grafo):
    k=len(grafo)
    clust_coef_local=np.zeros((k,1))    #Vector 
    for i in range(k):
        vecinos=np.where(grafo[i,:]==1)[0]
        nv=len(vecinos)
        if nv>=2: #Si sólo hay un vecino entonce el coeficiente de agrupamiento es 0 
            subgrafo=grafo[vecinos[:,np.newaxis],vecinos]   #Parear de modo que es el grafo de los vecinos
            clust_coef_local[i]=np.sum(subgrafo)/(nv**2-nv)
    return clust_coef_local
                    
def crearGrafoWattsStrogatz(k,p,v):
    grafo=np.zeros((k,k))
    contador=1
    
    for i in range(k):
        if i>=k-v:
            for j in range(1,v+1-contador):
                grafo[i,i+j]=1
            for j in range(contador):
                grafo[j,i]=1
            contador+=1
        else :
            for j in range(1,v+1):
                grafo[i,i+j]=1      #No hay duplicidad, son todos loe enlaces for
                
    grafo_aux=np.zeros((k,k))
    for i in range(k):
        enlaces=np.where(grafo[i,:]>0)[0]
        for j in range(0,len(enlaces)):
            if np.random.random()>(1-p):
                grafo[i,enlaces[j]]=0
                nuevo_nodo=i
                while (np.isin(nuevo_nodo,np.append(enlaces,i)) 
                              or grafo[nuevo_nodo,i]==1 
                              or grafo_aux[nuevo_nodo,i]==1):
                    nuevo_nodo=np.random.randint(0,k)
                    
                if i<nuevo_nodo:
                    grafo_aux[i,nuevo_nodo]=1
                else:
                    grafo_aux[nuevo_nodo,i]=1
                    
    grafo=grafo+grafo_aux

    for i in range (k):
        for j in range(k):
            grafo[j,i]=grafo[i,j]
    return grafo

    
def buscarComponentes(grafo):
    k=len(grafo) #¿Cuántos nodos hay?
    visitados=[]
    componentes=[]  #lista de listas, corresponden a los nodos que están conecatados
    
    for i in range(k):
        visitados.append(False) #tantos como nodos haya 
        
    for i in range(k):
        if visitados[i]==False:
            temporal=[] 
            componentes.append(buscarEnVecinos(temporal,i,visitados,grafo))
    return componentes 


#Función recursiva
def buscarEnVecinos(temporal,i,visitados,grafo):
    visitados[i]=True
    temporal.append(i) #lista temporal de la neurona que está 
    vecinos=np.where(grafo[i,:]>0)[0]   #Almacena nodos a los cuales esta conectado el nodo i; pueden ser 0
    for j in vecinos :  #Como un for each
        if visitados[j]==False:
            temporal=buscarEnVecinos(temporal,j,visitados,grafo)
    return temporal 
    
    
def verMatAdyacente(grafo):
    plt.figure()
    plt.imshow(grafo,aspect='auto')
    
    
def crearGrafoErdosRenyi(n,p):  #Grafo de red aleatoria 
    grafo_ER=np.zeros((n,n))    #Matriz

    for i in range(n):
        for j in range (i+1,n):     #Sólo lo hace en el triángulo superior...
            if np.random.random()>(1-p):    #1-p porque verdaderamente eso reflejaría el 0.98 de las neuronas 
                grafo_ER[i,j]=1
                grafo_ER[j,i]=1     #...pero como es simétrica la martiz el triangulo sup. = triangulo inf. 

    return grafo_ER

def crearGrafoBarabasiAlbert(m0,m,n):
    grafo=np.ones((m0,m0))
    np.fill_diagonal(grafo,0)

    lista=[]
    for i in range (len(grafo)):
        for j in range (m0-1):
            lista.append(i)
            
    for i in range(m0,n):
        grafo=np.c_[grafo,np.zeros(grafo.shape[0])] #axis 0 =columnas 
        grafo=np.r_[grafo,[np.zeros(grafo.shape[1])]] #axis 1= filas
        temporal=lista
        for j in range(m):
            nodo=np.random.choice(temporal)
            grafo[i,nodo]=1
            grafo[nodo,i]=1
            temporal=list(filter(lambda a:a!=nodo,temporal)) #función <lamba>: actúa como tal en sólo una línea de código;filter sólo toma los verdaderos 
            lista.append(i)
            lista.append(nodo)
    return grafo

def crearGrafo(raster,umbral):
    N,F=raster.shape
    grafo=np.zeros((N,N))   #Matriz adjunta
    coactividad=np.sum(raster,axis=0)
    indices=np.where(coactividad>=umbral)[0]
    for i in indices:
        neuronas=np.where(raster[:,i]==1)[0]
        for j in neuronas :
            for k in neuronas:
                if j!=k:
                    grafo[j,k]=1
    return grafo

def guardarGrafo(nombre_archivo,grafo):
    n=grafo.shape[0]
    with open(nombre_archivo,'w') as objeto_archivo:
        objeto_archivo.write('*Vertices '+str(n))
        for i in range(n):
            objeto_archivo.write('\n'+str(i+1)+' "'+str(i+1).zfill(3)+' "')
        objeto_archivo.write('\n*Edges')
        for i in range(n):
            for j in range(i+1,n):
                if grafo[i,j]==1:
                    objeto_archivo.write('\n'+str(i+1)+' '+str(j+1)+' 1')
        objeto_archivo.close()


def rasterPermutado(raster,opc):
    N,F=raster.shape
    raster_perm=np.zeros((N,F))
    if opc==1:
        for i in range(N):
            raster_perm[i,:]=raster[i,np.random.permutation(F)]
    elif opc==2:
        for j in range(N):  #Recorre las filas 
            IIE=np.diff(np.where(np.concatenate(([1],raster[j,:],[1]))==1)).flatten() #Concatenate: agrega los 1's (cuidado con la dimensión); Where: devuelve los índices donde hay 1's; Diff: resta el siguiente menos el anterior (elementos consecutivos); Flatten:"aplana" el vector, haciendo sólo necesario el uso de un sólo índice; **Las funciones de np regresan arreglos (vectores) tras su aplicación 
            t1=IIE[0]   #Diferencia entre el 1 concatenado y el primer spike 
            t2=IIE[-1]  #Si es negativo toma el último elemento. Esto puede ser visto como un arreglo que no se guarda linealmente sino de manera circular o tipo carrusel 
            IIE=np.delete(IIE,[0])  #Delete: borra los elementos de un arreglo en el índice indicado 
            IIE=np.delete(IIE,[len(IIE)-1]) #El argumento de Delete es un vector del cual se calcula su longitud y se le resta 1
            IIE_aleatorio=IIE[np.random.permutation(len(IIE))] #Revuelvo el orden de los índices con Permutation para luego guardarlos en este nuevo arreglo que cambio de orden los elementos 
            frames_activos=np.concatenate(([0],np.cumsum(IIE_aleatorio))) #Este raster va colocando 1's YA respetando el IIE del IIE_aleatorio   
            frame_inicio=np.random.randint(t1+t2)   
            frames_activos+=frame_inicio-1
            raster_perm[j,frames_activos]=1
    return raster_perm


def rasterMatCoactividad(raster,nsim,opc,choice):
    if choice==1:
        N,F=raster.shape
        matriz_coactividad_perm=np.zeros((nsim,N))
        global max_coactividad
        max_coactividad=0
        
        for i in range(nsim):
            raster_perm=rasterPermutado(raster,opc)
            coactividad_perm=np.sum(raster_perm,axis=0) 
            for j in range(int(max(coactividad_perm))):
                matriz_coactividad_perm[i,j]=len(np.where(coactividad_perm==(j+1))[0]) 
            if int(max(coactividad_perm))>max_coactividad:
                max_coactividad=int(max(coactividad_perm))     
        matriz_coactividad_perm=matriz_coactividad_perm[:,0:max_coactividad]
            
        return matriz_coactividad_perm
    elif choice==2:
        coactividad=np.sum(raster,axis=0) 
        matriz_coactividad_original=np.zeros((nsim,int(max(coactividad))))
        for i in range(nsim):
            for j in range(int(max(coactividad))):
                    matriz_coactividad_original[i,j]=len(np.where(coactividad==(j+1))[0])
        return matriz_coactividad_original
    
def rasterGrafos(mat_coact_perm,mat_coact_orig,nsim,):
    max_coactividad
    vector_grafos=np.zeros((12,1))
    f,c=vector_grafos.shape
    
    for i in range(nsim):
        for j in range(f):
            l=mat_coact_perm[i,:]
            if len(l)>f:
                l=np.delete(mat_coact_perm[0,:],np.s_[12:max_coactividad],axis=0)
            if mat_coact_orig[i,j]>l[j]:
                vector_grafos[j]+=1
    return vector_grafos
            

def rasterPlot(raster,fps,numFig):

        N,F=raster.shape

        plt.figure(num='Raster Plot '+str(numFig),figsize=(12,6)) 
        plt.clf()
        ax=plt.axes((0.05,0.35,0.75,0.6)) #Dibuja la figura con las dimensiones que se le asignan a cada lado de lo que vendría siendo un rectangulo

        for i in range(N):
            indices=np.where(raster[i,:]==1)[0] #Quiero encontrar los índices donde hay unos [0] me introduce dentro de la tupla inemediatamente  
            plt.plot(indices,raster[i,indices]*(1+i),marker='o',linestyle='None',markersize=2,color='black') #plt.plot(x,y) donde x y y son vectores de las misma dimensión 
            
        ax.set_xlim(1,F) #Los valores de los límites en los ejes 
        ax.set_ylim(1,N)
        plt.xticks([])
        plt.ylabel('Etiqueta de Neurona')
        
        ax=plt.axes((0.05,0.09,0.75,0.2))   
        coactividad=np.sum(raster,axis=0)
        fpm=60*fps
        tiempo=np.arange(1,F+1)/fpm #Range(nativo)=lista de números/ Arange=regresa una arreglo 
        plt.plot(tiempo,coactividad,color='black',linewidth=1)
        
        ax.set_xlim(np.min(tiempo),np.max(tiempo))
        ax.set_ylim(np.min(coactividad),np.max(coactividad)+1)
        plt.ylabel(' # de neuronas ')
        plt.xlabel(' Timepo(min) ')
        
        ax=plt.axes((0.85,0.35,0.1,0.6))
        actividad=np.sum(raster,axis=1)*100/F
        #Rotación de la gráfica
        
        base=plt.gca().transData
        rot=tfs.Affine2D().rotate_deg(270)
        
        plt.plot(np.flip(actividad,0), color='black',linewidth=1,transform=rot+base)
        plt.ylabel('% de tiempo activo')
        plt.yticks([])
        ax.set_ylim(-N+1,0)
        # plt.xticks(0,np.max(actividad+1))
        
            
