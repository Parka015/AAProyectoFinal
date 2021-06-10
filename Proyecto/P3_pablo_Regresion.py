# -*- coding: utf-8 -*-
"""
Created on Fri May 28 22:42:50 2021

@author: Pablo
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

np.random.seed(1)

#--------------------------------------------------------------------------------------#
#------------------------------- FUNCIONES PROPIAS ------------------------------------#
#--------------------------------------------------------------------------------------#

#Carga los datos (en forma de matriz) del path que se elija
def CargarDatosRegresionDe(path):
    with open(path) as File:
        reader = csv.reader(File, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL , skipinitialspace = True )
        
        x=[]
        y=[]
        
        primera_linea = True
        
        for row in reader:
            if not primera_linea:
                x.append(row)
            primera_linea = False
        
        
        x=np.array(x,np.float64)
        
        y = x[:,x[1].size-1]
        y=np.array(y,np.float64)

        
        #Elimino la ultima columna ya que son las etiquetas
        x = np.array(x[:,0:x[1].size-1:1])
        
        return x,y 

def CotaEtest(N,epsilon):
    cota = 1 - 2* np.exp(-2*N*epsilon*epsilon)
    return cota 

#####################################################################################




print('\n---------------------------------------------------')
print("                   REGRESSION")
print('---------------------------------------------------')



#--------------------------------------------------------------------------------------#
#------------------------------- CARGA DE LOS DATOS  ----------------------------------#
#--------------------------------------------------------------------------------------#


path_data = "datos/train.csv"

data, labels = CargarDatosRegresionDe(path_data)

separacion_test=0.2

# Separamos los datos en los conjuntos de train y test
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=separacion_test)

print("\nCon una separacion del", separacion_test ," de los datos test: ")

print("Dimensión X_train: ",X_train.shape)
print("Dimensión X_test: ",X_test.shape)


#--------------------------------------------------------------------------------------#
#--------------------------- VISUALIZACION DE LOS DATOS -------------------------------#
#--------------------------------------------------------------------------------------#

numeros = np.arange(X_train[0].size)

varianza = X_train.var(0)
minimo = X_train.min(0)
maximo = X_train.max(0)
    

plt.plot(numeros, varianza, 'red', linewidth=2)
plt.title('Varianza de cada una de las características')
plt.xlabel("Carácteristicas")
plt.ylabel("Varianza")
plt.show()

plt.plot(numeros, minimo, 'blueviolet', linewidth=2)
plt.title('Valor Minimo de cada una de las características')
plt.xlabel("Carácteristicas")
plt.ylabel("Minimo")
plt.show()

plt.plot(numeros, maximo,'lightseagreen', linewidth=2)
plt.title('Valor Máximo de cada una de las características')
plt.xlabel("Carácteristicas")
plt.ylabel("Maximo")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Son las mismas medidas estadisticas pero acotadas en el eje X e Y\n")

plt.plot(numeros, varianza, 'red', linewidth=1)
plt.title('Varianza de cada una de las características(Eje Y limitado) 1º Mitad')
plt.xlabel("Carácteristicas")
plt.ylabel("Varianza")
plt.ylim(0.0,2000.0)
plt.xlim(0.0,X_train[0].size/2)
plt.show()

plt.plot(numeros, varianza, 'red', linewidth=1)
plt.title('Varianza de cada una de las características(Eje Y limitado) 2º Mitad')
plt.xlabel("Carácteristicas")
plt.ylabel("Varianza")
plt.ylim(0.0,2000.0)
plt.xlim(X_train[0].size/2, X_train[0].size)
plt.show()

plt.plot(numeros, minimo, 'blueviolet', linewidth=2)
plt.title('Valor Minimo de cada una de las características(Eje Y limitado)')
plt.xlabel("Carácteristicas")
plt.ylabel("Minimo")
plt.ylim(-3.0,10.0)
plt.show()

plt.plot(numeros, maximo,'lightseagreen', linewidth=2)
plt.title('Valor Máximo de cada una de las características (Eje Y limitado)')
plt.xlabel("Carácteristicas")
plt.ylabel("Maximo")
plt.ylim(0.0,500.0)
plt.show()




#--------------------------------------------------------------------------------------#
#------------------------------ ELIMINAR LOS OUTLIERS ---------------------------------#
#--------------------------------------------------------------------------------------#

print("\n------ELIMINAR LOS OUTLIERS------\n")


LOF = LocalOutlierFactor(n_neighbors=20, contamination='auto')

# ············ Eliminar los outliers sobre los datos de entrenamiento ············#

LOF.fit(X_train,Y_train)

# Generamos un vector con valores mas cercanos a -1 cuanto más nos se aprpoxima el dato a la media.
outliers = LOF.negative_outlier_factor_

# Eliminamos los datos que consideramos demasiado alejados (menores que -1.5) 
X_train = X_train[outliers > -1.5]
Y_train = Y_train[outliers > -1.5]

print("Dimensión X_train (sin outliers): ",X_train.shape)

input("\n--- Pulsar tecla para continuar ---\n")

#--------------------------------------------------------------------------------------#
#---------------------------------- NORMALIZACIÓN -------------------------------------#
#--------------------------------------------------------------------------------------#

print("\n------NORMALIZACIÓN------\n")

scaler = StandardScaler()

#Normalizamos los datos de entrenamiento
X_train = scaler.fit_transform(X_train)

#Normalizamos test con la desciacion y media de los datos de training
X_test = scaler.transform(X_test)

print("\nNormalización realizada\n")

#----------------------------------------------------------------------------#
#---------------------------------- PCA -------------------------------------#
#----------------------------------------------------------------------------#

"""
print("Antes:", X_train.shape)

pca = PCA(n_components=0.99)
X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

print("Reducido:", X_train.shape)

"""

#---------------------------------------------------------------------------------#
#--------------------- VALIDACIÓN Y ELECCION DE MODELOS --------------------------#
#---------------------------------------------------------------------------------#
"""
print("\n-------VALIDACIÓN Y ELECCION DE MODELOS-------\n")

parameters = {'max_iter':[1000,5000,10000,15000], 'alpha':[0,0.0001,0.001,0.01,0.1, 0.5,1.0]}

print("\nLasso")

lasso_prueba = Lasso()


clf = GridSearchCV(lasso_prueba, parameters, cv=10, n_jobs = 4, pre_dispatch = 4, refit=True)
clf.fit(X_train, Y_train)

#print("\n\n",clf.cv_results_,"\n\n")
print("\nMejores parámetros ->",clf.best_params_,"\n\n")

#print("\n\n",clf.best_score_,"\n\n")

print("- R^2 Training: ", clf.score(X_train,Y_train))
print("- R^2 Test: ", clf.score(X_test,Y_test))
print("- RMSE (Training): ", mean_squared_error(Y_train, clf.predict(X_train), squared=False))
print("- RMSE (Test): ", mean_squared_error(Y_test, clf.predict(X_test), squared=False))

print("\nError Validación (del mejor resultado): ",clf.best_score_,"\n\n")

input("\n--- Pulsar tecla para continuar ---\n")

print("\n\nRidge")

ridge_prueba = Ridge()

parameters = {'max_iter':[500,1000,3000,5000, 8000], 'alpha':[0,0.0001,0.001,0.01,0.1, 0.5, 1.0]}

clf = GridSearchCV(ridge_prueba, parameters, cv=10, refit=True)
clf.fit(X_train, Y_train)

print("\nMejores parámetros ->",clf.best_params_,"\n\n")

print("- R^2 Training: ", clf.score(X_train,Y_train))
print("- R^2 Test: ", clf.score(X_test,Y_test))
print("- RMSE (Training): ", mean_squared_error(Y_train, clf.predict(X_train), squared=False))
print("- RMSE (Test): ", mean_squared_error(Y_test, clf.predict(X_test), squared=False))

print("\nError Validación (del mejor resultado): ",clf.best_score_,"\n\n")

input("\n--- Pulsar tecla para continuar ---\n")
"""

#---------------------------------------------------------------------------------#
#------------------------------ HIPOTESIS FINAL ----------------------------------#
#---------------------------------------------------------------------------------#
print('\n---------------------------------------------------')
print("                   Hipótesis Final")
print('---------------------------------------------------')

epsilon=0.03
print("\nLasso")

# Usando Regresión Lasso
lasso = Lasso(alpha=0.0001, max_iter=15000).fit(X_train, Y_train)

print("- R^2 (Training): ", lasso.score(X_train,Y_train))
print("- R^2 (Test): ", lasso.score(X_test,Y_test))
print("- RMSE (Training): ", mean_squared_error(Y_train, lasso.predict(X_train), squared=False))
print("- RMSE (Test): ", mean_squared_error(Y_test, lasso.predict(X_test), squared=False))


input("\n--- Pulsar tecla para continuar ---\n")

print("\nRidge (hipótesis final")

# Usando Regresión Ridge 
ridge = Ridge(alpha=0.01, max_iter=500).fit(X_train, Y_train)


print("- R^2 (Training): ", ridge.score(X_train,Y_train))
print("- R^2 (Test): ", ridge.score(X_test,Y_test))
print("- RMSE (Training): ", mean_squared_error(Y_train, ridge.predict(X_train), squared=False))
print("- RMSE (Test): ", mean_squared_error(Y_test, ridge.predict(X_test), squared=False))


print("\n\n- Cota Eout-> Para", Y_test.size ,"datos de test y para epsilon=",epsilon,
      " Podemos decir que Etest estará dentro +-", epsilon*100,
      '% de Eout con probabilidad {0:3.3f}'.format(CotaEtest(Y_test.size,epsilon) * 100),"%")


print("\n\nRango de valores de la etiqueta (Grados Kelvin) Max: ",
      np.max(labels),
      "Min: ",np.min(labels))

#---------------------------------------------------------------------------------#
#----------------------- VISUALIZACIÓN HIPOTESIS FINAL ---------------------------#
#---------------------------------------------------------------------------------#

#-------- Training

plt.plot(np.arange(50, 150), Y_train[50:150:1], 'red', linewidth=2)
plt.plot(np.arange(50, 150), ridge.predict(X_train)[50:150:1], 'lightseagreen', linewidth=2)
plt.xlabel('Datos')
plt.ylabel('Valor Etiqueta')
plt.title('Ridge - Datos Training')
plt.show()


#-------- Test

plt.plot(np.arange(50, 150), Y_test[50:150:1], 'red', linewidth=2)
plt.plot(np.arange(50, 150), ridge.predict(X_test)[50:150:1], 'lightseagreen', linewidth=2)
plt.xlabel('Datos')
plt.ylabel('Valor Etiqueta')
plt.title('Ridge - Datos Test')
plt.show()



