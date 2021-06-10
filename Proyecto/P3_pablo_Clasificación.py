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
##################################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier


np.random.seed(1)

#--------------------------------------------------------------------------------------#
#------------------------------- FUNCIONES PROPIAS ------------------------------------#
#--------------------------------------------------------------------------------------#

#Carga los datos (en forma de matriz) del path que se elija
def CargarDatosClasificacionDe(path):
    with open(path) as File:
        reader = csv.reader(File, delimiter=' ',
                            quoting=csv.QUOTE_MINIMAL , skipinitialspace = True )
        
        x=[]
        y=[]
        
        for row in reader:
                x.append(row)
              
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
print("                   CLASIFICACIÓN")
print('---------------------------------------------------')



#--------------------------------------------------------------------------------------#
#------------------------------- CARGA DE LOS DATOS  ----------------------------------#
#--------------------------------------------------------------------------------------#


path_data = "datos/Sensorless_drive_diagnosis.txt"

data, labels = CargarDatosClasificacionDe(path_data)

separacion_test=0.2

# Separamos los datos en los conjuntos de train y test
    # Gracias a stratify, puedo hacer una separación estratificada de los datos y para ello
    # le paso el conjunto con las etiquetas
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=separacion_test, stratify = labels)


print("\nCon una separacion del", separacion_test ," de los datos test: ")

print("Dimensión X_train: ",X_train.shape)
print("Dimensión X_test: ",X_test.shape)


# Con esto he comprobado que tengo el mismo número de datos por clase
""" 
a = np.zeros(11)
b = np.zeros(11)

for i in Y_train:
    a[int(i)-1] +=1
    
for i in Y_test:
    b[int(i)-1] +=1

for i in range(a.size):
    print("Clase ",i+1," Cuenta: ",a[i])
for i in range(b.size):  
    print("Clase ",i+1," Cuenta: ",b[i])
"""

#--------------------------------------------------------------------------------------#
#--------------------------- VISUALIZACION DE LOS DATOS -------------------------------#
#--------------------------------------------------------------------------------------#
"""
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

print("Es la varianza pero haciendo zoom en el intervalo [0,35] \n")

plt.plot(numeros, varianza, 'red', linewidth=1)
plt.title('Varianza de las características entre el intervalo 0-35')
plt.xlabel("Carácteristicas")
plt.ylabel("Varianza")
plt.ylim(0.0,4.0)
plt.xlim(0.0,35.0)
plt.show()
"""
#--------------------------------------------------------------------------------------#
#------------------------------ ELIMINAR LOS OUTLIERS ---------------------------------#
#--------------------------------------------------------------------------------------#

print("\n------ELIMINAR LOS OUTLIERS------\n")

#n_job=4 debido al elevado tiempo que tarda en hacerlo
LOF = LocalOutlierFactor(n_neighbors=20, n_jobs= 4 ,contamination='auto')

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

input("\n--- Pulsar tecla para continuar ---\n")

"""

#---------------------------------------------------------------------------------#
#--------------------- VALIDACIÓN Y ELECCION DE MODELOS --------------------------#
#---------------------------------------------------------------------------------#

"""

print("\n-------VALIDACIÓN Y ELECCION DE MODELOS-------\n")



####-----   LogisticRegression

parameters = {'max_iter':[100,125], 'solver':['newton-cg','saga'], 'penalty':['l2'] ,
              'C':[1, 10,100,1000,10000],'class_weight': ['balanced']}

print("\nRegresion Logistica")

logic_regression = LogisticRegression()

clf = GridSearchCV(logic_regression, parameters, cv=10, n_jobs=-1 , scoring='balanced_accuracy' ,refit=True)
clf.fit(X_train, Y_train)

print("\nMejores parámetros ->",clf.best_params_,"\n\n")

print("- Balanced_Accuracy (Training): ", balanced_accuracy_score(Y_train, clf.predict(X_train)))
print("- Balanced_Accuracy (Test): ", balanced_accuracy_score(Y_test, clf.predict(X_test)))

print("\nError Validación (del mejor resultado): ",clf.best_score_,"\n\n")

logReg_param=clf.best_params_
logReg_errVal=clf.best_score_

input("\n--- Pulsar tecla para continuar ---\n")

#Debido al problema de los wawrning no deja ver los resultados, por eso lo pongo otra vez

print("\nMejores parámetros ->",clf.best_params_,"\n\n")
print("\nError Validación (del mejor resultado): ",clf.best_score_,"\n\n")



####-----   PassiveAggressiveClassifier


parameters = {'C':[0.001, 0.01, 0.1, 1.0 ], 'max_iter':[800, 1000, 1200 ],
              'class_weight': ['balanced'], 'early_stopping': [True]}
  
    
print("\nPassive Aggressive Classifier")

PA_classifier = PassiveAggressiveClassifier()

clf = GridSearchCV(PA_classifier, parameters, cv=10, n_jobs=-1 , scoring='balanced_accuracy' ,refit=True)
clf.fit(X_train, Y_train)

print("\nMejores parámetros ->",clf.best_params_,"\n\n")

print("- Balanced_Accuracy (Training): ", balanced_accuracy_score(Y_train, clf.predict(X_train)))
print("- Balanced_Accuracy (Test): ", balanced_accuracy_score(Y_test, clf.predict(X_test)))

print("\nError Validación (del mejor resultado): ",clf.best_score_,"\n\n")

"""

#---------------------------------------------------------------------------------#
#------------------------------ HIPOTESIS FINAL ----------------------------------#
#---------------------------------------------------------------------------------#


print('\n---------------------------------------------------')
print("                   Hipótesis Final")
print('---------------------------------------------------')

epsilon=0.02

params_LR = {'C': 1000,\
                 'class_weight': 'balanced',\
                'penalty': 'l2',\
                 'max_iter': 100,\
                 'solver': 'newton-cg',\
                'n_jobs': -1}


print("\nRegresion Logistica")

PA_classifier = LogisticRegression(**params_LR).fit(X_train, Y_train)


print("- Balanced_Accuracy (Training): ", balanced_accuracy_score(Y_train, PA_classifier.predict(X_train)))
print("- Balanced_Accuracy (Test): ", balanced_accuracy_score(Y_test, PA_classifier.predict(X_test)))



input("\n--- Pulsar tecla para continuar ---\n")

params_PA = {'C': 0.01,\
                 'class_weight': 'balanced',\
                'early_stopping': True,\
                 'max_iter': 1200}


print("\nPassive Aggressive Classifier")

PA_classifier = PassiveAggressiveClassifier(**params_PA).fit(X_train, Y_train)


print("- Balanced_Accuracy (Training): ", balanced_accuracy_score(Y_train, PA_classifier.predict(X_train)))
print("- Balanced_Accuracy (Test): ", balanced_accuracy_score(Y_test, PA_classifier.predict(X_test)))


print("\n\n- Cota Eout-> Para", Y_test.size ,"datos de test y para epsilon=",epsilon,
      " Podemos decir que Etest estará dentro +-", epsilon*100,
      '% de Eout con probabilidad {0:3.3f}'.format(CotaEtest(Y_test.size,epsilon) * 100),"%")

