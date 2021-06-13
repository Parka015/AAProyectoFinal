# -*- coding: utf-8 -*-
"""
PROYECTO FINAL
Alejandro Pinel Martínez
Pablo Ruiz Mingorance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import plot_confusion_matrix, balanced_accuracy_score
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
from statistics import mean
import itertools

# Fijamos la semilla
np.random.seed(1)

#------------------------------------------------------------------------#
#------------------------------- Plot -----------------------------------#
#------------------------------------------------------------------------#

def PlotGraphic(Y, X=None, title=None, c='red' , line=2 ,axis_title=None, scale='linear', ylim=None):
    fig, ax = plt.subplots()
    if (X is not None):
        ax.plot(X, Y, c=c , linewidth=line)
    else:
        ax.plot(Y, c=c , linewidth=line)
        
    ax.set_yscale(scale)
    if (axis_title != None):
        ax.set_xlabel(axis_title[0])
        ax.set_ylabel(axis_title[1])
    
    if (ylim is not None):
        ax.set_ylim(ylim)
        
    ax.set_title(title)
    plt.show()

def plotData(tsne_result, y=None, title=None):
    fig, ax = plt.subplots()
    
    if (y is not None):
        i = 0
        for value in np.unique(y):
            indices = np.where(y == value)
            ax.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                       label=(value+1))
            i += 1
        ax.legend()
    else:
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1])
    
    
    ax.set_title(title)
    plt.show()
    
def PlotBoxPlot(X, title=None):
    fig, ax = plt.subplots()
    bp = ax.boxplot(X)
    ax.set_title(title)
    # ax.set_xticks(np.arange(0, X.shape[1]+1, 5))
    plt.show()
    
# Muestra gráfico de barras. Los datos deben ser ("title", value) o ("title", value, color)
def PlotBars(data, title=None, y_label=None, dateformat=False):
    strings = [i[0] for i in data]
    x = [i for i in range(len(data))]
    y = [i[1] for i in data]
    
    colors=None
    if (len(data[0]) > 2):
        colors = [i[2] for i in data]
    
    fig, ax = plt.subplots()
    
    if (title is not None):
        ax.set_title(title)
    if (y_label is not None):
        ax.set_ylabel(y_label)
    
    if (dateformat):
        fig.autofmt_xdate()
        
    x_labels=strings
    plt.xticks(x, x_labels)
    
    if (colors is not None):
        plt.bar(x, y, color=colors)
    else:
        plt.bar(x, y)
    plt.show()
    
# Genera y muestra en pantalla la matriz de confusión de un modelo
def generateConfusionMatrix(X_test, Y_test, model):
    preds = model.predict(X_test)
    conf_matrix = plot_confusion_matrix(model, X_test, Y_test)
    plt.show()

#------------------------------------------------------------------------#
#------------------------------- Data -----------------------------------#
#------------------------------------------------------------------------#

#Lee un fichero y devuelve una matriz
def readFile(filename, data_type=np.float64):
    data = np.loadtxt(filename)
    x = []
    
    for i in range(0, data.shape[0]):
        if (len(data.shape) == 1):
            x.append(data[i])
        elif (len(data.shape) == 2):
            x.append(np.array([data[i][j] for j in range(data.shape[1])]))
    		
    x = np.array(x, data_type)
    
    return x

#Leemos los datos del problema
def readDataHARS():
    X_1 = readFile('Datos/test/X_test.txt')
    X_2 = readFile('Datos/train/X_train.txt')
    X = np.concatenate((X_1, X_2), axis=0)
    
    Y_1 = readFile('Datos/test/Y_test.txt', np.int)
    Y_2 = readFile('Datos/train/Y_train.txt', np.int)
    Y = np.concatenate((Y_1, Y_2), axis=0)
    
    N_1 = readFile('Datos/test/subject_test.txt', np.int)
    N_2 = readFile('Datos/train/subject_train.txt', np.int)
    N = np.concatenate((N_1, N_2), axis=0)
    
    return X, Y, N

# Devuelve una matriz categorical para las etiquetas
def toCategorical(y, n_classes):
    cat = np.zeros((y.size, n_classes), dtype=float)
    for i in range(y.size):
        cat[i][y[i]] = 1
    return cat

# Inversa de categorical, devuelve un vector con las etiquetas
def toLabel(y):
    label = np.zeros((y.shape[0]), dtype=float)
    for i in range(y.shape[0]):
        label[i] = np.where(y[i][:] == 1)[0]
    return label

#------------------------------------------------------------------------#
#---------------------- Separación Train-Test ---------------------------#
#------------------------------------------------------------------------#

#Separar los datos en training y test
def splitData(X, Y, N, test_percent):
    n_individuals = np.max(N) - np.min(N) + 1
    test_individuals = np.random.choice(n_individuals, int(n_individuals * test_percent))
    
    mask = np.zeros_like(N, dtype=bool)
    for i in range(N.shape[0]):
        if (N[i]-1 in test_individuals):
            mask[i] = True
        else:
            mask[i] = False
    
    X_test = X[mask]
    Y_test = Y[mask]
    N_test = N[mask]
    X_train = X[~mask]
    Y_train = Y[~mask]
    N_train = N[~mask]
    
    #Permutamos los datos para que estén 
    X_train, Y_train, N_train = ShuffleData(X_train, Y_train, N_train)
    X_test, Y_test, N_test = ShuffleData(X_test, Y_test, N_test)
    
    # print (f"Train: {X_train.shape} {Y_train.shape}")
    # print (f"Test: {X_test.shape} {Y_test.shape}")
    
    return X_train, Y_train, N_train, X_test, Y_test, N_test

def ShuffleData(X, Y, N=None):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]
    if (N is not None):
        N = N[randomize]
        return X, Y, N
    return X, Y

#------------------------------------------------------------------------#
#-------------------- Información Sobre datos ---------------------------#
#------------------------------------------------------------------------#

# Muestra distintos estadísticos sobre los datos
def DataInformation(X_train, Y_train):
    #Distribución de las clases
    dataLabelDistribution(X_train, Y_train)
    
    #Información sobre los atributos
    showStatisticPlot(X_train, title="Media", par_estad='mean', ylim=[-1.01, 1.01])
    showStatisticPlot(X_train, title="Máximo", par_estad='max')
    showStatisticPlot(X_train, title="Mínimo", par_estad='min')
    showStatisticPlot(X_train, title="Desviación Típica", par_estad='std', ylim=[0.01, 1.01])
    
    
    

#Funcion para mostrar una gráfica sobre un parámetro estadistico
    # par_estad -> Sirve para especificar que parámetro estadistico mostrar (mean|var|max|min|std)
    # axis -> Sirve para especificar si se calculará por columnas (axis=0) o por filas (axis=1)
def showStatisticPlot( X, par_estad='mean' , axis=0, title=None, c='red' , line=2 ,axis_title=None , scale='linear', ylim=None ):
    
    numeros = np.arange(X[0].size)
    
    if par_estad == 'mean':
        media = X.mean(axis)
        PlotGraphic(media, title=title, c=c , line=line ,axis_title=axis_title, scale=scale, ylim=ylim)
        
    elif par_estad == 'var':    #Varianza
        varianza = X.var(axis)
        PlotGraphic(varianza, title=title, c=c , line=line ,axis_title=axis_title, scale=scale, ylim=ylim)
        
    elif par_estad == 'max':
        maximo = X.max(axis)
        PlotGraphic(maximo, title=title, c=c , line=line ,axis_title=axis_title, scale=scale, ylim=ylim)
    
    elif par_estad == 'min':
        minimo = X.min(axis)
        PlotGraphic(minimo, title=title, c=c , line=line ,axis_title=axis_title, scale=scale, ylim=ylim)
        
    elif par_estad == 'std':    #Desviacion estandar 
        std = X.std(axis)
        PlotGraphic(std, title=title, c=c , line=line ,axis_title=axis_title, scale=scale, ylim=ylim)
    

# tSNE para visualizar los datos
def tSNE(X, Y):
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
    tsne_results = tsne.fit_transform(X, Y)
    return tsne_results

# PCA para visualizar los datos
def applyPCA(X, Y, n_components=2):
    pca = fitPCA(X, Y, n_components)
    X = pca.transform(X)
    pca_results = pca.fit_transform(X, Y)
    return pca_results

# Ajustamos un PCA para reducir la dimensionalidad
def fitPCA(X, Y, n_components=0.95):
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X, Y)
    return pca

# PCA para reducir la dimensionalidad
def reduceDimensionality(X, Y, std_percent=0.95):
    print(f"Matriz inicial: {X.shape} std: {X.std()}")
    pca = PCA(n_components=std_percent, svd_solver='full')
    new_X = pca.fit_transform(X, Y)
    print(f"Matriz final: {new_X.shape} std: {new_X.std()}")
    return new_X

# Muestra un gráfico con el número de muestras de cada etiqueta
def dataLabelDistribution(X, Y):
    #Contamos el número de muestras de cada etiqueta
    labels = np.array(Y, np.int)
    data = []
    unique, counts = np.unique(labels, return_counts=True)
    for val, count in zip(unique, counts):
        data.append([f"{val}", count])
    PlotBars(data, "Número de muestras por etiqueta")

# Muestra un boxplot de los datos
def dataBoxPlot(X, Y):
    # Mostramos estadísticas en pantalla
    df_describe = pd.DataFrame(X)
    print(df_describe.describe())
    
    PlotBoxPlot(X, "Data Box-Plot")

# Muestra un gráfico con el número de muestras de cada individuo
def individualsDistribution(N):
    #Contamos el número de muestras de cada etiqueta
    labels = N
    data = []
    unique, counts = np.unique(labels, return_counts=True)
    for val, count in zip(unique, counts):
        data.append([f"{val}", count])
    PlotBars(data, "Número de muestras por individuo")
    

#------------------------------------------------------------------------#
#----------------------- Preprocesamiento -------------------------------#
#------------------------------------------------------------------------#

#Funcion para eliminar outliers usando los k-vecinos
    # neighbors -> Para controlar el parámetro k (cuantos vecinos)
    # ctm -> Valor de contaminacion
def outliersEliminationK_Neightbours(X, Y, N, neighbors=20 ,ctm='auto'):
    
    LOF = LocalOutlierFactor(n_neighbors=neighbors, n_jobs=-1, metric='manhattan' ,contamination=ctm)

    # ············ Eliminar los outliers sobre los datos de entrenamiento ············#
    
    LOF.fit(X,Y)
    
    # Generamos un vector con valores mas cercanos a -1 cuanto más nos se aprpoxima el dato a la media.
    outliers = LOF.negative_outlier_factor_
    
    # Eliminamos los datos que consideramos demasiado alejados (menores que -1.5) 
    X = X[outliers > -1.5]
    Y = Y[outliers > -1.5]
    N = N[outliers > -1.5]
    
    return X, Y, N

#Probaremos cómo cambia el resultadode un modelo simple según el número de parámetros
def ExperimentReduceDimensionality(X, Y, N, start=1, end=-1, interval=2, clasification=True):
    if (end == -1):
        end = X.shape[1]
    sizes = [i for i in range(start, end, interval)]
    Ecv = []
    X, Y, N = outliersEliminationK_Neightbours(X, Y, N)
    
    # Generamemos un modelo de regresión Logistica
    loss='log'
    learning_rate='adaptive'
    eta = 0.01
    regularization = 'None'
    alpha = 0.0001
    
    for i in sizes:
        # Reducimos la dimensionalidad a lo necesario
        PCA = fitPCA(X, Y, i)
        X_i = PCA.transform(X)
        # Normalizamos los datos
        normalize = generateNormalizer(X_i)
        X_i = normalize(X_i)
        
        #Entrenamos un modelo y nos quedamos con su error de validación
        model   = SGDClassifier(loss=loss, 
                        learning_rate=learning_rate,
                        eta0 = eta, 
                        penalty=regularization, 
                        alpha = alpha, 
                        max_iter = 100000,
                        shuffle=True, 
                        fit_intercept=True, 
                        n_jobs=-1)
        
        print(f"Experimento: dimensión {i}")
        results = crossValidationIndividuals(X_i, Y, N, model, verbose=1)
        
        Ecv.append(results[0])
    
    PlotGraphic(X=sizes, Y=Ecv, title="Ecv over dimension", axis_title=("Dimension", "Ecv"))
    
# Generamos un normalizador con los datos de train
def generateNormalizer(original_data):
    mean = original_data.mean(axis = 0)
    std = original_data.std(axis = 0)
    def normalize(new_data):
        result = new_data - mean
        result = result / std
        return result
    return normalize

# Preprocesamos los datos eliminando outliers, reduciendo la dimensionalidad y normalizando
def fitPreproccesser(X_train, Y_train, N_train, remove_outliers=True, reduce_dimensionality=0, normalize=True, show_info=True):

    n_data_original = X_train.shape[0]
    if (remove_outliers):
        X_train, Y_train, N_train = outliersEliminationK_Neightbours(X_train, Y_train, N_train)
        n_data_without_outliers = X_train.shape[0]
        
        if (show_info):
            print(f"Se han eliminado {n_data_original - n_data_without_outliers} outliers, el {(n_data_original - n_data_without_outliers)/n_data_original*100:.2f}%")
    
    # Ajustamos PCA
    if (reduce_dimensionality > 0):
        PCA = fitPCA(X_train, Y_train, reduce_dimensionality)
        X_train = PCA.transform(X_train)
    
    # Ajustamos normalizador
    if (normalize):
            normalize = generateNormalizer(X_train)
    
    def preproccess(X, Y, N, is_test=False):
        
        # Eliminamos los outliers
        if (remove_outliers and not is_test):
            X, Y, N = outliersEliminationK_Neightbours(X, Y, N)
        
        # Reducimos dimensionalidad
        if (reduce_dimensionality > 0):
            X = PCA.transform(X)
        
        # Normalizamos los datos
        if (normalize):
            X = normalize(X)
        
        return X, Y, N
    
    return preproccess

#------------------------------------------------------------------------#
#-------------------------- Ajuste de modelos ---------------------------#
#------------------------------------------------------------------------#

#Debido a las caracter
def crossValidationIndividuals(X, Y, N, model, cv=5, verbose=0):
    # Primero, dividimos los datos en K folds 
    # SEPARANDO POR INDIVIDUOS
    X_fold = []
    Y_fold = []
    
    individuals = np.unique(N)
    np.random.shuffle(individuals)
    N_fold = np.array_split(individuals, cv)
    
    for fold in range(len(N_fold)):
        mask = np.zeros_like(N, dtype=bool)
        for i in range(N.shape[0]):
            if (N[i]-1 in N_fold[fold]):
                mask[i] = True
            else:
                mask[i] = False
                
        X_fold.append(X[mask])
        Y_fold.append(Y[mask])
    
    ecv = []
    ein = []
    times = []
    
    # Ahora entrenamos con k-1 de los folds
    for fold in range(len(N_fold)):
        
        # Unimos todos los folds menos uno
        X_i = None
        Y_i = None
        for i in range(len(N_fold)):
            if (fold != i):
                if (X_i is None):
                    X_i = X_fold[i]
                    Y_i = Y_fold[i]
                else:
                    X_i = np.concatenate((X_i, X_fold[i]), axis=0)
                    Y_i = np.concatenate((Y_i, Y_fold[i]), axis=0)
        
        X_i, Y_i = ShuffleData(X_i, Y_i)
        
        model = clone(model)
        
        start_time = time.time()
        
        #Ahora entrenamos con los datos
        model.fit(X_i, Y_i)
        
        end_time = time.time()
        seconds = end_time - start_time
        times.append(seconds)
        
        # Resultados sobre los datos de train
        train_predict = model.predict(X_i)
        train_error = balanced_accuracy_score(Y_i, train_predict)
        ein.append(train_error)
        
        # Resultados sobre los datos de validación
        val_predict = model.predict(X_fold[fold])
        val_error = balanced_accuracy_score(Y_fold[fold], val_predict)
        ecv.append(val_error)
        
        if (verbose >= 2):
            print(f"Fold {fold} Training {X_i.shape} Validation {X_fold[fold].shape}")
            print(f"Ecv {val_error} Ein {train_error} Tiempo {seconds}")
    
    if (verbose >= 1):
        print(f"Resultados Medios CV: Ecv {mean(ecv)} Ein {mean(ein)} Tiempo {mean(times)}")
    
    return [mean(ecv), mean(ein), mean(times)]


#Hace la funcion equivalente a GridSearchCV, se proporcionan unos datos(x) unas etiquetas (Y)
#Una lista del usuario al que corresponde a cada dato y etiqueta y un diccionario de parámetros
def gridSearch(X, Y, N, model, dictParameters, verbose=0): #-> errorValidation, errorTrain, tiempoMedio, parametros
 
    parametros=[]
    valores=[]
    
    
    for i in dictParameters:
        # if len(dictParameters[i]) > 1:  #Para aquellas claves que tengan más de un valor,
            parametros.append(i)            # la clave se guarda en parametros
            valores.append(dictParameters[i])   # Y los 'valores' que puede tomar en la variable valores
    
    combinaciones=list(itertools.product(*valores)) # Generamos todas las combinaciones de los distintos
                                                # valores de los parámetros. Devuelve una lista de tuplas
    
    diccionario_parametros=dictParameters.copy()
    
    results = []
    
    for combinacion in combinaciones:       #Para cada combinacion (es una tupla), actualizaremos los valores que deben
        for j in range(len(parametros)):            #tomar las claves del diccionario (los parámetros)
            aux = {parametros[j]:combinacion[j]}
            diccionario_parametros.update(aux)  #Actualizamos el valor asociado a una clave
        
        model.set_params(**diccionario_parametros)
        
        # Realizamos cross-validation
        results.append(crossValidationIndividuals(X, Y, N, model))
        results[-1].append(diccionario_parametros)
        
        if (verbose >= 2):
            print(f"Parámetros: {diccionario_parametros}")
            print(f"Resultados Medios CV: Ecv {results[-1][0]} Ein {results[-1][1]} Tiempo {results[-1][2]}")
    
    # Seleccionamos la mejor versión
    best_model = results[0]
    for i in results:
        if (i[0] > best_model[0]):
            best_model = i
            
    return best_model
        

def fitBestParameters(name, X_train, Y_train, N_train, model, parameters, preprocesser, verbose=0):
    # Preprocesamos los datos
    X_train, Y_train, N_train = preprocesser(X_train, Y_train, N_train, is_test=False)
    
    if (verbose >= 1):
        print (f"\n{name}: Training set {X_train.shape}")
    
    # # Generamos una grid search
    # clf = GridSearchCV(model, parameters, cv=5, n_jobs=-1, pre_dispatch=4, verbose=4, return_train_score=True)
    # # Ajustamos el modelo
    # clf.fit(X_train, Y_train)
    
    results = gridSearch(X_train, Y_train, N_train, model, parameters, verbose-1)
    
    if (verbose >= 1):
        print(f"{name} mejores parámetros:\n{results[3]}")
        print(f"Ecv: {results[0]} Ein: {results[1]}")
        print(f"Tiempo: {results[2]} seg")
    
    # En esta lista guardaremos el mejor modelo = 
    # [nombre, accuracy_cv, copy of the model, parameters]
    return [name, results[0], clone(model), results[3], preprocesser]
    


#------------------------------------------------------------------------#
#----------------------- Modelos escogidos ------------------------------#
#------------------------------------------------------------------------#

# Simplemente devuelve el mejor modelo de una lista de resultados
def GetBestModel(results_list):
    best_model = results_list[0]
    for result in results_list:
        if (result[1] > best_model[1]):
            best_model = result
    return best_model

def SelectBestModel(X_train, Y_train, verbose=True):
    results = []
    max_iter = 100000
    
    preprocessador1 = fitPreproccesser(X_train, Y_train, reduce_dimensionality=160)
    preprocessador2 = fitPreproccesser(X_train, Y_train, reduce_dimensionality=0)

    ################### Regresión Logística ###################
    
    parameters = {'max_iter':[100000], 
                  'loss':['log'],
                  'learning_rate':['adaptive'],
                  
                  'penalty':['l1', 'l2'],
                  'alpha':[0, 0.001, 0.0001, 0.00001],
                  'eta0':[0.1, 0.01, 0.001],
                  }
    
    model = SGDClassifier()
    # results.append(fitBestParameters("Regresión Logística - 160", X_train, Y_train, model, parameters, preprocessador1))
    
    model = SGDClassifier()

    #results.append(fitBestParameters("Regresión Logística - 561", X_train, Y_train, model, parameters, preprocessador2))

    ########################  SVM  ########################
    
    parameters = {'max_iter':[100000],
                  'cache_size':[200, 400],
                  'class_weight': ['balanced'],
                  'kernel':['poly'],
                  'degree':[1, 2, 3, 4, 5],
                  'C':[0.1, 1, 10, 100, 1000, 10000, 100000]}
    
    model = SVC()
    # results.append(fitBestParameters("SVC - 160", X_train, Y_train, model, parameters, preprocessador1))
    
    model = SVC()
    # results.append(fitBestParameters("SVC - 561", X_train, Y_train, model, parameters, preprocessador2))
    
    """
    ################### Perceptron Multicapa ###################
    
    parameters = {'max_iter':[100000],
                  'learning_rate':['adaptive'],
                  
                  'activation':['tanh', 'relu'],
                  'hidden_layer_sizes':[[50, 50], [100, 50], [100, 100]],
                  'alpha':[0, 1, 0.1, 0.01, 0.001, 0.0001],
                  'learning_rate_init':[0.1, 0.01, 0.001],
                  }
    
    model = MLPClassifier()
    results.append(fitBestParameters("Perceptron Multicapa - 160", X_train, Y_train, model, parameters, preprocessador1))
    
    model = MLPClassifier()
    results.append(fitBestParameters("Perceptron Multicapa - 561", X_train, Y_train, model, parameters, preprocessador2))

    ################### Random Forest ###################
    
    
    parameters = {'criterion':['gini'],
                  'min_samples_split':[2],
                  'min_samples_leaf':[1],
                  'max_features':['sqrt'],
                  'min_impurity_decrease':[0],
                  'bootstrap':[True],
                  'class_weight': ['balanced'],
                  
                  'n_estimators':[10, 100, 500, 1000],
                  'max_depth':[None, 25, 50],
                  }
    
    model = RandomForestClassifier()
    # results.append(fitBestParameters("Random Forest - 160", X_train, Y_train, model, parameters, preprocessador1))
    
    model = RandomForestClassifier()
    # results.append(fitBestParameters("Random Forest - 561", X_train, Y_train, model, parameters, preprocessador2))
    
    """   
    
    ################### Selección del mejor modelo ###################
    
    best_model = GetBestModel(results)
    
    if (verbose):
        print(f"\nEl mejor modelo es: {best_model[0]} con parámetros: {best_model[3]}")
        print(f"Ecv: {best_model[1]}")
    
    return best_model

def TrainTestDefinitiveModel(X_train, Y_train, X_test, Y_test, model, name, preprocesser):
    
    #Primero, preprocesamos los datos con el preprocesser
    X_train, Y_train = preprocesser(X_train, Y_train)
    X_test, Y_test = preprocesser(X_test, Y_test, is_test=True)
    
    # Entrenamos a nuestro modelo definitivo
    results = model.fit(X_train, Y_train)
    
    # Resultados sobre los datos de train
    train_predict = model.predict(X_train)
    train_error = balanced_accuracy_score(Y_train, train_predict)
    # train_error = model.score(X_train, Y_train)
    
    # Resultados sobre los datos de test
    test_predict = model.predict(X_test)
    test_error = balanced_accuracy_score(Y_test, test_predict)
    # test_error = model.score(X_test, Y_test)
    
    print(f"\nMODELO DEFINITIVO: {name}")
    print(f"Ein: {train_error}")
    print(f"Etest: {test_error}")
    
    #Imprimimos la matriz de confusión
    generateConfusionMatrix(X_test, Y_test, model)




#------------------------------------------------------------------------#
#------------------------------- MAIN -----------------------------------#
#------------------------------------------------------------------------#


def main():
    print("Proyecto Final")
    
    X_all, Y_all, N = readDataHARS()
    X_train, Y_train, N_train, X_test, Y_test, N_test = splitData(X_all, Y_all, N, 0.3)
    print("Conjunto de datos originales: ")
    print(f"Train: {X_train.shape} {Y_train.shape}")
    print(f"Test: {X_test.shape} {Y_test.shape}")
    
    # loss='log'
    # learning_rate='adaptive'
    # eta = 0.01
    # regularization = 'None'
    # alpha = 0.0001
    # model   = SGDClassifier(loss=loss, 
    #                     learning_rate=learning_rate,
    #                     eta0 = eta, 
    #                     penalty=regularization, 
    #                     alpha = alpha, 
    #                     max_iter = 100000,
    #                     shuffle=True, 
    #                     fit_intercept=True, 
    #                     n_jobs=-1)
    
    # crossValidationIndividuals(X_train, Y_train, N_train, model, cv=5, verbose=2)
    
    # ExperimentReduceDimensionality(X_train, Y_train, N_train, start=5, end=-1, interval=10)
    # ExperimentReduceDimensionality(X_train, Y_train, N_train, start=2, end=100)
    
    
    parameters = {'max_iter':[100000], 
                  'loss':['log'],
                  'learning_rate':['adaptive'],
                  
                  'penalty':['l1', 'l2'],
                  'alpha':[0.001],
                  'eta0':[0.1, 0.01],
                  }
    
    model = SGDClassifier()
    
    preprocessador1 = fitPreproccesser(X_train, Y_train, N_train, reduce_dimensionality=160)
    
    fitBestParameters("RL", X_train, Y_train, N_train, model, parameters, preprocessador1, verbose=1)
    
    return 0
    
    
    
    individualsDistribution(N)
    DataInformation(X_train, Y_train)
    
    # Seleccionamos los mejores parámetros con grid search y cross validation
    # y nos quedamos con la mejor hipótesis
    SelectBestModel(X_train, Y_train);
    
    
    preprocessador1 = fitPreproccesser(X_train, Y_train, N_train, reduce_dimensionality=160)
    
    X_train_preprocessed, Y_train_preprocessed = preprocessador1(X_train, Y_train)
    DataInformation(X_train_preprocessed, Y_train_preprocessed)
    
    # Experimentamos con la dimensionalidad para obtener el valor ideal de reducción
    # ExperimentReduceDimensionality(X_train, Y_train, N_train, start=6, end=-1, interval=5)
    # ExperimentReduceDimensionality(X_train, Y_train, N_train, start=2, end=100)
    
    # La experimentación nos ha mostrado que la mejor hipótesis es el MLP
    best_hypotesis = MLPClassifier(max_iter=100000, learning_rate='adaptive',
                                   activation='tanh', hidden_layer_sizes=[100, 50],
                                   learning_rate_init=0.01, alpha=0.001)

    # TrainTestDefinitiveModel(X_train, Y_train, X_test, Y_test, best_hypotesis, "Perceptron Multicapa - 160", preprocessador1)
    
    
    
if __name__ == '__main__':
  main()