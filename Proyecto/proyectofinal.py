# -*- coding: utf-8 -*-
"""
PROYECTO FINAL
Alejandro Pinel Martínez
Pablo Ruiz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import plot_confusion_matrix, mean_squared_error
from sklearn.svm import LinearSVR

# Fijamos la semilla
np.random.seed(1)

##################### Plot #####################

def PlotGraphic(Y, X=None, title=None, axis_title=None, scale='linear'):
    if (X is not None):
        plt.plot(X, Y)
    else:
        plt.plot(Y)
        
    plt.yscale(scale)
    if (axis_title != None):
        plt.xlabel(axis_title[0])
        plt.ylabel(axis_title[1])
    plt.title(title)
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

def PlotRegressionResult(Y, predictions, title=None):
    fig, ax = plt.subplots()
    ax.scatter(Y, predictions, s=0.5)
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
    ax.set_xlabel('Real label')
    ax.set_ylabel('Predicted')
    if (title is not None):
        plt.title(title)
    plt.show()

##################### Data #####################

#Cargar los datos del problema de regresión
def readDataSDD():
    # Leemos el fichero
    file = 'datos/Sensorless_drive_diagnosis.txt'
    data = np.loadtxt(file)
    x = []
    y = []
    
    for i in range(0, data.shape[0]):
        x.append(np.array([data[i][j] for j in range(data.shape[1]-1)]))
        y.append(np.array([data[i][data.shape[1] - 1]]))
    		
    x = np.array(x, np.float64)
    y = np.array(y, np.int)
    y = toCategorical(y - 1, 11)
    
    return x, y

#Cargar los datos del problema
def readDataRegression():
    # Leemos el fichero
    file = 'datos/train.csv'
    data = np.loadtxt(file, delimiter=',', skiprows=1)
    x = []
    y = []
    
    for i in range(0, data.shape[0]):
        x.append(np.array([data[i][j] for j in range(data.shape[1]-1)]))
        y.append(np.array([data[i][data.shape[1] - 1]]))
    		
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
    
    return x, y


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

##################### Separación Training-Test #####################

#Separar los datos en training, validation y test
def splitData(X, Y, test_percent, val_percent = None, stratify=False):
    if (stratify):
        stratify = Y
    else:
        stratify = None
    
    #Separamos un porcentaje de test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
        test_size=test_percent,
        shuffle=True,                #Aleatorio
        stratify=stratify            #Mantiene las proporciones
        )
    
    if (val_percent is not None):
        #Del conjunto de train, separamos un conjunto de validación
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
            test_size=val_percent,
            shuffle=True,                #Aleatorio
            stratify=stratify            #Mantiene las proporciones
            )
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    return X_train, Y_train, X_test, Y_test

##################### Visualizar Datos #####################

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
    labels = np.array(toLabel(Y), np.int)
    data = []
    unique, counts = np.unique(labels, return_counts=True)
    for val, count in zip(unique, counts):
        data.append([f"{val+1}", count])
    PlotBars(data, "Número de muestras por etiqueta")

# Muestra un boxplot de los datos
def dataBoxPlot(X, Y):
    # Mostramos estadísticas en pantalla
    df_describe = pd.DataFrame(X)
    print(df_describe.describe())
    
    PlotBoxPlot(X, "Data Box-Plot")
    
    

##################### Preprocesamiento de datos #####################

# Eliminamos los datos extremos
def removeOutliers(X, Y, std_factor = 5):
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    lower = mean - std * std_factor
    upper = mean + std * std_factor
    
    mask  = np.zeros(shape=(X.shape[0]), dtype=bool)
    for i in range(X.shape[0]):
        is_outlier = False 
        for j in range(X.shape[1]):
            if (X[i, j] < lower[j] or X[i, j] > upper[j]):
                is_outlier = True
        mask[i] = not is_outlier
    
    X, Y = X[mask, :], Y[mask]
    
    return X, Y

# Probamos distintos valores de varianza para eliminar valores
def ExperimentOutliers(X, Y):
    var_factors = [3, 4, 5, 6, 7, 8, 9, 10]
    percentage = []
    n_data_original = X.shape[0]
    
    for i in var_factors:
        X2, Y2 = removeOutliers(X, Y, i)
        percentage.append((n_data_original - X2.shape[0]) / n_data_original)
    
    PlotGraphic(X=var_factors, Y=percentage, title="Outliers removed", axis_title=("STD Factor", "Percent of data removed"))

#Probaremos cómo cambia el resultadode un modelo simple según el número de parámetros
def ExperimentReduceDimensionality(X, Y, start=6, interval=2, clasification=True):
    # sizes = [i for i in range(start, X.shape[1], interval)]
    sizes = [i for i in range(start, X.shape[1], interval)]
    Ecv = []
    X, Y = removeOutliers(X, Y, 5)
    
    # Generamemos un modelo de regresión Logistica o de regresión
    if (clasification):
        loss='log'
        Y = toLabel(Y)
    else:
        loss='squared_loss'
        Y = np.ravel(Y)
        
    learning_rate='adaptive'
    eta_list = [0.01]
    regularization_list = ['None']
    alpha_list = [0.0001]
    polynomial_degree = 1
    
    for i in sizes:
        # Reducimos la dimensionalidad a lo necesario
        PCA = fitPCA(X, Y, i)
        X_i = PCA.transform(X)
        # Normalizamos los datos
        normalize = generateNormalizer(X_i)
        X_i = normalize(X_i)
        
        #Entrenamos un modelo y nos quedamos con su error de validación
        if (clasification):
            results = linearModelCrossValidation("Linear Regression", X_i, Y, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, verbose=True)
        else:
            results = linearModelCrossValidation("Regression", X_i, Y, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, verbose=True)
        
        Ecv.append(results[1])
    
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
def preproccess(X_train, Y_train, X_test, Y_test, 
                remove_outliers=0, reduce_dimensionality=0, normalize=False, show_info=True):

    # Mostramos los datos iniciales
    if (show_info):
        # dataLabelDistribution(X_train, Y_train)
        dataBoxPlot(X_train, Y_train)
    n_data_original = X_train.shape[0]
    
    # Eliminamos los outliers
    if (remove_outliers > 0):
        X_train, Y_train = removeOutliers(X_train, Y_train, remove_outliers)
        n_data_without_outliers = X_train.shape[0]
        
        if (show_info):
            dataBoxPlot(X_train, Y_train)
            print(f"Se han eliminado {n_data_original - n_data_without_outliers} outliers, el {(n_data_original - n_data_without_outliers)/n_data_original*100:.2f}%")
    
    if (reduce_dimensionality > 0):
        # Reducimos la dimensionalidad
        # PCA = fitPCA(X_train, Y_train, 0.99)
        PCA = fitPCA(X_train, Y_train, reduce_dimensionality)
        X_train = PCA.transform(X_train)
        
        if (show_info):
            dataBoxPlot(X_train, Y_train)
    
    if (normalize):
        # Normalizamos los datos
        normalize = generateNormalizer(X_train)
        X_train = normalize(X_train)
        
        if (show_info):
            dataBoxPlot(X_train, Y_train)
            # dataLabelDistribution(X_train, Y_train)
        
    # Por último, utilizamos el PCA y el normalizer YA AJUSTADOS sobre los datos de test
    # De esta forma nos aseguramos que no se han visto los datos de test al ajustar
    if (reduce_dimensionality > 0):
        X_test = PCA.transform(X_test)
    if (normalize):
        X_test = normalize(X_test)
    
    return X_train, Y_train, X_test, Y_test

##################### Ajuste de modelos lineales #####################

#Devuelve el modelo requerido con los parámetros dados
def generateModel(loss, learning_rate, eta, regularizer, alpha,max_iter = 100000, 
                               seed = 1):
    # Creamos modelo con los parámetros adecuados
    # Dependiendo de si es clasificación o regresión usaremos SGDClassifier o SGDRegressor
    # REGRESION
    if (loss=='squared_loss'):
        model   = SGDRegressor(loss=loss, 
                            learning_rate=learning_rate,
                            eta0 = eta, 
                            penalty=regularizer, 
                            alpha = alpha, 
                            max_iter = max_iter,
                            shuffle=True, 
                            fit_intercept=True, 
                            random_state = seed)
    # Caso especial, SVR
    elif (learning_rate=='SVR'):
        model   = LinearSVR(loss = loss,
                            epsilon=eta,
                            C = alpha, 
                            max_iter = max_iter,
                            random_state = seed)
    # CLASIFICACION
    else:
        model   = SGDClassifier(loss=loss, 
                            learning_rate=learning_rate,
                            eta0 = eta, 
                            penalty=regularizer, 
                            alpha = alpha, 
                            max_iter = max_iter,
                            shuffle=True, 
                            fit_intercept=True, 
                            random_state = seed,
                            n_jobs=-1)
    return model

# Realiza una selección de parámetros de un modelo lineal de clasificación usando cross-validation
# Devuelve una lista con el mejor modelo, mejores parámetros y sus resultados
def linearModelCrossValidation(name, X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, 
                               regularization_list, alpha_list, max_iter = 100000, 
                               seed = 1, verbose=True):
    
    # La función de scoring según si es regresión o clasificación
    if (loss=='squared_loss' or learning_rate=='SVR'):
        scoring = 'neg_mean_squared_error'
    else:
        scoring = 'accuracy'
    
    # Realizamos las transformaciones polinomiales necesarias
    poly  = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    
    if (verbose):
        print (f"\n{name}: Training set {X_train.shape}")
    
    # En esta lista guardaremos el mejor modelo = 
    # [nombre, accuracy_cv, accuracy_train, copy of the model, parameters]
    best_model = ["", -1e6, 0, 0, []]
    
    for regularizer in regularization_list:
        for eta in eta_list:
            for alpha in alpha_list:
                # Creamos modelo con los parámetros adecuados
                model = generateModel(loss, learning_rate, eta, regularizer, alpha, max_iter, seed)
                
                # Calculamos resultados con 5-Fold Cross Validation
                results = cross_validate(model, X_train, Y_train, 
                                         scoring=scoring,
                                         cv=5,       
                                         return_train_score=True,
                                          n_jobs=-1
                                         )
                
                parameters = [loss, learning_rate, polynomial_degree, eta, regularizer, alpha]
                
                # Vamos guardando el mejor modelo
                if(results['test_score'].mean() > best_model[1]):
                    best_model[0] = name
                    best_model[1] = results['test_score'].mean()
                    best_model[2] = results['train_score'].mean()
                    best_model[3] = model
                    best_model[4] = parameters
                    
                        
                if (verbose):
                    print (f"Resultados {parameters}: Ein {results['train_score'].mean()} Ecv: {results['test_score'].mean()}")
                
                #Si estamos probando sin regularización, no tiene sentido probar los alphas, solo probamos una vez
                if (regularizer == 'None'):
                    break
    if (verbose):
        print(f"Mejores parámetros para {name}: {best_model[4]}")
        print(f"Ecv: {best_model[1]} Ein {best_model[2]}")
    
    return best_model

def linearModelTrain(name, X_train, Y_train, loss, learning_rate, polynomial_degree, eta, 
                               regularizer, alpha, max_iter = 100000, 
                               seed = 1):
    
    # Realizamos las transformaciones polinomiales necesarias
    poly  = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    
    # Creamos modelo con los parámetros adecuados
    model = generateModel(loss, learning_rate, eta, regularizer, alpha, max_iter, seed)
    
    # Ajustamos el modelo con todos los datos
    results = model.fit(X_train, Y_train)
    
    return model
    
# Simplemente devuelve el mejor modelo de una lista de resultados
def GetBestModel(results_list):
    best_model = results_list[0]
    for result in results_list:
        if (result[1] > best_model[1]):
            best_model = result
    return best_model

##################### Modelos de clasificación #####################

def SelectBestModelClassification(X_train, Y_train, verbose=True):
    results = []
    max_iter = 100000
    
    ################### Regresión Logística ###################
    
    # Estamos usando regresión logística, luego la función de perdida será la logística
    loss='log'
    learning_rate = 'adaptive'
    
    # Parámetros que probaremos
    eta_list = [0.1, 0.01, 0.001]
    regularization_list = ['None', 'l1', 'l2']
    alpha_list = [0.001, 0.0001, 0.00001]
    
    #Versión sin transformaciones
    polynomial_degree = 1
    results.append(linearModelCrossValidation("Logistic Regression", X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, max_iter=max_iter))
    
    #Versión con transformaciones polinomicas
    polynomial_degree = 2
    results.append(linearModelCrossValidation("Logistic Regression Polynomical Transformation", X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, max_iter=max_iter))
    
    ################### Perceptron ###################
    
    # Estamos usando perceptron, usamos su función de pérdida
    loss='perceptron'
    learning_rate = 'constant'
    
    # Parámetros que probaremos
    eta_list = [1]
    regularization_list = ['None']
    alpha_list = [0]
    
    #Versión sin transformaciones
    polynomial_degree = 1
    results.append(linearModelCrossValidation("Perceptron", X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, max_iter=max_iter))
    
    #Versión con transformaciones polinomicas
    polynomial_degree = 2
    results.append(linearModelCrossValidation("Perceptron  Polynomical Transformation", X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, max_iter=max_iter))
    
    ################### SVM ###################
    
    # Estamos usando SVM, usamos su función de pérdida
    loss='hinge'
    learning_rate = 'adaptive'
    
    # Parámetros que probaremos
    eta_list = [0.1, 0.01, 0.001]
    regularization_list = ['None', 'l1', 'l2']
    alpha_list = [0.001, 0.0001, 0.00001]
    
    #Versión sin transformaciones
    polynomial_degree = 1
    results.append(linearModelCrossValidation("SVM", X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, max_iter=max_iter))
    
    #Versión con transformaciones polinomicas
    polynomial_degree = 2
    results.append(linearModelCrossValidation("SVM  Polynomical Transformation", X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, max_iter=max_iter))
    
    
    ################### Selección del mejor modelo ###################
    
    best_model = GetBestModel(results)
    
    if (verbose):
        print(f"El mejor modelo es: {best_model[0]} con parámetros: {best_model[4]}")
        print(f"Ein {best_model[2]} Ecv: {best_model[1]}")
    
    return best_model

def DefinitiveModelClassification(X_train, Y_train, X_test, Y_test, definitive_model):
    #Extraemos todos los parámetros
    max_iter = 100000
    name = definitive_model[0]
    loss, learning_rate, polynomial_degree, eta, regularizer, alpha = definitive_model[4]
    
    # Entrenamos a nuestro modelo definitivo
    model = linearModelTrain(name, X_train, Y_train, loss, learning_rate, polynomial_degree, eta, regularizer, alpha, max_iter=max_iter)
    
    poly  = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    # Resultados sobre los datos de train
    train_error = model.score(X_train, Y_train)
    
    # Resultados sobre los datos de test
    X_test = poly.fit_transform(X_test)
    test_error = model.score(X_test, Y_test)
    
    print(f"\nMODELO DEFINITIVO: {name}")
    print(f"Ein: {train_error}")
    print(f"Etest: {test_error}")
    
    #Imprimimos la matriz de confusión
    generateConfusionMatrix(X_test, Y_test, model)

##################### Modelos de regresión #####################

def SelectBestModelRegression(X_train, Y_train, verbose=True):
    results = []
    max_iter = 100000
    
    ################### Regresión ###################
    
    # Estamos usando regresión logística, luego la función de perdida será la logística
    loss='squared_loss'
    learning_rate = 'adaptive'
    
    # Parámetros que probaremos
    eta_list = [0.1, 0.01, 0.001, 0.0001]
    regularization_list = ['None', 'l1', 'l2']
    alpha_list = [0.01, 0.001, 0.0001, 0.00001]
    
    #Versión sin transformaciones
    polynomial_degree = 1
    results.append(linearModelCrossValidation("Linear Regression", X_train, Y_train, loss, learning_rate, polynomial_degree, eta_list, regularization_list, alpha_list, max_iter=max_iter))
    ################### SVR ###################
    
    # Estamos usando regresión logística, luego la función de perdida será la logística
    loss='epsilon_insensitive'
    learning_rate = 'SVR'
    
    # Parámetros que probaremos
    epsilon_list = [0, 0.1, 1, 2, 5, 8, 10]
    regularization_list = ['-']
    C_list = [10, 1, 0.5, 0.1, 0.01]
    
    polynomial_degree = 1
    results.append(linearModelCrossValidation("SVR", X_train, Y_train, loss, learning_rate, polynomial_degree, epsilon_list, regularization_list, C_list, max_iter=max_iter))
    
    
    ################### Selección del mejor modelo ###################
    
    best_model = GetBestModel(results)
    
    if (verbose):
        print(f"El mejor modelo es: {best_model[0]} con parámetros: {best_model[4]}")
        print(f"Ein {best_model[2]} Ecv: {best_model[1]}")
    
    return best_model

def DefinitiveModelRegression(X_train, Y_train, X_test, Y_test, definitive_model):
    #Extraemos todos los parámetros
    max_iter = 100000
    name = definitive_model[0]
    loss, learning_rate, polynomial_degree, eta, regularizer, alpha = definitive_model[4]
    
    # Entrenamos a nuestro modelo definitivo
    model = linearModelTrain(name, X_train, Y_train, loss, learning_rate, polynomial_degree, eta, regularizer, alpha, max_iter=max_iter)
    
    poly  = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    # Resultados sobre los datos de train
    train_predictions = model.predict(X_train)
    train_error = mean_squared_error(train_predictions, Y_train)
    
    # Resultados sobre los datos de test
    X_test = poly.fit_transform(X_test)
    test_predictions = model.predict(X_test)
    test_error  = mean_squared_error(test_predictions, Y_test)
    
    print(f"\nMODELO DEFINITIVO: {name}")
    print(f"Ein: {train_error}")
    print(f"Etest: {test_error}")
    
    # Dibujamos gráfico del resultado
    PlotRegressionResult(Y_train, train_predictions, "Resultado Regresión Entrenamiento")
    PlotRegressionResult(Y_test, test_predictions, "Resultado Regresión Test")





def clasificationProblem():
    print("Clasificación")
    
    # Cargamos los datos
    X, Y = readDataSDD()
    # Separamos Training y Test
    X_train, Y_train, X_test, Y_test = \
        splitData(X, Y, test_percent=0.2, stratify=True)
    
    print("Conjunto de datos originales: ")
    print(f"Train: {X_train.shape} {Y_train.shape}")
    print(f"Test: {X_test.shape} {Y_test.shape}")
    

    # Experimentamos con los outliers
    # ExperimentOutliers(X_train, Y_train)
    
    # Experimentamos con la reducción de la dimensionalidad
    # ExperimentReduceDimensionality(X_train, Y_train)
    
    X_train, Y_train, X_test, Y_test = preproccess(X_train, Y_train, X_test, Y_test,
                                         remove_outliers=5, reduce_dimensionality=14,
                                         normalize=True, show_info=True)
    
    print("Datos tras el preprocesamiento: ")
    print(f"Train: {X_train.shape} {Y_train.shape}")
    print(f"Test: {X_test.shape} {Y_test.shape}")
    
    Y_train = toLabel(Y_train)
    Y_test = toLabel(Y_test)
    classification_model = SelectBestModelClassification(X_train, Y_train)
    DefinitiveModelClassification(X_train, Y_train, X_test, Y_test, classification_model)

def regressionProblem():
    print("Regresión")
    
    # Cargamos los datos
    X, Y = readDataRegression()
    # Separamos Training y Test
    X_train, Y_train, X_test, Y_test = \
        splitData(X, Y, test_percent=0.2)
    
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)
        
    print("Conjunto de datos originales: ")
    print(f"Train: {X_train.shape} {Y_train.shape}")
    print(f"Test: {X_test.shape} {Y_test.shape}")
    
    # Experimentamos con los outliers
    # ExperimentOutliers(X_train, Y_train)
    
    # Experimentamos con la reducción de la dimensionalidad
    # ExperimentReduceDimensionality(X_train, Y_train, clasification=False)
    
    X_train, Y_train, X_test, Y_test = preproccess(X_train, Y_train, X_test, Y_test,
                                         remove_outliers=5, reduce_dimensionality=70,
                                         normalize=True, show_info=True)
    
    print("Datos tras el preprocesamiento: ")
    print(f"Train: {X_train.shape} {Y_train.shape}")
    print(f"Test: {X_test.shape} {Y_test.shape}")
    
    regression_model = SelectBestModelRegression(X_train, Y_train)
    DefinitiveModelRegression(X_train, Y_train, X_test, Y_test, regression_model)

##################### Main #####################

def main():
    print("Práctica 3 - Alejandro Pinel Martínez")
    
    clasificationProblem()
    regressionProblem()
    
    
    
if __name__ == '__main__':
  main()