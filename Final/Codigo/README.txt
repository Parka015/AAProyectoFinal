-|PROYECTO FINAL|-

Autores:
	-Alejandro Pinel Martínez
	-Pablo Ruiz Mingorance

Para ejecutar nuestro código es necesario seguir los siguientes pasos:

  Descomprimir la carpeta Codigo que está entregada junto a esta memoria.
  
  Descargarse los datos de Human Activity Recognition Using Smartphones de esta página de UCI:
  https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
  
  Dentro de la página, seleccionar la opción Data Folder y la opción UCI HAR Dataset.zip.
  
  Descomprimir los datos descargados en el directorio Codigo/Datos/.
  
  Por último, ejecutar el archivo Codigo/proyectofinal.py

La estructura final de directorios debe ser esta:


- Codigo/
    - Datos/
        - test/
            - subject_test.txt
            - X_test.txt
            - y_test.txt
            - ...
        - train/
            - subject_test.txt
            - X_train.txt
            - y_train.txt
            - ...
    - proyectofinal.py

En el código, primero se realizará un análisis de los datos iniciales, imprimiendo algunas de las gráficas del primer y segundo apartado. 

Después, se realizará el preprocesado descrito en la memoria y se mostrarán las gráficas del resultado.

Se entrenará la mejor hipótesis (descrita en la memoria) con los datos de train y se probará con los de test. Se mostrarán gráficas de los resultados.

El proceso de selección de hiperparámetros y selección de la mejor hipótesis está comentado, debido a que es un proceso que tarda mucho. Si se quiere realizar, se puede descomentar la llamada a la función \textbf{SelectBestModel}.

Si se quiere ejecutar alguno de los otros experimentos, basta con descomentar la llamada a su respectiva función en el main.