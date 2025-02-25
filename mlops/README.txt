¡Bienvenido!

Este proyecto entrena un modelo para predecir la lluvia en seis ciudades de Australia y permite realizar inferencias con nuevos datos a través de un contenedor Docker. 
El archivo principal a ejecutar es mlops.ipynb. 



>> Estructura del proyecto:

1. Carpeta mlops:

1.1. mlops.ipynb: Notebook principal que define el pipeline (utilizando transformadores personalizados de clases.py), entrena el modelo y ejecuta el contenedor Docker para realizar inferencias.

1.2. Carpeta df_regresion:
1.2.1. weatherAUS.csv: Dataset completo (no se usa directamente para el entrenamiento).
1.2.2. dfs_lluvia.ipynb: Notebook que contiene el script para dividir el dataset completo (weatherAUS.csv) y generar los archivos df_train.csv, df_val.csv y df_test.csv.
1.2.3. df_train.csv, df_val.csv, df_test.csv: Archivos resultantes de la división del dataset. Se usan para entrenar y evaluar el modelo.

1.3. Carpeta docker: Contiene los archivos necesarios para construir y ejecutar el contenedor Docker. 
1.3.1. Dockerfile
1.3.2. inference.py
1.3.3. pipeline.pkl
1.3.4. requirements.txt

1.4. Archivo clases.py: Define los transformadores personalizados (ColumnDropper, LocationFilter, DataFrameImputer, WindComponentsTransformer, DateComponentsTransformer) utilizados en el pipeline.



>> Cómo ejecutar el código:

Requisitos:

1. Es indispensable tener Docker instalado y en ejecución en tu sistema.

2. Modificar la ruta en la siguiente línea para que apunte a la ubicación correcta de la carpeta df_regresion en tu sistema: 
!docker run --rm --name inference-python-test -v "/c/Users/franc/OneDrive/Escritorio/FACU/2025/AA1/mlops/df_regresion:/temp" inference-python-test 

Si la ruta no es correcta, el contenedor no podrá acceder a los datos y fallará la ejecución.



>> Pasos a seguir:

1. Abre mlops.ipynb y ejecuta todas las celdas en orden. El notebook realizará lo siguiente:
1.1. Transformará los datos de df_train.csv aplicando limpieza, imputación, generación de variables y estandarización.
1.2. Entrenará el modelo con los datos transformados.
1.3. Guardará el pipeline entrenado en ./docker/pipeline.pkl.
1.4. Construirá la imagen Docker.
1.5. Ejecutará un contenedor Docker (de forma automática, sin necesidad de comandos manuales) que utiliza el pipeline para realizar inferencias sobre df_test.csv.
1.6. El contenedor generará el archivo output.csv en la carpeta df_regresion con las predicciones.
1.7. Revisa la salida del notebook para ver las métricas de desempeño (R2 y RMSE) y los resultados de las predicciones.

En caso de dudas, revisa los comentarios en el notebook o en los scripts para mayor detalle. ¡Buena suerte y feliz análisis!
