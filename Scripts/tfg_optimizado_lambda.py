# -*- coding: utf-8 -*-

'''
Módulo de importaciones

- Este módulo se va ampliando conforme avanza el desarrollo
'''
#Importación de los datasets
import os
import sys
#from google.colab import drive

#Carga de datos
from pandas import read_csv #Leer el CSV
from PIL import Image, ImageOps  #Librería de imágenes
import io
import pandas as pd

#Mostrar imágenes
import matplotlib.pyplot as plt

#Para cambiar píxeles imagen
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Activation, Dense, Flatten, Conv2D, MaxPooling2D, 
    concatenate, GlobalAveragePooling2D, Lambda, Input, BatchNormalization, SeparableConv2D
)
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


#Separación de datos
from sklearn import model_selection

#Obtener resultados
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

#Early stopping
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Keras tuner

import keras_tuner as kt
#from keras_tuner.tuners import GridSearch
#from keras_tuner.tuners import Hyperband

#Para el generador
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, caracteristicas1, caracteristicas2, etiquetas, batch_size=200, shuffle=True):
        self.caracteristicas1 = np.array(caracteristicas1)  # Convertir a np.array si no lo es
        self.caracteristicas2 = np.array(caracteristicas2)  # Convertir a np.array si no lo es
        self.etiquetas = np.array(etiquetas)  # Convertir a np.array si no lo es
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.caracteristicas1))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.caracteristicas1) / self.batch_size))  # Usar ceil para manejar el último lote

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X1 = np.stack(self.caracteristicas1[indexes])  # Apilar características 1 en un array
        X2 = np.stack(self.caracteristicas2[indexes])  # Apilar características 2 en un array
        y = self.etiquetas[indexes]  # Obtener las etiquetas correspondientes
        return [X1, X2], y

"""##Módulo de montaje##"""

'''
Módulo de montaje
'''

'''
def montaje_drive():
  drive.mount('/content/gdrive', force_remount=True)
  os.chdir('/content/gdrive/MyDrive/TFG/')
'''
"""##Variables globales##"""


if len(sys.argv) < 5:
    print("Debes proporcionar al menos cuatro argumentos desde la terminal.")
    sys.exit(1)

nuevo_ancho = int(sys.argv[1])
nuevo_alto = nuevo_ancho
canales = int(sys.argv[2])
numero_modelo = int(sys.argv[3])
epocas = int(sys.argv[4])
optimizador = "adam"
perdida = "mean_squared_error"

print(f"Ancho: {nuevo_ancho}")
print(f"Alto: {nuevo_alto}")
print(f"Canales: {canales}")
print(f"Modelo: {numero_modelo}")
print(f"Epocas: {epocas}")


"""##Módulo separación datos##"""

def separar_datos(dataset_entrenar, tamano_validacion, tamano_test, semilla):
  caracteristicas1 = np.stack(dataset_entrenar['primera_imagen'].to_numpy())
  caracteristicas2 = np.stack(dataset_entrenar['segunda_imagen'].to_numpy())
  objetivo = dataset_entrenar['puntuacion'].astype('float32')

  #caracteristicas1_train, caracteristicas1_val, caracteristicas2_train, caracteristicas2_val, etiquetas_train, etiquetas_val = model_selection.train_test_split(
  #    caracteristicas1, caracteristicas2, objetivo, test_size=tamano_test, random_state=semilla)

  caracteristicas1_train, caracteristicas1_temp, caracteristicas2_train, caracteristicas2_temp, etiquetas_train, etiquetas_temp = model_selection.train_test_split(
      caracteristicas1, caracteristicas2, objetivo, test_size=tamano_validacion, random_state=semilla)

  caracteristicas1_val, caracteristicas1_test, caracteristicas2_val, caracteristicas2_test, etiquetas_val, etiquetas_test = model_selection.train_test_split(
      caracteristicas1_temp, caracteristicas2_temp, etiquetas_temp, test_size=tamano_test, random_state=semilla)

  return caracteristicas1_train, caracteristicas1_val, caracteristicas1_test, caracteristicas2_train, caracteristicas2_val, caracteristicas2_test, etiquetas_train, etiquetas_val, etiquetas_test

"""##Módulo creación modelos##

Este modelo sirve como prueba inicial para ver la mejora del rendimiento de los distintos modelos.

###Modelo Básico###
"""

def crear_modelo_basico(hp):
  filtros = hp.Choice('filtros_basico_1', values=[128, 256, 512, 1024])
  filtros2 = hp.Choice('filtros_basico_2', values=[128, 256, 512])
  filtros3 = hp.Choice('filtros_basico_3', values=[128, 256, 512])
  filtros4 = hp.Choice('filtros_basico_4', values=[8, 16, 32, 64, 128, 256])
  filtros5 = hp.Choice('filtros_basico_5', values=[8, 16, 32, 128])
  kernel = hp.Choice('kernels_basico', values=[2])
  neuronas = hp.Choice('neuronas_basico', values=[1024, 2048, 4096])
  neuronas2 = hp.Choice('neuronas_basico_2', values=[32, 64, 128, 256])
  neuronas3 = hp.Choice('neuronas_basico_3', values=[8, 16, 32, 64])
  neuronas4 = hp.Choice('neuronas_basico_4', values=[4, 8, 16, 32])
 
  modelo1 = Sequential([
      Conv2D(filters=filtros, kernel_size=kernel, input_shape=(nuevo_ancho, nuevo_alto, canales), activation='relu'),
      Conv2D(filters=filtros2, kernel_size=kernel, activation='relu'),
      Conv2D(filters=filtros3, kernel_size=kernel, activation='relu'),
      Conv2D(filters=filtros4, kernel_size=kernel, activation='relu'),
      Conv2D(filters=filtros5, kernel_size=kernel, activation='relu'),
      MaxPooling2D(pool_size=(2,2)),
      Flatten()
  ])

  modelo2 = Sequential([
      Conv2D(filters=filtros, kernel_size=kernel, input_shape=(nuevo_ancho, nuevo_alto, canales), activation='relu'),
      Conv2D(filters=filtros2, kernel_size=kernel, activation='relu'),
      Conv2D(filters=filtros3, kernel_size=kernel, activation='relu'),
      Conv2D(filters=filtros4, kernel_size=kernel, activation='relu'),
      Conv2D(filters=filtros5, kernel_size=kernel, activation='relu'),
      MaxPooling2D(pool_size=(2,2)),
      Flatten()
  ])

  concatenacion = concatenate([modelo1.output, modelo2.output])


  capa_densa = Dense(neuronas, activation='relu')(concatenacion)
  capa_densa = Dense(neuronas2, activation='relu')(capa_densa)
  capa_densa = Dense(neuronas3, activation='relu')(capa_densa)
  capa_densa = Dense(neuronas4, activation='relu')(capa_densa)
  capa_output = Dense(1, activation='linear')(capa_densa)


  modeloFinal = Model(inputs=[modelo1.input, modelo2.input], outputs=capa_output)

  modeloFinal.compile(optimizer=optimizador, loss=perdida)

  return modeloFinal

"""###Modelo Estándar###"""

def crear_modelo_estandar(hp):
  filtros = hp.Choice('filtros_estandar', values=[1024, 2048, 4096])
  kernel = hp.Choice('kernels_estandar', values=[8, 16])
  neuronas = hp.Choice('neuronas_estandar', values=[1024, 2048, 4096, 8192])
  
  #filtros2 = hp.Choice('filtros_estandar_2', values=[4, 8, 16, 32, 64])
  #kernel2 = hp.Choice('kernels_estandar_2', values=[2])
  #neuronas2 = hp.Choice('neuronas_estandar_2', values=[4, 8, 16, 32, 64, 128]) 
  
  #filtros3 = hp.Choice('filtros_estandar_3', values=[2, 4, 8, 16, 32, 64])
  #kernel3 = hp.Choice('kernels_estandar_3', values=[2])
  #neuronas3 = hp.Choice('neuronas_estandar_3', values[2, 4, 8, 16, 32, 64])
  
  #filtros4 = hp.Choice('filtros_estandar_4', values=[2, 4, 8, 16, 32])
  #kernel4 = hp.Choice('kernels_estandar_4', values=[2])
  #neuronas4 = hp.Choice('neuronas_estandar_4', values[2, 4, 8, 16, 32, 64])


  modelo1 = Sequential([
      Conv2D(filters=filtros, kernel_size=kernel, padding='same', input_shape=(nuevo_ancho, nuevo_alto, canales), activation='relu'),
      MaxPooling2D(pool_size=(2,2)),
      #Conv2D(filters=filtros2, kernel_size=kernel2, activation='relu'),
      #MaxPooling2D(pool_size=(2,2)),
      #Conv2D(filters=filtros3, kernel_size=kernel3, activation='relu'),
      #MaxPooling2D(pool_size=(2,2)),
      #Conv2D(filters=filtros4, kernel_size=kernel4, activation='relu'),
      #MaxPooling2D(pool_size=(2,2)),
      GlobalAveragePooling2D()
  ])

  modelo2 = Sequential([
      Conv2D(filters=filtros, kernel_size=kernel, padding='same', input_shape=(nuevo_ancho, nuevo_alto, canales), activation='relu'),
      MaxPooling2D(pool_size=(2,2)),
      #Conv2D(filters=filtros2, kernel_size=kernel2, activation='relu'),
      #MaxPooling2D(pool_size=(2,2)),
      #Conv2D(filters=filtros3, kernel_size=kernel3, activation='relu'),
      #MaxPooling2D(pool_size=(2,2)),
      #Conv2D(filters=filtros4, kernel_size=kernel4, activation='relu'),
      #MaxPooling2D(pool_size=(2,2)),
      GlobalAveragePooling2D()
  ])

  concatenacion = concatenate([modelo1.output, modelo2.output])

  capa_densa = Dense(neuronas, activation='relu')(concatenacion)
  #capa_densa = Dense(neuronas2, activation='relu')(capa_densa)
  capa_output = Dense(1, activation='linear')(capa_densa)


  modeloFinal = Model(inputs=[modelo1.input, modelo2.input], outputs=capa_output)

  modeloFinal.compile(optimizer=optimizador, loss=perdida)

  return modeloFinal

"""###Modelo Lambda###"""

def crear_modelo_medio():
  print("Opcion 1")

  input_layer1 = Input(shape=(nuevo_ancho, nuevo_alto, canales))
  input_layer2 = Input(shape=(nuevo_ancho, nuevo_alto, canales))

  modelo1 = Conv2D(filters=2048, kernel_size=8, activation='relu', padding='same')(input_layer1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=512, kernel_size=2, activation='relu', padding='same')(modelo1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo1)
  modelo1 = GlobalAveragePooling2D() (modelo1)


  modelo2 = Conv2D(filters=2048, kernel_size=8, activation='relu', padding='same')(input_layer2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=512, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = GlobalAveragePooling2D() (modelo2)

  #concatenacion = concatenate([modelo1.output, modelo2.output])

  resta = Lambda(lambda x: abs(x[0] - x[1]))([modelo1, modelo2])

  capa_densa = Dense(2048, activation='relu')(resta)
  capa_densa = Dense(1024, activation='relu')(capa_densa)
  capa_densa = Dense(128, activation='relu')(capa_densa)
  #capa_densa = Dense(neuronas4, activation='relu')(capa_densa)
  capa_output = Dense(1, activation='linear')(capa_densa)


  modeloFinal = Model(inputs=[input_layer1, input_layer2], outputs=capa_output)

  modeloFinal.compile(optimizer=optimizador, loss=perdida)

  return modeloFinal

def crear_modelo_medio_2():
  print("Opcion 2")

  input_layer1 = Input(shape=(nuevo_ancho, nuevo_alto, canales))
  input_layer2 = Input(shape=(nuevo_ancho, nuevo_alto, canales))

  modelo1 = Conv2D(filters=2048, kernel_size=4, activation='relu', padding='same')(input_layer1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=512, kernel_size=2, activation='relu', padding='same')(modelo1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo1)

  modelo1 = GlobalAveragePooling2D() (modelo1)


  modelo2 = Conv2D(filters=2048, kernel_size=4, activation='relu', padding='same')(input_layer2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=512, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = GlobalAveragePooling2D() (modelo2)

  #concatenacion = concatenate([modelo1.output, modelo2.output])

  resta = Lambda(lambda x: abs(x[0] - x[1]))([modelo1, modelo2])

  capa_densa = Dense(2048, activation='relu')(resta)
  capa_densa = Dense(1024, activation='relu')(capa_densa)
  capa_densa = Dense(128, activation='relu')(capa_densa)
  #capa_densa = Dense(neuronas4, activation='relu')(capa_densa)
  capa_output = Dense(1, activation='linear')(capa_densa)


  modeloFinal = Model(inputs=[input_layer1, input_layer2], outputs=capa_output)

  modeloFinal.compile(optimizer=optimizador, loss=perdida)

  return modeloFinal

def crear_modelo_medio_3():
  print("Opcion 3")
  input_layer1 = Input(shape=(nuevo_ancho, nuevo_alto, canales))
  input_layer2 = Input(shape=(nuevo_ancho, nuevo_alto, canales))

  modelo1 = Conv2D(filters=2048, kernel_size=4, activation='relu', padding='same')(input_layer1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=512, kernel_size=2, activation='relu', padding='same')(modelo1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo1)
  modelo1 = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same')(modelo1)

  modelo1 = GlobalAveragePooling2D() (modelo1)


  modelo2 = Conv2D(filters=2048, kernel_size=4, activation='relu', padding='same')(input_layer2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=512, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = GlobalAveragePooling2D() (modelo2)

  #concatenacion = concatenate([modelo1.output, modelo2.output])

  resta = Lambda(lambda x: abs(x[0] - x[1]))([modelo1, modelo2])

  capa_densa = Dense(2048, activation='relu')(resta)
  capa_densa = Dense(1024, activation='relu')(capa_densa)
  capa_densa = Dense(128, activation='relu')(capa_densa)
  #capa_densa = Dense(neuronas4, activation='relu')(capa_densa)
  capa_output = Dense(1, activation='linear')(capa_densa)


  modeloFinal = Model(inputs=[input_layer1, input_layer2], outputs=capa_output)

  modeloFinal.compile(optimizer=optimizador, loss=perdida)

  return modeloFinal

def crear_modelo_medio_4():
  print("Opcion 4")
  input_layer1 = Input(shape=(nuevo_ancho, nuevo_alto, canales))
  input_layer2 = Input(shape=(nuevo_ancho, nuevo_alto, canales))

  modelo1 = Conv2D(filters=2048, kernel_size=8, activation='relu', padding='same')(input_layer1)
  modelo1 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo1)
  modelo1 = Conv2D(filters=512, kernel_size=4, activation='relu', padding='same')(modelo1)
  modelo1 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo1)
  modelo1 = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same')(modelo1)
  modelo1 = MaxPooling2D(pool_size=(2,2)) (modelo1)
  modelo1 = GlobalAveragePooling2D() (modelo1)


  modelo2 = Conv2D(filters=2048, kernel_size=8, activation='relu', padding='same')(input_layer2)
  modelo2 = Conv2D(filters=1024, kernel_size=4, activation='relu', padding='same')(modelo2)
  modelo2 = Conv2D(filters=512, kernel_size=4, activation='relu', padding='same')(modelo2)
  modelo2 = Conv2D(filters=256, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same')(modelo2)
  modelo2 = MaxPooling2D(pool_size=(2,2)) (modelo2)
  modelo2 = GlobalAveragePooling2D() (modelo2)

  #concatenacion = concatenate([modelo1.output, modelo2.output])

  resta = Lambda(lambda x: abs(x[0] - x[1]))([modelo1, modelo2])

  capa_densa = Dense(2048, activation='relu')(resta)
  capa_densa = Dense(1024, activation='relu')(capa_densa)
  capa_densa = Dense(128, activation='relu')(capa_densa)
  #capa_densa = Dense(neuronas4, activation='relu')(capa_densa)
  capa_output = Dense(1, activation='linear')(capa_densa)


  modeloFinal = Model(inputs=[input_layer1, input_layer2], outputs=capa_output)

  modeloFinal.compile(optimizer=optimizador, loss=perdida)

  return modeloFinal

"""##Módulo de compilación##"""

'''
Compilación del modelo
'''
def compilar_modelo(modelo, optimizador, perdida):
  modelo.compile(optimizer = optimizador,loss = perdida)

"""
##Módulo de entrenamiento##"""

def entrenar_modelo(modelo, caracteristicas1_train, caracteristicas2_train, etiquetas_train, epocas, batch_size, caracteristicas1_val, caracteristicas2_val, etiquetas_val):
  historial = modelo.fit([caracteristicas1_train, caracteristicas2_train], etiquetas_train , validation_data=(
      [caracteristicas1_val, caracteristicas2_val], etiquetas_val), epochs=epocas, verbose=0, batch_size=batch_size,
             callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=40, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=20, min_delta=0, min_lr=0.00001)]
             )
  return historial

def entrenar_modelo_generador(modelo, epocas, batch_size, train_generator, val_generator):
  historial = modelo.fit(train_generator, validation_data=val_generator, epochs=epocas, verbose=0, batch_size=batch_size,
             callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=40, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=20, min_delta=0, min_lr=0.00001)]
             )
  return historial

"""##Módulo de predicción##"""

def predecir_modelo(modelo, caracteristicas1_test, caracteristicas2_test):
  predicciones = modelo.predict([caracteristicas1_test, caracteristicas2_test])
  return predicciones

"""##Módulo de resultados##"""

'''
Cálculo de las métricas de R cuadrado, MSE, RMSE, MAE, MAPE
'''

def obtener_resultados(etiquetas_test, predicciones):
  mse = mean_squared_error(etiquetas_test, predicciones)
  print(f'MSE: {mse}')

  mae = mean_absolute_error(etiquetas_test, predicciones)
  print(f'MAE: {mae}')

  suma_errors = 0
  epsilon = 0.0001  # Un valor muy pequeño
  for true_val, pred_val in zip(etiquetas_test, predicciones):
      true_val = max(true_val, epsilon)
      suma_errors += abs((true_val - pred_val) / true_val)
  mape2 = (suma_errors / len(etiquetas_test)) * 100
  print(f'MAPE: {mape2}')

  rmse = mean_squared_error(etiquetas_test, predicciones, squared=False)
  print("RMSE:", rmse)

  r2 = r2_score(etiquetas_test, predicciones)
  print("R2:", r2)

  etiquetas_test_numpy = etiquetas_test.values.flatten()  # Convertir el DataFrame a una matriz numpy 1D
  prediccionesFlatten = prediccionesLambda.flatten() 
  no_zero_indices = np.where(etiquetas_test != 0)
  etiquetas_test_filtered = etiquetas_test_numpy[no_zero_indices]
  predicciones_filtered = prediccionesFlatten[no_zero_indices]

  suma_errors = 0
  epsilon = 0.0001  # Un valor muy pequeño
  for true_val, pred_val in zip(etiquetas_test_filtered, predicciones_filtered):
      true_val = max(true_val, epsilon)
      suma_errors += abs((true_val - pred_val) / true_val)
  mape = (suma_errors / len(etiquetas_test_filtered)) * 100
  print(f'MAPE Filtrado: {mape}')

"""##Módulo de ejecución##

###Ejecución carga datos###
"""

def cargar_datos(csv, url_imagenes):
  #Ejecutar módulo de carga de datos, con la opción de csv
  asociaciones = leer_csv(csv, ',', '.')

  #Ejecutar módulo de carga de datos, con la opción de leer de Drive
  dataset_imagenes, imagenes_corruptas = obtener_imagenes(url_imagenes)

  #Ejecutar módulo de carga de datos, con la opción de eliminar las imágenes corruptas
  asociaciones = eliminar_corruptas(asociaciones, imagenes_corruptas)
  return asociaciones, dataset_imagenes


"""###Ejecución de modelos###

####Ejecución modelo básico####
"""

def ejecutar_basico(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto):
  print("Modelo básico")
  #modelo = crear_modelo_basico(kt.HyperParameters())
  #modeloBasico = crear_modelo_basico(nuevo_ancho, nuevo_alto, canales)
  tunerBasico = kt.BayesianOptimization(
    crear_modelo_basico,
    objective='val_loss',
    alpha=0.0000001,
    beta=20,
    overwrite=True
  )

  tunerBasico.search([datos['caracteristicas1_train'], datos['caracteristicas2_train']], datos['etiquetas_train'],
             epochs=50,
             batch_size=batch_size,
             verbose=0,
             validation_data=([datos['caracteristicas1_val'], datos['caracteristicas2_val']], datos['etiquetas_val']),
             callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=40, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=10, min_delta=0, min_lr=0.000001)])


  #compilar_modelo(modeloBasico, "adam", "mean_squared_error")
  #predicciones_basico = predecir_modelo(modeloBasico, datos['caracteristicas1_test'], datos['caracteristicas2_test'])

  mejores_hiperparametros = tunerBasico.get_best_hyperparameters(num_trials=1)[0]
  mejor_modelo_Basico = tunerBasico.hypermodel.build(mejores_hiperparametros)
  entrenar_modelo(mejor_modelo_Basico, datos['caracteristicas1_train'], datos['caracteristicas2_train'], datos['etiquetas_train'], epocas, batch_size, datos['caracteristicas1_val'], datos['caracteristicas2_val'], datos['etiquetas_val'])
  predicciones_basico = predecir_modelo(mejor_modelo_Basico, datos['caracteristicas1_test'], datos['caracteristicas2_test'])
  print("Hiperparámetros: ")
  print(mejores_hiperparametros.values)
  print("Resultados: ")
  obtener_resultados(datos['etiquetas_test'], predicciones_basico)
  return predicciones_basico

"""####Ejecución modelo estándar####"""

def ejecutar_estandar(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto):
  print("Modelo estándar")

  tunerEstandar = kt.BayesianOptimization(
    crear_modelo_estandar,
    objective='val_loss',
    alpha=0.00000001,
    beta=20,
    overwrite=True
  )

  tunerEstandar.search([datos['caracteristicas1_train'], datos['caracteristicas2_train']], datos['etiquetas_train'],
             epochs=5,
             batch_size=batch_size,
             verbose=0,
             validation_data=([datos['caracteristicas1_val'], datos['caracteristicas2_val']], datos['etiquetas_val']),
             callbacks=[
  EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10, restore_best_weights=True, verbose=1),
  ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=10, min_delta=0, min_lr=0.0001)])

  #modeloEstandar = crear_modelo_estandar(nuevo_ancho, nuevo_alto, canales)
  #compilar_modelo(modeloEstandar, "adam", "mean_squared_error")
  #entrenar_modelo(modeloEstandar, datos['caracteristicas1_train'], datos['caracteristicas2_train'], datos['etiquetas_train'], epocas, batch_size, datos['caracteristicas1_val'], datos['caracteristicas2_val'], datos['etiquetas_val'])
  mejores_hiperparametros = tunerEstandar.get_best_hyperparameters(num_trials=1)[0]
  mejor_modelo_estandar = tunerEstandar.hypermodel.build(mejores_hiperparametros)
  entrenar_modelo(mejor_modelo_estandar, datos['caracteristicas1_train'], datos['caracteristicas2_train'], datos['etiquetas_train'], epocas, batch_size, datos['caracteristicas1_val'], datos['caracteristicas2_val'], datos['etiquetas_val'])
  predicciones_estandar = predecir_modelo(mejor_modelo_estandar, datos['caracteristicas1_test'], datos['caracteristicas2_test'])
  print("Hiperparámetros: ")
  print(mejores_hiperparametros.values)
  print("Resultados: ")
  obtener_resultados(datos['etiquetas_test'], predicciones_estandar)
  return predicciones_estandar

"""####Ejecución modelo lambda####"""


def ejecutar_lambda(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto, numero_modelo):

  print("Modelo lambda sin generador")
  if numero_modelo == 2:
    mejor_modelo_lambda = crear_modelo_medio()
  elif numero_modelo == 3:
    mejor_modelo_lambda = crear_modelo_medio_2()
  elif numero_modelo == 4:
    mejor_modelo_lambda = crear_modelo_medio_3()
  elif numero_modelo == 5:
    mejor_modelo_lambda = crear_modelo_medio_4()

  historial = entrenar_modelo(mejor_modelo_lambda, datos['caracteristicas1_train'], datos['caracteristicas2_train'], datos['etiquetas_train'], epocas, batch_size, datos['caracteristicas1_val'], datos['caracteristicas2_val'], datos['etiquetas_val'])
  predicciones_lambda = predecir_modelo(mejor_modelo_lambda, datos['caracteristicas1_test'], datos['caracteristicas2_test'])

  print("Resultados: ")
  obtener_resultados(datos['etiquetas_test'], predicciones_lambda)

  train_losses = historial.history['loss']
  val_losses = historial.history['val_loss']

  plt.figure(figsize=(10, 8)) 
  plt.plot(train_losses, label='Entrenamiento')
  plt.plot(val_losses, label='Validación')
  plt.xlabel('Epocas')
  plt.ylabel('Perdida (Loss)')
  plt.title('Curva de aprendizaje')
  plt.legend()

  plt.savefig('learning_curve' + str(numero_modelo) + '.png')

  plt.figure(figsize=(10, 8))
  plt.scatter(datos['etiquetas_test'], predicciones_lambda, color='blue', alpha=0.6)

  # Añadir etiquetas y título
  plt.title('Gráfico de Dispersión')
  plt.xlabel('Reales')
  plt.ylabel('Predicciones')

  # Agregar una línea diagonal de referencia (idealmente los puntos deberían estar cerca de esta línea)
  plt.plot(datos['etiquetas_test'], datos['etiquetas_test'], color='red', linestyle='--')
  plt.savefig('dispersion' + str(numero_modelo) + '.png')
  mejor_modelo_lambda.save('modeloLambda' + str(numero_modelo) + '.h5')
  return predicciones_lambda


def ejecutar_lambda_generado(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto, numero_modelo):

  print("Modelo lambda")
  if numero_modelo == 2:
    mejor_modelo_lambda = crear_modelo_medio()
  elif numero_modelo == 3:
    mejor_modelo_lambda = crear_modelo_medio_2()
  elif numero_modelo == 4:
    mejor_modelo_lambda = crear_modelo_medio_3()
  elif numero_modelo == 5:
    mejor_modelo_lambda = crear_modelo_medio_4()

  train_generator = DataGenerator(datos['caracteristicas1_train'], datos['caracteristicas2_train'], datos['etiquetas_train'], batch_size=batch_size)
  val_generator = DataGenerator(datos['caracteristicas1_val'], datos['caracteristicas2_val'], datos['etiquetas_val'], batch_size=batch_size)

  historial = entrenar_modelo_generador(mejor_modelo_lambda, epocas, batch_size, train_generator, val_generator)
  predicciones_lambda = predecir_modelo(mejor_modelo_lambda, datos['caracteristicas1_test'], datos['caracteristicas2_test'])

  print("Resultados: ")
  obtener_resultados(datos['etiquetas_test'], predicciones_lambda)

  train_losses = historial.history['loss']
  val_losses = historial.history['val_loss']

  plt.figure(figsize=(10, 8)) 
  plt.plot(train_losses, label='Entrenamiento')
  plt.plot(val_losses, label='Validación')
  plt.xlabel('Epocas')
  plt.ylabel('Perdida (Loss)')
  plt.title('Curva de aprendizaje')
  plt.legend()

  plt.savefig('learning_curve' + str(numero_modelo) + '.png')

  plt.figure(figsize=(10, 8))
  plt.scatter(datos['etiquetas_test'], predicciones_lambda, color='blue', alpha=0.6)

  # Añadir etiquetas y título
  plt.title('Gráfico de Dispersión')
  plt.xlabel('Reales')
  plt.ylabel('Predicciones')

  # Agregar una línea diagonal de referencia (idealmente los puntos deberían estar cerca de esta línea)
  plt.plot(datos['etiquetas_test'], datos['etiquetas_test'], color='red', linestyle='--')
  plt.savefig('dispersion' + str(numero_modelo) + '.png')
  mejor_modelo_lambda.save('modeloLambda' + str(numero_modelo) + '.h5')
  return predicciones_lambda

"""#Selección de la ejecución#

##Selección carga y preprocesamiento##
"""

#montaje_drive()

csv = 'dataset-v2.csv'
#url_imagenes = '/content/gdrive/MyDrive/TFG/imagenes.tgz'
url_imagenes = 'imagenes.tgz'

opciones = [None] * 7
#Tamano
opciones[0] = True
#Gris
if canales == 1:
  opciones[1] = True
else:
  opciones[1] = False
#Normalizar
opciones[2] = True
#Blanco-negro
opciones[3] = True
#Eliminar canal
opciones[4] = False
#Eliminar filas
opciones[5] = False
#Eliminar 0
opciones[6] = False

dataset_entrenar = pd.read_pickle('dataset_entrenar64x64.pkl')


"""##Selección separación datos##"""

tamano_entrenamiento = 0.4
tamano_test = 0.5
semilla = 1

caracteristicas1_train, caracteristicas1_val, caracteristicas1_test, caracteristicas2_train, caracteristicas2_val, caracteristicas2_test, etiquetas_train, etiquetas_val, etiquetas_test = separar_datos(
    dataset_entrenar, tamano_entrenamiento, tamano_test, semilla)

datos = {
    'caracteristicas1_train': caracteristicas1_train,
    'caracteristicas1_val': caracteristicas1_val,
    'caracteristicas1_test': caracteristicas1_test,
    'caracteristicas2_train': caracteristicas2_train,
    'caracteristicas2_val': caracteristicas2_val,
    'caracteristicas2_test': caracteristicas2_test,
    'etiquetas_train': etiquetas_train,
    'etiquetas_val': etiquetas_val,
    'etiquetas_test': etiquetas_test
}
print(dataset_entrenar.shape)

"""##Selección modelo##"""

#epocas=10
batch_size=1000
#canales=1

import tensorflow as tf

with tf.device('/GPU'):
  for i in range(1):
    if numero_modelo == 0:
      prediccionesBasico = ejecutar_basico(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto)
    elif numero_modelo == 1:
      prediccionesEstandar = ejecutar_estandar(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto)
    elif numero_modelo >= 2:
      prediccionesLambda = ejecutar_lambda(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto, numero_modelo)
      #prediccionesLambda = ejecutar_lambda_generado(epocas, batch_size, canales, datos, nuevo_ancho, nuevo_alto, numero_modelo)
