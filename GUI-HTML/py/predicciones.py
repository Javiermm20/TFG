#pip install keras numpy pandas Pillow
#tensorflow 2.8.2
#Python 3.7.9

#tensorflow 2.11.0 para que vaya el modelo básico

'''
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Carga el modelo desde el archivo h5
modelo = keras.models.load_model('modelo_entrenado.h5')

# Ejemplo de datos para hacer una predicción
# Asegúrate de que tus datos de entrada coincidan con lo que el modelo espera
datos_de_entrada = np.array([[0.1, 0.2, 0.3]])  # Por ejemplo, un array de 1x3 para un modelo de clasificación

# Realiza la predicción
prediccion = modelo.predict(datos_de_entrada)

# Imprime la predicción
print("Predicción:", prediccion)


def formatear_tamano_imagenes(imagenes, nuevo_ancho, nuevo_alto):
  for i in range(len(imagenes)):
    tamaño_despues_preprocesado = (nuevo_ancho, nuevo_alto)
    imagenes[i] = imagenes[i].resize(tamaño_despues_preprocesado, Image.LANCZOS)
  return imagenes

def pasar_a_array(imagen):
    imagen_array = img_to_array(imagen)
    return imagen_array

def castear_a_tensor(dataset_entrenar):
  dataset_entrenar['primera_imagen'] = dataset_entrenar['primera_imagen'].apply(pasar_a_array)
  dataset_entrenar['segunda_imagen'] = dataset_entrenar['segunda_imagen'].apply(pasar_a_array)
  return dataset_entrenar

def normalizacion(dataset_entrenar):
  for i in range(len(dataset_entrenar['primera_imagen'])):
    dataset_entrenar['primera_imagen'][i] = dataset_entrenar['primera_imagen'][i] / 255
    dataset_entrenar['segunda_imagen'][i] = dataset_entrenar['segunda_imagen'][i] / 255
  return dataset_entrenar

def intercambio_blanco_negro(dataset_entrenar):
  for i in range(len(dataset_entrenar['primera_imagen'])):
    dataset_entrenar['primera_imagen'][i] = 1 - dataset_entrenar['primera_imagen'][i]
    dataset_entrenar['segunda_imagen'][i] = 1 - dataset_entrenar['segunda_imagen'][i]
  return dataset_entrenar
'''
#def preprocesar(dataset_imagenes, nuevo_ancho, nuevo_alto, opciones):
'''
  Cambiar tamaño imágenes
  
  '''
  #dataset_imagenes['Imagen'] = formatear_tamano_imagenes(dataset_imagenes['Imagen'], nuevo_ancho, nuevo_alto)

'''
  Castear a tensor
  '''
  #dataset_entrenar = castear_a_tensor(dataset_entrenar)

'''
  Normalización
  '''
  #dataset_entrenar = normalizacion(dataset_entrenar)

'''
  Paso de blanco a negro
  '''
  #dataset_entrenar = intercambio_blanco_negro(dataset_entrenar)

  #return dataset_entrenar


#modeloPrediccion = keras.models.load_model('../pesos/modeloLambda.h5')

import sys
import json

import json
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps
import base64
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def json_a_dataset(datos):
  ids = []
  imagenes = []

  # Itera sobre el diccionario JSON
  for item in datos:
      # Extrae el ID y los datos de imagen binarios
      _id = item['_id']
      imagen_datos = item['datos']

      imagen_64 = base64.b64decode(imagen_datos)
      imagen_pil = Image.open(BytesIO(imagen_64))

      # Agrega el ID y la imagen a las listas
      ids.append(_id)
      imagenes.append(imagen_pil)

  # Crea un DataFrame de pandas con los IDs y las imágenes
  df = pd.DataFrame({'ID': ids, 'Imagen': imagenes})
  return df

def formatear_tamano_imagenes(imagenes, nuevo_ancho, nuevo_alto):
  imagenes_procesadas = []
    
  for img in imagenes:
      img_procesada = img.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
      imagenes_procesadas.append(img_procesada)
  return imagenes_procesadas

def pasar_a_array(imagen):
  return img_to_array(imagen)

def castear_a_tensor(imagenes):
  imagenes_tensor = imagenes.apply(pasar_a_array)

  return imagenes_tensor

def normalizacion(imagenes):
  imagenes_normalizadas = imagenes/ 255

  return imagenes_normalizadas

def intercambio_blanco_negro(imagenes):
  imagenes_blanco_negro = 1 - imagenes

  return imagenes_blanco_negro

def conversion_escala_grises(imagenes):
  imagenes_gris = [ImageOps.grayscale(img) for img in imagenes]
  return imagenes_gris



try:
    moleculaSeleccionada = sys.argv[1]
    datos = json.loads(sys.stdin.read())
    dataset = json_a_dataset(datos)
    from platform import python_version
    #print(python_version())
    #print("Tensorflow: " + tf.__version__)
    
    modelo = keras.models.load_model('../GUI-HTML/pesos/pesosBasico.h5')
    #print(moleculaSeleccionada)
    #print(dataset)

    dataset['Imagen'] = formatear_tamano_imagenes(dataset['Imagen'], 32, 32)
    dataset['Imagen'] = conversion_escala_grises(dataset['Imagen'])
    dataset['Imagen'] = castear_a_tensor(dataset['Imagen'])
    dataset['Imagen'] = normalizacion(dataset['Imagen'])
    dataset['Imagen'] = intercambio_blanco_negro(dataset['Imagen'])

    imagen_molecula_seleccionada = dataset.loc[dataset['ID'] == moleculaSeleccionada, 'Imagen'].values[0]
    imagenes_otras_moleculas = dataset.loc[dataset['ID'] != moleculaSeleccionada, 'Imagen'].tolist()

    otros_ids = dataset.loc[dataset['ID'] != moleculaSeleccionada, 'ID'].tolist()

    tuplas = [(imagen_molecula_seleccionada, img) for img in imagenes_otras_moleculas]

    resultados = []
    for idx, tupla in enumerate(tuplas):
      imagen_seleccionada = moleculaSeleccionada
      imagen_otra = otros_ids[idx]

      imagen_1_expanded = np.expand_dims(tupla[0], axis=0)
      imagen_2_expanded = np.expand_dims(tupla[1], axis=0)

      prediccion = modelo.predict([imagen_1_expanded, imagen_2_expanded], verbose=0)

      prediccion_lista = prediccion.tolist()

      resultados.append({
          'ID_1': imagen_seleccionada,
          'ID_2': imagen_otra,
          'Prediccion': prediccion_lista
      })

    print(json.dumps(resultados))


except Exception as e:
    sys.stderr.write(f'Error en el script Python: {str(e)}\n')
    sys.exit(1)