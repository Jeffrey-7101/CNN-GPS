import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt #visualizar imagenes y el resultado de las predicciones

# Cargar el conjunto de datos MNIST,  consta de un gran número de imágenes en escala de grises 
    # de dígitos escritos a mano, que van desde el 0 al 9. 
    # Cada imagen tiene una resolución de 28x28 píxeles,
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Visualizar algunas imágenes de entrenamiento
plt.figure(figsize=(10, 10))
for i in range(25):  # Muestra las primeras 25 imágenes
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Etiqueta: {y_train[i]}')
    plt.axis('off')
plt.show()

# Normalizar las imágenes y convertirlas a tensores
    #Un tensor generalmente se representa como una matriz multidimensional que puede tener una o más dimensiones. 
    #Las imágenes comunmente se representan como tensores 2D(escala de grises, "blanco y negro") o 3D(RGB), dependiendo de si son en escala de grises o en color.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Crear el modelo CNN simple
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)), #reducimos el tamaño espacial de la representación(imagen)
    layers.Flatten(),#aplanamos la salida osea: convertir los datos de entrada que tienen una estructura multidimensional (como tensores) en una forma unidimensional o un vector
    
    #Esta línea crea una capa de procesamiento en la red que consta de 128 "neuronas". 
    #Cada neurona toma la información que viene de las capas anteriores y realiza cálculos en ella. 
    #La palabra "relu" significa que estas neuronas utilizan una forma específica de matemáticas para tomar decisiones y aprender patrones en los datos.
    layers.Dense(128, activation='relu'),
    
    #crea una segunda capa de procesamiento con 10 neuronas. 
    # Cada una de estas neuronas ayuda a decir si una imagen se parece a uno de los 10 dígitos posibles (0, 1, 2, ..., 9). 
    # La palabra "softmax" significa que estas neuronas calcularán las probabilidades de que una imagen sea cada uno de esos dígitos.
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', #se utiliza para ajustar o "entrenar" la red neuronal, es decir, para hacer que la red aprenda de los datos.
              loss='sparse_categorical_crossentropy',#es una medida de cuán lejos están las predicciones del modelo de las respuestas correctas
              metrics=['accuracy'])#medidas adicionales que se utilizan para evaluar el rendimiento del modelo

# Crear listas para almacenar métricas
# el modelo comenzará a aprender a partir de los datos de entrenamiento y realizará varias "épocas" (5 en este caso).
# Cada época implica ver todo el conjunto de datos de entrenamiento, realizar predicciones y ajustar los pesos de la red para mejorar el rendimiento. 
# El proceso continuará durante las 5 épocas y el historial del entrenamiento se almacenará en la variable history, 
# lo que permite visualizar cómo el modelo ha progresado en términos de precisión y pérdida durante el entrenamiento.

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Visualizar las métricas y la pérdida durante el entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()
