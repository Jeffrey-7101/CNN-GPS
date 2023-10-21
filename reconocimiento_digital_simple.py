import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt #visualizar imagenes y el resultado de las predicciones

# Cargar el conjunto de datos MNIST,
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

x_train = x_train / 255.0
x_test = x_test / 255.0

# Crear el modelo CNN simple
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)), 
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Crear listas para almacenar métricas
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
