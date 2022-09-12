import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Limpiar los datos
car_data = pd.read_csv("car_data.csv", header = 0)
car_data.drop(['User ID'], axis = 1, inplace = True)
car_data['Gender'].replace(['Male', 'Female'],[0, 1], inplace=True)

# Obtener la Y de la base de datos
Y = car_data['Purchased']
X = car_data.drop(['Purchased'], axis = 1)

# Normalizar los datos
scaler = StandardScaler()
X_trans = scaler.fit_transform(X)

# Dividir en train y test
x_train, x_test, y_train, y_test = train_test_split(X_trans, Y, test_size = 0.2, random_state = 55)


# MODELO INICIAL

# Crear el modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 2, activation = "relu", input_dim = x_train.shape[1]))
model.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

# Compilar el modelo
model.compile(
    optimizer = "sgd",
    loss = "binary_crossentropy",
    metrics = [tf.keras.metrics.BinaryAccuracy()]
)

# Entrenar el modelo
print('MODELO 1')
print('Inicio del entrenamiento')
historia = model.fit(x_train, y_train, epochs = 100, verbose = True, validation_split = 0.2)
print("Modelo entrenado")

# Graficar la pérdida
plt.subplot(1, 3, 1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(historia.history["loss"], label='Train Loss')
plt.plot(historia.history["val_loss"], label='Validation Loss')
plt.legend()

# Graficar el accuracy
plt.subplot(1, 3, 2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(historia.history["binary_accuracy"], color = "green", label='Train Accuracy')
plt.plot(historia.history["val_binary_accuracy"], label='Validation Accuracy')

plt.legend()
plt.show()

# Predicciones
y_hat=model.predict(x_test)
y_hat=[0 if val<0.5 else 1 for val in y_hat]
print("Roc AUC Modelo 1: ", roc_auc_score(y_test,y_hat))
print("Binary Accuracy Modelo 1: ", accuracy_score(y_test,y_hat))


# MODELO FINAL

# Crear el modelo
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(units = 8, activation = "relu", input_dim = x_train.shape[1]))
model2.add(tf.keras.layers.Dense(units = 4, activation = "relu"))
model2.add(tf.keras.layers.Dense(units = 2, activation = "relu"))
model2.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

# Compilar el modelo
model2.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
)

# Entrenar el modelo
print('MODELO 2')
print('Inicio del entrenamiento')
historia = model2.fit(X_trans, Y, epochs = 400, validation_split = 0.2, verbose = 2)
print("Modelo entrenado")

# Graficar la pérdida
plt.subplot(1, 3, 1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(historia.history["loss"], label='Train Loss')
plt.plot(historia.history["val_loss"], label='Validation Loss')

# Graficar el accuracy
plt.subplot(1, 3, 2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(historia.history["binary_accuracy"], color = "green", label='Train Accuracy')
plt.plot(historia.history["val_binary_accuracy"], label='Validation Accuracy')

plt.legend()
plt.show()

# Predicciones
y_hat=model2.predict(x_test)
y_hat=[0 if val<0.5 else 1 for val in y_hat]
print("Roc AUC Modelo 2: ", roc_auc_score(y_test,y_hat))
print("Test Accuracy Modelo 2: ", accuracy_score(y_test,y_hat))
