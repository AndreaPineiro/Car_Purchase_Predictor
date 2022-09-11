# 1. Aprendizaje-Maquina-Sin-Framework
Implementación de una técnica de aprendizaje máquina sin el uso de un framework.

Para esta actividad se realizó una red neuronal de 2 capas (1 sola capa escondida). Para esto fue necesaria la implementación de funciones de activación vistas en clase como Rely y Sigmoid; así como fue necesaria la implementación del gradient descent en el modelo.

La información usada para esta actividad fue encontrada en Kaggle (https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset?resource=download). En donde se plantea un problema en el que hay que saber dependiendo del género, la edad y el salario de las personas si compraran o no un carro.

Se uso el 20% de los datos para test y se obtuvo un muy buen resultado de 79% de exactitud.

# 2. Aprendizaje-Maquina_Tensorflow
Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.

Para esta actividad se uso tensorflow como framework de aprendizaje de máquina. Se creó primero un modelo inicial, que posteriormente fue actualizado y mejorado para obtener mejores resultados.
La API usada es *tf.keras* sirve para construir y entrenar modelos de aprendizaje profundo. 

## Modelo 1
El modelo inicial es una red neuronal que contiene 1 capa oculta con 2 neuronas. 
- Para la primera capa se usa la función de activación *relu*
- Para la segunda capa se usa la función de activación *sigmoid*

El modelo usa el algoritmo *sgd* de optimización y las métricas son *BinaryCrossentropy* y *BinaryAccuracy*
Este modelo tiene aproximadamente un 0.84 de Accuracy en las pruebas realizadas.

## Modelo 2
Para mejorar el modelo se agregó 1 capa oculta más con 3 neuronas
- Para la primera capa se usa la función de activación *relu*
- Para la segunda capa se usa la función de activación *relu*
- Para la tercera capa se usa la función de activación *sigmoid*

El modelo usa el algoritmo *adam* de optimización y las métricas son *BinaryCrossentropy* y *BinaryAccuracy*, igual que en el modelo anterior para una mejor comparación.
Se uso el método de *fit* de *tf.keras* con el validation_split de los datos para test y de esta manera poder también calcular el Accuracy y el Loss en test para cada epoch.

Este modelo tiene aproximadamente un 0.9 de Accuracy en las pruebas realizadas. 

El Accuracy de ambos modelos varía, pero el modelo 2 siempre tiene un mejor accuracy, por lo que se mejoró el modelo inicial.
