# ============================================================================
# MT3006 - LABORATORIO 5, PROBLEMA 1
# ----------------------------------------------------------------------------
# En este problema usted debe emplear tensorflow para construir y entrenar una
# red neuronal convolucional simple, para encontrar un modelo que permita 
# clasificar imágenes de las letras A, B y C. 
# ============================================================================
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from matplotlib import rcParams
from scipy import io

# Se ajusta el tamaño de letra y de figura
rcParams.update({'font.size': 18})
pyplot.rcParams['figure.figsize'] = [12, 12]

# Se carga la data de entrenamiento y validación desde los archivos .mat
lettersTrain = io.loadmat('lettersTrainSet.mat')
lettersTest = io.loadmat('lettersTestSet.mat')

# Se extraen las observaciones de entrenamiento y validación. La data importada
# presenta las dimensiones (alto, ancho, canales, batch). En este caso se tiene
# sólo un canal dado que las imágenes son en escala de grises.
XTrain = lettersTrain['XTrain']
XTest = lettersTest['XTest']
# Se extraen las labels de entrenamiento y validación, estas están dadas en 
# forma de un array de chars indicando la letra a la que corresponden: 'A', 
# 'B' y 'C'.
TTrain = lettersTrain['TTrain_cell']
TTest = lettersTest['TTest_cell']

# Se obtiene un vector con 20 índices aleatorios entre 0 y 1500-1 para obtener
# los ejemplos de imagen a visualizar. 
perm = np.random.permutation(1500)[:20]

# Se re-arregla la data de entrada para que presente las dimensiones (batch, 
# alto, ancho, canales) ya que es la forma en la que Keras la espera por 
# defecto
XTrain = np.transpose(XTrain, axes = [3,0,1,2])
XTest = np.transpose(XTest, axes = [3,0,1,2])

# Se grafican 20 ejemplos de imagen seleccionados aleatoriamente
fig,axs = pyplot.subplots(4,5)
axs = axs.reshape(-1)

for j in range(len(axs)):
    axs[j].imshow(np.squeeze(XTrain[perm[j],:,:,:]),cmap='gray')
    axs[j].axis('off')

# Se extraen las categorías como los valores únicos (diferentes) del array 
# original de labels
classes = np.unique(TTrain)
# Se crean arrays de ceros con las mismas dimensiones de los arrays originales
# de labels
YTrainLabel = np.zeros_like(TTrain)
YTestLabel = np.zeros_like(TTest)

# Se convierte la categoría desde una letra 'A', 'B', 'C' a un número 0, 1 o 2
# respectivamente
for nc in range(len(classes)):
    YTrainLabel[TTrain == classes[nc]] = nc
    YTestLabel[TTest == classes[nc]] = nc

# Se elimina la dimensión "adicional" de los vectores para poder hacer un 
# one-hot encoding con la misma en Keras
YTrainLabel = YTrainLabel.reshape(-1)
YTestLabel = YTestLabel.reshape(-1)
    
# Se efectúa un one-hot encoding para las labels
YTrain = tf.keras.utils.to_categorical(YTrainLabel)
YTest = tf.keras.utils.to_categorical(YTestLabel)

# COMPLETAR: definición, entrenamiento y evaluación del modelo.
# NOTA: durante la predicción puede emplear la función argmax de numpy para
# deshacer el one-hot encoding


#-------------------------------------------------------------------------------
#       Definición del modelo
#-------------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(10,10),
        activation='relu',
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(4, 4),
        strides=2
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        3,
        activation='tanh',
    ),
    tf.keras.layers.Dense(
        3,
        activation='softmax',
    ),
    tf.keras.layers.Dropout(
    rate=0.1
    )

])


#-------------------------------------------------------------------------------
#       Entrenamiento
#-------------------------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0104,momentum=0.000001),
              loss='categorical_crossentropy',
              metrics=['accuracy','Precision'])
#BEST PARAMETERS
# Best result -> learning = 0.05 momentum=0.000001
# leraning = 0.0104 momentum = 0.000001
# learning_rate=0.01,momentum=0.0000001

hist = model.fit(XTrain,YTrain,batch_size=32,epochs=100,
                 validation_data=(XTest,YTest))

#-------------------------------------------------------------------------------
#       Evaluación del modelo
#-------------------------------------------------------------------------------
model.evaluate(XTest, YTest)[1]

#-------------------------------------------------------------------------------
#       Grafica de la evaluación
#-------------------------------------------------------------------------------

#Evaluamos el modelo
print("\nPrueba del modelo\n")
model.evaluate(XTest, YTest)[1]

#Graficamos los datos
pyplot.subplot(1,2,1)
pyplot.plot(hist.history['loss'])
pyplot.plot(hist.history['val_loss'])
pyplot.title('Model loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Val'], loc='upper right')

pyplot.subplot(1,2,2)
pyplot.plot(hist.history['accuracy'])
pyplot.plot(hist.history['val_accuracy'])
pyplot.title('Model accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Val'], loc='upper right')
pyplot.show()
