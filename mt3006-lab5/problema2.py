# ============================================================================
# MT3006 - LABORATORIO 5, PROBLEMA 2
# ----------------------------------------------------------------------------
# En este problema usted debe emplear tensorflow para construir y entrenar una
# red neuronal para identificar un modelo no lineal para el sistema de Lorenz 
# con parámetros tal que exhiba caos. 
# ============================================================================
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from matplotlib import rcParams
from scipy import integrate

# Se ajusta el tamaño de letra y de figura
rcParams.update({'font.size': 18})
pyplot.rcParams['figure.figsize'] = [12, 12]

# Se efectúa la simulación del sistema Lorenz
dt = 0.01                   # tiempo de muestreo
T = 8                       # tiempo de simulación
t = np.arange(0, T+dt, dt)  # vector de tiempos
# Parámetros del sistema de Lorenz tal que presente caos
beta = 8/3          
sigma = 10
rho = 28

# Definición de la dinámica del sistema de Lorenz
def lorenzDynamics(x_y_z, t0, sigma = sigma, beta = beta, rho = rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Se preparan los arrays para almacenar la data del sistema Lorenz a generarse
nnInput = np.zeros((100*(len(t)-1), 3))
nnOutput = np.zeros_like(nnInput)

# Se generan 100 condiciones iniciales aleatoriamente para generar 
# trayectorias mediante la solución numérica de la dinámica del sistema

# Se generan 100 condiciones iniciales aleatorias y luego se emplean para 
# generar las trayectorias del sistema Lorenz, integrando numéricamente la 
# dinámica del sistema
np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))
x_t = np.asarray([integrate.odeint(lorenzDynamics, x0_j, t) for x0_j in x0])

# Se grafican las trayectorias generadas y se almacenan los pares 
# entrada(x[k])-salida(x[k+1]) para el entrenamiento posterior de la red 
# neuronal 
fig, ax = pyplot.subplots(1, 1, subplot_kw = {'projection': '3d'})

for j in range(100):
    nnInput[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
    nnOutput[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
    x, y, z = x_t[j,:,:].T
    ax.plot(x, y, z,linewidth=1)
    ax.scatter(x0[j,0], x0[j,1], x0[j,2], color='r')
             
ax.view_init(18, -113)
pyplot.show()

# COMPLETAR: 
# 1. Definición, entrenamiento y evaluación del modelo.
# 2. Comparación de la simulación obtenida mediante integración numérica con 
#    la obtenida por la red neuronal.


#-----------------------------------------------------------------------------
#                       Entrenamiento
#-----------------------------------------------------------------------------

Xt=nnInput[:56000,:]
Yt=nnInput[:56000,:]

X=nnInput[56000:80000,:]
Y=nnInput[56000:80000,:]

model=tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation='sigmoid'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax'),
    tf.keras.layers.Dense(10,activation='tanh'),
    tf.keras.layers.Dense(10,activation='tanh'),
    tf.keras.layers.Dense(10,activation='tanh'),
    
    tf.keras.layers.Dense(3,activation='elu'),
    tf.keras.layers.Dense(3,activation='tanh'),
    tf.keras.layers.Dense(3,activation='tanh')
    
    ])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.009,momentum=0.7),
              loss='mean_absolute_error',
              metrics=[tf.keras.metrics.MeanSquaredError(),'mae','mse','accuracy'])

#-----------------------------------------------------------------------------
#                       Evaluación
#-----------------------------------------------------------------------------

history = model.fit(Xt,Yt, validation_data=(X,Y),epochs=100)

_, train_acc, *is_anything_else_being_returned = model.evaluate(Xt,Yt,verbose=0)
_, val_acc, *is_anything_else_being_returned = model.evaluate(X,Y,verbose=0)
print('Train: %.3f, Test %.3f'%(train_acc,val_acc))


#Grafica de la evolución de la pérdida durante el entrenamiento y la validación
pyplot.subplot(1,2,1)
pyplot.title('Loss')
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Val'], loc='upper right')

#Se grafica la evolución de la exactitud durante el entrenamiento y la validación
pyplot.subplot(1,2,2)
pyplot.plot(history.history['accuracy'],label='train')
pyplot.plot(history.history['val_accuracy'],label='test')
pyplot.title('Model accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Val'], loc='upper right')
pyplot.show()

#Graficar el Mean Squared Error
pyplot.subplot(1,2,2)
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['mse'],label='train')
pyplot.plot(history.history['val_mse'],label='test')
pyplot.ylabel('Mean Squared Error')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Val'], loc='upper right')

#Graficar el MAE
pyplot.subplot(1,2,2)
pyplot.title('Maximum Squared Error')
pyplot.plot(history.history['mae'],label='train')
pyplot.plot(history.history['val_mae'],label='test')
pyplot.ylabel('Maximum Squared Error')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Val'], loc='upper right')
pyplot.show()

