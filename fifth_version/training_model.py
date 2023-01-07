# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:12:55 2023

@author: jflopez
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.regularizers import L1L2,L1, L2
from sklearn.model_selection import KFold
import funciones_auxiliares
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16,9)

datos_internos = pd.read_excel('data_modelo.xlsx')
datos_externos = pd.read_excel('datos_externos.xlsx')

nMesesAtras = 9
nMesesAdelante = 1

X, Y = funciones_auxiliares.CrearDataset(datos_internos,datos_externos,nMesesAtras,nMesesAdelante)

#%%
# Divido en datos de modelado y datos de prediccion
split = int((X.shape[0])*0.80)
X_modelado, X_prediccion = X[:split].values, X[split:].values
Y_modelado, Y_prediccion = Y[:split].values, Y[split:].values

# Defino el número de pliegues para el cross validation
kfold = KFold(n_splits=5, shuffle=True)

# Inicializo una lista para almacenar los resultados del cross validation
resultados_cv_test_loss = []
resultados_cv_train_loss = []

for train_index, test_index in kfold.split(X_modelado):
    X_train_cv, X_test_cv = X_modelado[train_index], X_modelado[test_index]
    Y_train_cv, Y_test_cv = Y_modelado[train_index], Y_modelado[test_index]

    model = Sequential()
    model.add(Dense(128,input_dim=X_modelado.shape[1],activation='relu'))
    model.add(Dense(64, activation='relu',
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-3),
                        activity_regularizer=L2(1e-3) 
                    ) 
              )
    
    model.add(Dense(32,activation='relu',
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-3),
                        activity_regularizer=L2(1e-3) 
                    ) 
              )
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mean_absolute_error'])

    historia = model.fit(x = X_train_cv, y = Y_train_cv,epochs=600, batch_size=24)

    # Calculamos la métrica de evaluación val y almacenamos el resultado
    val_loss = model.evaluate(X_test_cv, Y_test_cv, verbose=0)
    resultados_cv_test_loss.append(val_loss)
    
    loss = historia.history['loss'][-1]
    resultados_cv_train_loss.append(loss)

mse_test_loss_cv = np.mean(np.array(resultados_cv_test_loss)[:,0])
mse_train_loss_cv = np.array(resultados_cv_train_loss).mean()
mse_prediccion_loss = model.evaluate(X_prediccion, Y_prediccion, verbose=0)
print('erorr mse test cv:',mse_test_loss_cv)
print('error mse train cv:',mse_train_loss_cv)
print('error mse prediccion:',mse_prediccion_loss[0])
    

Y_predict = model.predict(X_prediccion)
df_prediccion = pd.DataFrame({'Prediccion':Y_predict[:,0],
                              'Validacion':Y_prediccion[:,0]},
                             index=X[split:].index)
plot_prediccion = df_prediccion.plot(title = 'Prediccion para el dataset datos_prediccion',ylabel='Toneladas',ylim=(0,2000))


Y_predict_all = model.predict(X.values)
df_prediccion_all = pd.DataFrame({'Prediccion':Y_predict_all[:,0],
                                  'Validacion':Y.values[:,0]},
                                 index=X.index)

plot_prediccion_all = df_prediccion_all.plot(title = 'Prediccion para el dataset datos_modelado y datos_prediccion',ylabel='Toneladas',ylim=(0,2300))
    













