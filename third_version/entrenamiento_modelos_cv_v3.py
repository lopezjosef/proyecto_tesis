# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 00:04:38 2022

@author: jflopez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from sklearn.model_selection import KFold

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16,9)

#Cargo los datos normalizados en toneladas
df = pd.read_excel('data_modelo_features_norm.xlsx')
df_externas = pd.read_excel('variables_externas_mes.xlsx')
df_externas.index = df_externas.Mes_
df_externas = df_externas.drop('Mes_',axis=1)

# Para cada agrupacion, se guarda el loss y el val loss
resultados = pd.DataFrame()

agrupaciones = df.Agrupacion.drop_duplicates().to_list()


for agrupacion in agrupaciones:
    resultados_agrupacion = pd.DataFrame()
    # Definimos el número de pliegues para el cross validation
    kfold = KFold(n_splits=5, shuffle=True)
    
    # Inicializamos una lista para almacenar los resultados del cross validation
    resultados_cv_val_loss = []
    resultados_cv_loss = []
    
    #Creo el dataset con todas las variables de entrada
    X = pd.pivot_table(df.drop(['Y(t+2)','Y(t+1)'],axis=1),index='Mes',columns='Agrupacion',aggfunc='sum')
    X = pd.merge(X,df_externas,how='left',left_index=True,right_index=True)

    #Creo el dataset con las variables de salida para la agrupacion deseada
    Y = df.loc[df['Agrupacion'] == agrupacion,['Y(t+2)','Y(t+1)']]
    
    #Divido el 90% de los datos para entrenar con cross validation y dejo el 10% para testear luego 
    split = int((X.shape[0])*0.85)
    X_train, X_validation = X[:split].values, X[split:].values
    Y_train, Y_validation = Y[:split].values, Y[split:].values
    
    
    # Iteramos sobre los pliegues
    loss_accepted = 10000
    for train_index, test_index in kfold.split(X_train):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
    
        model = Sequential()
        model.add(Dense(128,input_dim=234,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(32,activation='linear'))
        model.add(Dense(16,activation='relu'))
        model.add(Dense(2,activation='linear'))
    
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MSE'])
    
        historia = model.fit(x = X_train_cv, y = Y_train_cv,epochs=500, batch_size=32)
    
        # Calculamos la métrica de evaluación val y almacenamos el resultado
        val_loss = model.evaluate(X_test_cv, Y_test_cv, verbose=0)
        resultados_cv_val_loss.append(val_loss)
        
        loss = historia.history['loss'][-1]
        resultados_cv_loss.append(loss)
        
        if val_loss[0] < loss_accepted:
            mejor_modelo = model
    
    mae_val_loss_cv = np.median(np.array(resultados_cv_val_loss)[:,0])
    std_val_loss_cv = np.array(resultados_cv_val_loss)[:,0].std()  
    mae_loss_cv = np.array(resultados_cv_loss).mean()
    
    Y_predict = mejor_modelo.predict(X_validation)
    
    df_prediccion = pd.DataFrame({'Prediccion':Y_predict[:,1],
                   'Validacion':Y_validation[:,1]},
                 index=X[split:].index)
    df_prediccion.loc[df_prediccion['Prediccion'] < 0,'Prediccion'] = 0
    plot_prediccion = df_prediccion.plot(title = 'Prediccion para los datos de test: '+agrupacion)
    plot_prediccion.figure.savefig('graficos3/plot_prediccion_'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)
    
    entrenamiento = pd.DataFrame({'loss':historia.history['loss']})
    entrenamiento_plot = entrenamiento.plot()
    entrenamiento_plot.figure.savefig('graficos3/entrenamiento_'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)
    
   
    pred = Y_predict[:,:].sum(axis=1)
    val = Y_validation[:,:].sum(axis=1)
    dif = val-pred
    dif_plot_density = pd.DataFrame(dif).plot(kind='density')
    dif_plot_density.figure.savefig('graficos3/densidad_errores_'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)
    
    resultados_agrupacion['Agrupacion'] = [agrupacion]
    resultados_agrupacion['mae_val_loss'] = [mae_val_loss_cv]
    resultados_agrupacion['mae_loss'] = [mae_loss_cv]
    resultados_agrupacion['std_mae_val_loss'] = [std_val_loss_cv]
    resultados_agrupacion['media_errores_nominales'] = [dif.mean()]
    
    resultados = resultados.append(resultados_agrupacion)
    mejor_modelo.save('modelos3/'+'model_'+agrupacion.replace('/','').replace(' ','')+'.h5') 
    
    print('Entrenamiento completado:'+agrupacion)
    

resultados.to_excel('resultados_entrenamiento_3.xlsx',index=False)
    
    