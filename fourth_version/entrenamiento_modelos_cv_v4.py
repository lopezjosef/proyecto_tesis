# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:11:12 2023

@author: jflopez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.regularizers import L1L2,L1, L2
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
agrupacion = agrupaciones[0]

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
    X.index = pd.PeriodIndex(X.index,freq='M') 
    
    ids_validation = X.index.year.isin([2008,2014,2022])
    ids_train = ~X.index.year.isin([2008,2014,2022])

    
    #Creo el dataset con las variables de salida para la agrupacion deseada
    Y = df.loc[df['Agrupacion'] == agrupacion,['Y(t+2)','Y(t+1)']]
    Y = pd.DataFrame(Y.sum(axis=1))
    Y.index=X.index

    #Divido el 90% de los datos para entrenar con cross validation y dejo el 10% para testear luego 
    X_train, X_validation = X[ids_train].values, X[ids_validation].values
    Y_train, Y_validation = Y[ids_train].values, Y[ids_validation].values
    
    
    # Iteramos sobre los pliegues
    loss_accepted = 10000
    for train_index, test_index in kfold.split(X_train):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
    
        model = Sequential()
        model.add(Dense(128,input_dim=234,activation='relu',kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),bias_regularizer=L2(1e-4),activity_regularizer=L2(1e-5)))
        model.add(Dense(64,activation='linear',kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),bias_regularizer=L2(1e-4),activity_regularizer=L2(1e-5)))
        model.add(Dense(32,activation='relu',kernel_regularizer=L1L2(l1=1e-3, l2=1e-5),bias_regularizer=L2(1e-3),activity_regularizer=L2(1e-5)))
        model.add(Dense(16,activation='linear',kernel_regularizer=L1L2(l1=1e-3, l2=1e-4),bias_regularizer=L2(1e-3),activity_regularizer=L2(1e-3)))
        model.add(Dense(1,activation='linear'))
    
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MSE'])
    
        historia = model.fit(x = X_train_cv, y = Y_train_cv,epochs=500, batch_size=32)
    
        # Calculamos la métrica de evaluación val y almacenamos el resultado
        val_loss = model.evaluate(X_test_cv, Y_test_cv, verbose=0)
        resultados_cv_val_loss.append(val_loss)
        
        loss = historia.history['loss'][-1]
        resultados_cv_loss.append(loss)
        
        if val_loss[0] < loss_accepted:
            mejor_modelo = model
            loss_accepted = val_loss[0]
    
    mae_val_loss_cv = np.median(np.array(resultados_cv_val_loss)[:,0])
    std_val_loss_cv = np.array(resultados_cv_val_loss)[:,0].std()  
    mae_loss_cv = np.array(resultados_cv_loss).mean()
    
    Y_predict = mejor_modelo.predict(X_validation)
    
    df_prediccion = pd.DataFrame({'Prediccion':Y_predict[:,0],
                   'Validacion':Y_validation[:,0]},
                 index=X[ids_validation].index)
    df_prediccion.loc[df_prediccion['Prediccion'] < 0,'Prediccion'] = 0

    for año in df_prediccion.index.year.drop_duplicates():
        plot_prediccion = df_prediccion.loc[df_prediccion.index.year == año].plot(title = 'Prediccion para los datos de test: '+agrupacion)
        plot_prediccion.figure.savefig('graficos4/plot_prediccion_'+agrupacion.replace('/','').replace(' ','')+'_'+str(año)+'.png', dpi=150)
    
    entrenamiento = pd.DataFrame({'loss':historia.history['loss']})
    entrenamiento_plot = entrenamiento.plot()
    entrenamiento_plot.figure.savefig('graficos4/entrenamiento_'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)
    
   
    pred = Y_predict[:,:].sum(axis=1)
    val = Y_validation[:,:].sum(axis=1)
    dif = val-pred
    dif_plot_density = pd.DataFrame(dif).plot(kind='density')
    dif_plot_density.figure.savefig('graficos4/densidad_errores_'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)
    
    resultados_agrupacion['Agrupacion'] = [agrupacion]
    resultados_agrupacion['mae_val_loss'] = [mae_val_loss_cv]
    resultados_agrupacion['mae_loss'] = [mae_loss_cv]
    resultados_agrupacion['std_mae_val_loss'] = [std_val_loss_cv]
    resultados_agrupacion['media_errores_nominales'] = [dif.mean()]
    
    resultados = resultados.append(resultados_agrupacion)
    mejor_modelo.save('modelos4/'+'model_'+agrupacion.replace('/','').replace(' ','')+'.h5') 
    
    print('Entrenamiento completado:'+agrupacion)
    

resultados.to_excel('resultados_entrenamiento_4.xlsx',index=False)
    
    