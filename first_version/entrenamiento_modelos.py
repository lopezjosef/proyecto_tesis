# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 04:57:44 2022

@author: jflopez
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16,9)

#Cargo los datos normalizados en toneladas
df = pd.read_excel('data_modelo_features_norm.xlsx')

# Para cada agrupacion, se guarda el loss y el val loss
resultados = pd.DataFrame()

agrupaciones = df.Agrupacion.drop_duplicates().to_list()

for agrupacion in agrupaciones:
    resultados_agrupacion = pd.DataFrame()
    
    data = df.loc[df['Agrupacion'] == agrupacion]
    data.index = data.Mes
    split = int((data.shape[0])*0.8)

    X = data.drop(['Y(t+2)','Y(t+1)','Mes','Agrupacion'],axis=1)
    Y = data[['Y(t+2)','Y(t+1)']]

    X_train, X_validation = X[:split].values, X[split:].values
    Y_train, Y_validation = Y[:split].values, Y[split:].values


    model = Sequential()
    model.add(Dense(32,input_dim=9,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(2,activation='linear'))
    
    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['MSE'])

    model.summary()

    historia = model.fit(x = X_train, y = Y_train,
                         validation_data = (X_validation, Y_validation),
                         epochs=500, batch_size=32,validation_batch_size=32)
    
    Y_predict = model.predict(X_validation)
    
    df_mes_1 = pd.DataFrame({'Prediccion':Y_predict[:,1],
                  'Validacion':Y_validation[:,1]},
                 index=data[split:].index)
    plot_mes_1 = df_mes_1.plot()
    plot_mes_1.figure.savefig('graficos/pred_mes_1'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)
    
    entrenamiento = pd.DataFrame({'loss':historia.history['loss'],'val_loss':historia.history['val_loss']})
    entrenamiento_plot = entrenamiento.plot()
    #entrenamiento_plot.figure.savefig('graficos/entrenamiento'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)
    
    pred = Y_predict[:,:].sum(axis=1)
    val = Y_validation[:,:].sum(axis=1)
    dif = val-pred
    dif_plot_density = pd.DataFrame(dif).plot(kind='density')
    #dif_plot_density.figure.savefig('graficos/densidad_errores'+agrupacion.replace('/','').replace(' ','')+'.png', dpi=150)

    resultados_agrupacion['Agrupacion'] = [agrupacion]
    resultados_agrupacion['val_loss'] = [historia.history['val_loss'][-1]]
    resultados_agrupacion['loss'] = [historia.history['loss'][-1]]
    resultados_agrupacion['media_errores'] = [dif.mean()]

    resultados = resultados.append(resultados_agrupacion)
    #model.save('modelos/'+'model_'+agrupacion.replace('/','').replace(' ','')+'.h5') 
    
    print('Entrenamiento completado:'+agrupacion)
    

resultados.to_excel('resultados_entrenamiento.xlsx',index=False)
    
    