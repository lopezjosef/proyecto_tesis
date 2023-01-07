# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:09:06 2023

@author: jflopez
"""
import pandas as pd
import numpy as np
#creo una funcion que me convierta los datos en un problema de aprendizaje supervisado
def Supervisado(nPeriodosX=0,nPeriodosY=0,df_serie_y=None,df_serie_x=None):
    for i in range(nPeriodosX+nPeriodosY-1):
        df_serie_y['X(t-'+str(i+1)+')'] = pd.Series()
        df_serie_y['X(t-'+str(i+1)+')'].iloc[i+1:] = df_serie_x.iloc[:-(i+1),0]
    
    df_serie_y = df_serie_y.dropna(subset=['X(t-'+str(nPeriodosX+nPeriodosY-1)+')'])
    lista = []
    for i in range(nPeriodosY,0,-1):
        lista.append('Y(t+'+str (i)+')')
    for i in range(nPeriodosX):
        lista.append('Y(t-'+str(i+1)+')')
    df_serie_y.columns = lista
    return df_serie_y

def CrearVariablesMeses():
    def ejes_mes(Mes):
        angulo = (Mes*(2*np.pi))/12
        eje_x = np.cos(angulo)
        eje_y = np.sin(angulo)
        
        return [Mes,eje_x,eje_y]
    
    lista = [ejes_mes(mes) for mes in range(1,13)]
    meses_feature = pd.DataFrame(lista,columns=['NumeroMes','EjeX','EjeY'])
    
    return meses_feature

def CrearDataset(datos_internos,datos_externos,nMesesAtras,nMesesAdelante):
    """
    Parameters
    ----------
    datos_internos : pd.DataFrame()
        Dataframe con datos de pedidos.
    datos_externos : pd.DataFrame()
        Dataframe con variables externas.
    nMesesAtras : int
        Meses a utilizar como variables de entrada.
    nMesesAdelante : int
        Meses a predecir.

    Returns
    -------
    Devuelve un dataframe X y uno Y.

    """
    
    datos_internos = datos_internos.groupby(['Mes'],as_index=False).agg({'PedidosKg':sum})
    datos_internos['PedidosKg'] /= 1000
    datos_internos.index = pd.PeriodIndex(datos_internos.Mes,freq='M')
    
    variables_mes = CrearVariablesMeses()

    datos_externos.index = pd.PeriodIndex(datos_externos.Mes,freq='M')
    datos_externos['NumeroMes'] = datos_externos.index.month

    datos_externos = pd.merge(datos_externos,variables_mes,on='NumeroMes',how='left')
    datos_externos = datos_externos.drop(['NumeroMes'],axis=1)
    datos_externos.index = pd.PeriodIndex(datos_externos.Mes,freq='M')

    #Transformo datos
    df_variables_externas = pd.DataFrame()
    for indice in datos_externos.columns[1:]:
        data = datos_externos[[indice]]
        df_aux = Supervisado(nMesesAtras,0,data[[indice]],data[[indice]])
        df_aux['Mes'] = df_aux.index
        df_aux['Indice'] = indice
        df_aux.index.name = 'index'
        df_variables_externas = df_variables_externas.append(df_aux)


    df_variables_externas = pd.pivot_table(df_variables_externas,index='Mes',columns='Indice')
    df_variables_externas.index.name = 'Mes'


    df_variables_internas = Supervisado(nMesesAtras,nMesesAdelante,datos_internos[['PedidosKg']],datos_internos[['PedidosKg']])

    #Creo el dataset con todas las variables de entrada
    X = pd.merge(df_variables_internas.drop(['Y(t+{n})'.format(n=i) for i in range(1,nMesesAdelante+1)],axis=1),df_variables_externas,how='left',left_index=True,right_index=True)

    #Creo el dataset con las variables de salida para la agrupacion deseada
    Y = df_variables_internas[['Y(t+{n})'.format(n=i) for i in range(1,nMesesAdelante+1)]]
    
    return X, Y