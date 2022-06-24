


import pandas as pd
import numpy as np

# Escalador
from sklearn.preprocessing import MinMaxScaler

# Para especificar los índices de validación y entrenamiento en el conjunto de entrenamiento
from sklearn.model_selection import PredefinedSplit

# Imputación de valores perdidos
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.naive_bayes import CategoricalNB

# Para el gráfico de las observaciones utilizadas vs el total
import plotly.express as px



class PreprocesamientoLimpiezaDatos:



    ANCHURA_GRAFICO = 700
    ALTURA_GRAFICO = 350
    POSX_TITULO_CENTRADO = 0.5
    POSY_TITULO_CENTRADO = 0.9
    etiqueta_y = 'y'
    etiqueta_x = 'x'
    etiqueta_num_observaciones = 'Cantidad de observaciones'
    etiqueta_tipo_observaciones = 'Tipo de observaciones'
    nombre_observaciones_totales = 'Observaciones totales <br>(sin procesar) '
    nombre_observaciones_eliminadas_predecir_resultado = 'Observaciones utilizadas <br>para predecir el resultado '
    nombre_observaciones_eliminadas_clustering = 'Observaciones utilizadas <br>para el clustering '
    titulo_grafica_comparacion_observaciones = 'Observaciones utilizadas con respecto al total'
    secuencia_colores = px.colors.qualitative.Set2



    def __init__(self, atributos_categoricos):
        self.escalador = MinMaxScaler()
        self.atributos_categoricos = atributos_categoricos
        


    def escalarDatos(self, datos):
        return pd.DataFrame(self.escalador.fit_transform(datos), columns = datos.columns)


    def escalarDatosSplit(self, datos_entrenar, datos_validacion, datos_prueba):
        datos_entrenar_escalados = self.escalador.fit_transform(datos_entrenar)
        datos_validacion_escalados = self.escalador.transform(datos_validacion)
        datos_prueba_escalados = self.escalador.transform(datos_prueba)
        datos_entrenar_escalados_df = pd.DataFrame(datos_entrenar_escalados, columns = datos_entrenar.columns)
        datos_validacion_df = pd.DataFrame(datos_validacion_escalados, columns = datos_validacion.columns)
        datos_prueba_escalados_df = pd.DataFrame(datos_prueba_escalados, columns = datos_prueba.columns)
        return datos_entrenar_escalados_df, datos_validacion_df, datos_prueba_escalados_df


    def crearSplits(self, datos):
        datos_entrenamiento_validacion = datos.sample(frac = 0.8, random_state = 7)
        datos_prueba = datos.drop(datos_entrenamiento_validacion.index)
        datos_entrenamiento = datos_entrenamiento_validacion.sample(frac = 0.8, random_state = 33)
        datos_validacion = datos_entrenamiento_validacion[~datos_entrenamiento_validacion.index.isin(datos_entrenamiento.index)]
        return datos_entrenamiento.reset_index(drop = True), datos_validacion.reset_index(drop = True), datos_prueba.reset_index(drop = True)


    def separarXY(self, datos_entrenamiento, datos_validacion, datos_prueba, columna_y):
        datos_entrenamiento_predictores = datos_entrenamiento.drop(labels = columna_y, axis = 1).reset_index(drop = True)
        datos_validacion_predictores = datos_validacion.drop(labels = columna_y, axis = 1).reset_index(drop = True)
        datos_prueba_predictores = datos_prueba.drop(labels = columna_y, axis = 1).reset_index(drop = True)
        datos_entrenamiento_y = datos_entrenamiento[columna_y].reset_index(drop = True)
        datos_validacion_y = datos_validacion[columna_y].reset_index(drop = True)
        datos_prueba_y = datos_prueba[columna_y].reset_index(drop = True)
        return datos_entrenamiento_predictores, datos_validacion_predictores, datos_prueba_predictores, datos_entrenamiento_y, datos_validacion_y, datos_prueba_y


    def splitEntrenamientoValidacion(self, datos_entrenamiento_predictores, datos_validacion_predictores, datos_entrenamiento_y, datos_validacion_y):
        datos_entrenamiento_validacion_predictores = np.concatenate((datos_entrenamiento_predictores, datos_validacion_predictores))
        datos_entrenamiento_validacion_y = np.concatenate((datos_entrenamiento_y, datos_validacion_y))
        datos_entrenamiento_validacion_predictores_df = pd.DataFrame(datos_entrenamiento_validacion_predictores, columns = datos_entrenamiento_predictores.columns)
        datos_entrenamiento_validacion_y_df = pd.DataFrame(datos_entrenamiento_validacion_y, columns = [datos_entrenamiento_y.name])
        indices_entrenamiento = np.ones(datos_entrenamiento_predictores.shape[0]) * -1
        indices_validacion = np.ones(datos_validacion_predictores.shape[0])
        indices_datos_entrenamiento_validacion = PredefinedSplit(np.concatenate((indices_entrenamiento, indices_validacion)))
        return datos_entrenamiento_validacion_predictores_df, datos_entrenamiento_validacion_y_df, indices_datos_entrenamiento_validacion


    def dividirDatosEnEntrenamientoTest(self, datos, columna_y):
        
        # Dividir el conjunto de datos en los conjuntos de entrenamiento, validación y test
        datos_entrenamiento, datos_validacion, datos_prueba = self.crearSplits(datos)

        # Lidiar con los valores perdidos
        if datos_entrenamiento.isna().sum().sum() > 0:
            datos_entrenamiento = self.imputacionSegunTipoVariable(datos_entrenamiento)
   
        num_datos_prueba = datos_prueba.shape[0]
        if datos_prueba.isna().sum().sum() > 0:
            datos_prueba = self.eliminarValoresPerdidos(datos_prueba)
            
        num_datos_validacion = datos_validacion.shape[0]
        if datos_validacion.isna().sum().sum() > 0:
            datos_validacion = self.eliminarValoresPerdidos(datos_validacion)
            
        num_datos_eliminados = (num_datos_prueba - datos_prueba.shape[0]) + (num_datos_validacion - datos_validacion.shape[0])

        # Cada split de train/test se subdivide en X e y
        datos_entrenamiento_predictores, datos_validacion_predictores, datos_prueba_predictores, datos_entrenamiento_y, datos_validacion_y, datos_prueba_y = self.separarXY(datos_entrenamiento, datos_validacion, datos_prueba, columna_y)

        datos_entrenamiento_predictores_sin_escalar = datos_entrenamiento_predictores.copy()
        datos_entrenamiento_y_sin_escalar = datos_entrenamiento_y.copy()

        # Escalar entrenamiento y aplicar su transformación a la validación al test
        datos_entrenamiento_predictores, datos_validacion_predictores, datos_prueba_predictores = self.escalarDatosSplit(datos_entrenamiento_predictores, datos_validacion_predictores, datos_prueba_predictores)

        # Datos de entranamiento y validación juntos, con los índices de las observaciones de entrenamiento y las de validación
        datos_entrenamiento_validacion_predictores, datos_entrenamiento_validacion_y, indices_datos_entrenamiento_validacion = self.splitEntrenamientoValidacion(datos_entrenamiento_predictores, datos_validacion_predictores, datos_entrenamiento_y, datos_validacion_y)

        return datos_entrenamiento_predictores, datos_validacion_predictores, datos_prueba_predictores, datos_entrenamiento_y, datos_validacion_y, datos_prueba_y, datos_entrenamiento_predictores_sin_escalar, datos_entrenamiento_y_sin_escalar, datos_entrenamiento_validacion_predictores, datos_entrenamiento_validacion_y, indices_datos_entrenamiento_validacion, num_datos_eliminados


    def imputacionValoresPerdidos(self, datos, son_datos_categoricos):

        if son_datos_categoricos:
            modelo = CategoricalNB()

        else:
            modelo = linear_model.BayesianRidge()

        imputador_valores = IterativeImputer(estimator = modelo, imputation_order = 'ascending', max_iter = 20, random_state = 33)
        datos_imputados = imputador_valores.fit_transform(datos)
        datos_imputados_df = pd.DataFrame(datos_imputados, columns = datos.columns)

        return datos_imputados_df


    def separarDatosSegunTipoVariable(self, datos):
        columnas_categoricas = []
        columnas_numericas = []

        for columna in datos.columns:
            if columna in self.atributos_categoricos:
                columnas_categoricas.append(columna)
            else:
                columnas_numericas.append(columna)

        datos_numericos = datos[columnas_numericas].reset_index(drop = True)
        datos_categoricos = datos[columnas_categoricas].reset_index(drop = True)
        return datos_numericos, datos_categoricos, columnas_categoricas


    def imputacionSegunTipoVariable(self, datos):
        columnas_ordenadas = datos.columns
        datos_numericos, datos_categoricos, columnas_categoricas = self.separarDatosSegunTipoVariable(datos)
        datos_numericos = self.imputacionValoresPerdidos(datos_numericos, False)
        datos_categoricos = self.imputacionValoresPerdidos(datos_categoricos, True)
        datos_sin_valores_perdidos = pd.concat([datos_categoricos, datos_numericos], axis = 1)
        datos_sin_valores_perdidos = datos_sin_valores_perdidos[columnas_ordenadas]
        return datos_sin_valores_perdidos


    def eliminarValoresPerdidos(self, datos):
        return datos.dropna(axis = 0).reset_index(drop = True)


    def figuraComparacionObservaciones(self, num_total_observaciones, num_observaciones_eliminadas_predecir_victoria, num_observaciones_clustering):
        
        nombres_comparacion_observaciones = [self.nombre_observaciones_totales, self.nombre_observaciones_eliminadas_clustering, self.nombre_observaciones_eliminadas_predecir_resultado]
        valores_comparacion_observaciones = [num_total_observaciones, num_observaciones_clustering, num_total_observaciones - num_observaciones_eliminadas_predecir_victoria]

        figura = px.bar(y = nombres_comparacion_observaciones, x = valores_comparacion_observaciones,
                        color_discrete_sequence = self.secuencia_colores, color = nombres_comparacion_observaciones,
                        title = self.titulo_grafica_comparacion_observaciones,
                        labels = {
                                    self.etiqueta_y: self.etiqueta_tipo_observaciones,
                                    self.etiqueta_x: self.etiqueta_num_observaciones
                                })
        figura.update_layout(showlegend = False, autosize = False, width = self.ANCHURA_GRAFICO, height = self.ALTURA_GRAFICO, 
                            title_x = self.POSX_TITULO_CENTRADO, title_y = self.POSY_TITULO_CENTRADO)
        return figura


    # Separar la información de la partida en dos (con la información específica de cada equipo)
    def separarPartidaPorEquipos(self, datos, columnas, nombre_atributo_lado_mapa):
        datos_generales_partida = datos.iloc[:,:2]
        datos_equipo_t1 = datos.iloc[:,2:31]
        datos_equipo_t2 = datos.iloc[:,31:]

        datos_equipo_t1 = pd.concat([datos_generales_partida, datos_equipo_t1], axis = 1)
        datos_equipo_t2 = pd.concat([datos_generales_partida, datos_equipo_t2], axis = 1)
        datos_equipo_t1.columns = columnas
        datos_equipo_t2.columns = columnas

        # 'mapside': lado del mapa en el que juega el equipo (equipo azul = 0, equipo rojo = 1)
        datos_equipo_t1[nombre_atributo_lado_mapa] = [0.0 for i in range(len(datos_equipo_t1))]
        datos_equipo_t2[nombre_atributo_lado_mapa] = [1.0 for i in range(len(datos_equipo_t2))]

        datos = pd.concat([datos_equipo_t1, datos_equipo_t2], axis = 0)

        datos.reset_index(drop = True, inplace = True)

        return datos