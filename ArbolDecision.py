


import pandas as pd
import numpy as np

# Gráficos
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn import tree

# Validación cruzada
from sklearn.model_selection import cross_val_score

from SeleccionadorAtributos import SeleccionadorAtributos
from ImportanciaPermutacion import ImportanciaPermutacion



class ArbolDecision:



    ESTADO_ALEATORIO = 33
    POSX_TITULO_GRAFICA = 0.5
    POSY_TITULO_GRAFICA = 0.9
    ANCHURA_GRAFICO = 700
    ALTURA_GRAFICO = 500
    TAM_FIGURA_ARBOL = (16,6)
    DPI_FIGURA_ARBOL = 300
    PROFUNDIDAD_FIGURA_ARBOL = 2
    TAM_TEXTO_FIGURA_ARBOL = 12
    sequencia_colores = px.colors.qualitative.Set2
    posicion_arriba_derecha = 'top right'
    modo_linea_puntos = 'dash'
    nombre_modelo_clasificador = 'clasificador'
    nombre_modelo_regresor = 'regresor'
    nombre_modelo_arbol_decision_intermedio = ' con árbol de decisión para predecir '
    nombre_generico_modelo_arbol_decision_seis_atributos = 'Árbol de decisión 6 atributos'
    titulo_grafica_importancia_atributos = 'Importancia de los atributos '
    titulo_grafica_mejor_alfa = ' vs alfa en el entrenamiento y en la validación '
    titulo_grafica_grafo_arbol_inicio = 'Principales nodos del arbol podado con alfa '
    etiqueta_alfa = 'Alfa'
    etiqueta_test = 'Validación'
    etiqueta_entrenamiento = 'Entrenamiento'
    etiqueta_mejor_alfa = '  Alfa seleccionado: '
    nombre_precision = 'precisión'
    nombre_r2 = 'R^2'
    nombres_clases_figura_arbol = ['Perder', 'Ganar']
    nombre_importancia = 'Importancia'
    color_violeta = 'orchid'
    color_azul = 'cornflowerblue'
    color_rojo = 'lightcoral'



    def __init__(self, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, predictores_entrenar_sin_escalar, y_entrenar_sin_escalar, columna_y, es_clasificacion):
        self.predictores_entrenar = predictores_entrenar
        self.y_entrenar = y_entrenar.values.ravel()
        self.predictores_prueba = predictores_prueba
        self.y_prueba = y_prueba.values.ravel()
        self.es_clasificacion = es_clasificacion
        self.predictores_entrenamiento_validacion = predictores_entrenamiento_validacion
        self.y_entrenamiento_validacion = y_entrenamiento_validacion.values.ravel()
        self.indices_entrenamiento_validacion = indices_entrenamiento_validacion
        if es_clasificacion:
            self.modelo = DecisionTreeClassifier(random_state = self.ESTADO_ALEATORIO)
            self.nombre_puntuacion = self.nombre_precision
            self.nombre_arbol = '<br>(' + self.nombre_modelo_clasificador + self.nombre_modelo_arbol_decision_intermedio + columna_y + ')'
        else:
            self.modelo = DecisionTreeRegressor(random_state = self.ESTADO_ALEATORIO)
            self.nombre_puntuacion = self.nombre_r2
            self.nombre_arbol = '<br>(' + self.nombre_modelo_regresor + self.nombre_modelo_arbol_decision_intermedio + columna_y + ')'
        self.predictores_entrenar_sin_escalar = predictores_entrenar_sin_escalar
        self.y_entrenar_sin_escalar = y_entrenar_sin_escalar
        seleccionador_atributos = SeleccionadorAtributos()
        self.atributos_seleccionados = seleccionador_atributos.seleccionarAtributos(self.modelo, predictores_entrenar, y_entrenar, True)



    def podaAlfa(self):
        self.modelo.fit(self.predictores_entrenar, self.y_entrenar)
        camino = self.modelo.cost_complexity_pruning_path(self.predictores_entrenar[self.atributos_seleccionados], self.y_entrenar) 
        
        puntuaciones_entrenamiento = []
        puntuaciones_validacion = []
        valores_alfa = camino.ccp_alphas[:-1]

        for alfa in valores_alfa:
            arbol = self.crearModeloArbolDecision(alfa)
            arbol.fit(self.predictores_entrenar[self.atributos_seleccionados], self.y_entrenar)
            puntuaciones_entrenamiento.append(arbol.score(self.predictores_entrenar[self.atributos_seleccionados], self.y_entrenar))
            # Resetear el modelo
            arbol = clone(arbol)
            puntuacion_validacion_cruzada = cross_val_score(estimator = arbol, X = self.predictores_entrenamiento_validacion[self.atributos_seleccionados], y = self.y_entrenamiento_validacion, cv = self.indices_entrenamiento_validacion, n_jobs = 1)
            puntuaciones_validacion.append(np.mean(puntuacion_validacion_cruzada))

        indice_mejor_alfa = np.argmax(puntuaciones_validacion)
        mejor_alfa = valores_alfa[indice_mejor_alfa]
        self.modelo = self.crearModeloArbolDecision(mejor_alfa)
        figura = self.figuraPuntuacionSegunAlfa(valores_alfa, puntuaciones_entrenamiento, puntuaciones_validacion, mejor_alfa)

        return mejor_alfa, figura


    def crearModeloArbolDecision(self, alfa):
        if self.es_clasificacion:
            arbol = DecisionTreeClassifier(random_state = self.ESTADO_ALEATORIO, ccp_alpha = alfa)
        else:
            arbol = DecisionTreeRegressor(random_state = self.ESTADO_ALEATORIO, ccp_alpha = alfa)
        return arbol


    def figuraPuntuacionSegunAlfa(self, valores_alfa, puntuaciones_entrenamiento, puntuaciones_validacion, mejor_alfa):
        figura = go.Figure()
        figura.add_trace(go.Scatter(x = valores_alfa, y = puntuaciones_entrenamiento, name = self.etiqueta_entrenamiento, marker = dict(color = self.color_violeta)))
        figura.add_trace(go.Scatter(x = valores_alfa, y = puntuaciones_validacion, name = self.etiqueta_test, marker = dict(color = self.color_azul)))

        figura.update_layout(title = self.nombre_puntuacion.capitalize() + self.titulo_grafica_mejor_alfa + self.nombre_arbol, 
                            xaxis_title = self.etiqueta_alfa, yaxis_title = self.nombre_puntuacion.capitalize(), title_x = self.POSX_TITULO_GRAFICA, title_y = self.POSY_TITULO_GRAFICA)

        # Acotar el rango del eje x para facilitar la visualización
        if mejor_alfa < 0.001:
            if mejor_alfa < 0.00005:
                rango_x = (0, 0.0005)
            else:
                rango_x = (0, 0.002)
            figura.update_xaxes(range = rango_x)
        
        figura.add_vline(x = mejor_alfa, line_color = self.color_rojo, line_dash = self.modo_linea_puntos, 
                        annotation_text = self.etiqueta_mejor_alfa + str(round(mejor_alfa, 5)), annotation_position = self.posicion_arriba_derecha)
        
        return figura


    def arbolPodado(self):
        self.mejor_alfa, figura_mejor_alfa = self.podaAlfa()
        self.modelo = self.modelo.fit(self.predictores_entrenar[self.atributos_seleccionados], self.y_entrenar)
        puntuacion = self.modelo.score(self.predictores_prueba[self.atributos_seleccionados], self.y_prueba)
        return round(self.mejor_alfa, 5), puntuacion, figura_mejor_alfa, (self.nombre_generico_modelo_arbol_decision_seis_atributos, self.modelo)

    
    def importanciaAtributos(self):
        importancia_permutacion = ImportanciaPermutacion()
        importancia_permutacion_df = importancia_permutacion.importanciaAtributos(self.modelo, self.predictores_entrenar, self.y_entrenar, self.atributos_seleccionados)
        importancia_atributos_df = pd.DataFrame(self.modelo.feature_importances_, index = self.atributos_seleccionados, columns = [self.nombre_importancia])
        importancia_atributos_df = importancia_atributos_df.sort_values(by = self.nombre_importancia, ascending = False)
        titulo = self.titulo_grafica_importancia_atributos + self.nombre_arbol
        figura = importancia_permutacion.graficaImportanciaAtributos(importancia_atributos_df, titulo)
        return importancia_permutacion_df, figura


    def figuraGrafoArbol(self):
        datos_predictores_sin_escalar_mejores_atributos = self.predictores_entrenar_sin_escalar[self.atributos_seleccionados]
        self.modelo = self.modelo.fit(datos_predictores_sin_escalar_mejores_atributos, self.y_entrenar_sin_escalar)

        figura = plt.figure(dpi = self.DPI_FIGURA_ARBOL, figsize = self.TAM_FIGURA_ARBOL)
        tree.plot_tree(self.modelo, feature_names = self.atributos_seleccionados,  class_names = self.nombres_clases_figura_arbol, 
                        max_depth = self.PROFUNDIDAD_FIGURA_ARBOL, rounded = True, filled = True,  proportion = True, fontsize = self.TAM_TEXTO_FIGURA_ARBOL)
        
        # Quitar el fondo
        figura.patch.set_alpha(0)

        return figura