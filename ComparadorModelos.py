


import numpy as np
import pandas as pd

# Para seleccionar los atributos en el modelo de votación
from SeleccionadorAtributos import SeleccionadorAtributos

from ImportanciaPermutacion import ImportanciaPermutacion

# Importancia por permutación
from sklearn.inspection import permutation_importance

# Gráficos
import plotly.express as px

# Modelos de votación
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor



class ComparadorModelos:



    POSX_TITULO_GRAFICA = 0.5
    POSY_TITULO_GRAFICA = 0.9
    nombre_puntuacion = 'puntuación'
    nombre_importancia = 'importancia'
    nombre_clasificacion = 'clasificación'
    nombre_regresion = 'regresión'
    nombre_precision = 'precisión'
    nombre_r2 = 'R^2'
    etiqueta_importancia = 'Importancia'
    etiqueta_x = 'x'
    etiqueta_y = 'y'
    etiqueta_atributos = 'Atributos'
    nombre_modelo_votacion = 'Votación de modelos'
    tipo_votacion_suave = 'soft'
    tipo_votacion_dura = 'hard'
    modo_texto_fuera_barras = 'outside'
    titulo_grafica_puntuaciones_globales_inicio = 'Importancia de los atributos en el modelo con mayor '
    titulo_grafica_comparacion_modelos_intermedio = ' en los distintos modelos'
    nombre_titulo_grafica_comparacion_modelos_final = ' para predecir '
    nombre_modelo_regresion_basico = 'Regresión'
    nombre_modelo_arbol_decision_basico = 'Árbol de decisión'
    nombre_modelo_votacion_basico = 'Votación'
    nombre_modelo_regresion_un_atributo = 'Regresión con 1 atributo'
    nombre_modelo_regresion_logistica_umbral_por_defecto = ',<br>umbral 0.5'
    nombre_umbral = 'umbral'
    nombre_modelo_regresion_seis_atributos = 'Regresión con 6 atributos'
    nombre_modelo_arbol_decision_seis_atributos = 'Árbol de decisión con 6 atributos'
    nombre_modelo_regresion_tres_atributos_inicio = 'Regresión con ' 
    nombre_modelo_regresion_tres_atributos_fin = ' atributos'
    secuencia_colores = px.colors.qualitative.Set2



    def __init__(self, columna_y, puntuacion_mejor_predictor_individual, puntuacion_atributos_seleccionados_regresion, puntuacion_atributos_seleccionados_arbol_decision, puntuacion_atributos_seleccionados_red_neuronal, puntuacion_mejor_combinacion_regresion, num_atributos_combinacion, nombre_modelo_red_neuronal, alfa_arbol_decision, importancia_atributos_seleccionados_regresion, importancia_combinacion_atributos_regresion, importancia_atributos_arbol_decision, importancia_atributos_red_neuronal, es_clasificacion, precision_mejor_umbral_regresion_logistica = None, mejor_umbral_regresion_logistica = None):
        self.atributos_seleccionados_votacion = []
        self.puntuacion_atributos_seleccionados_arbol_decision = puntuacion_atributos_seleccionados_arbol_decision
        self.puntuacion_atributos_seleccionados_red_neuronal = puntuacion_atributos_seleccionados_red_neuronal
        self.puntuacion_mejor_combinacion_regresion = puntuacion_mejor_combinacion_regresion
        self.puntuacion_atributos_seleccionados_regresion = puntuacion_atributos_seleccionados_regresion
        self.puntuacion_mejor_predictor_individual = puntuacion_mejor_predictor_individual
        self.nombre_modelo_red_neuronal = nombre_modelo_red_neuronal
        self.num_atributos_combinacion = num_atributos_combinacion
        self.alfa_arbol_decision = alfa_arbol_decision
        self.es_clasificacion = es_clasificacion
        self.importancia_permutacion = ImportanciaPermutacion()
        self.seleccionador_atributos = SeleccionadorAtributos()

        if es_clasificacion:
            self.nombre_tipo_puntuacion = self.nombre_precision
            self.precision_mejor_umbral_regresion_logistica = precision_mejor_umbral_regresion_logistica
            self.mejor_umbral_regresion_logistica = mejor_umbral_regresion_logistica
            self.titulo_grafica_comparacion_modelos_final = ' (' + self.nombre_clasificacion + self.nombre_titulo_grafica_comparacion_modelos_final + columna_y + ')'
            self.nombre_modelo_regresion_un_atributo += self.nombre_modelo_regresion_logistica_umbral_por_defecto
            self.nombre_modelo_regresion_seis_atributos += self.nombre_modelo_regresion_logistica_umbral_por_defecto
        else:
            self.nombre_tipo_puntuacion = self.nombre_r2
            self.titulo_grafica_comparacion_modelos_final = ' (' + self.nombre_regresion + self.nombre_titulo_grafica_comparacion_modelos_final + columna_y + ')'

        # Importancia de los atributos en cada modelo
        if importancia_atributos_seleccionados_regresion is not None:
            self.importancia_atributos_regresion = importancia_atributos_seleccionados_regresion
        else:
            self.importancia_atributos_regresion = importancia_combinacion_atributos_regresion
        
        self.importancia_atributos_arbol_decision = importancia_atributos_arbol_decision
        self.importancia_atributos_red_neuronal = importancia_atributos_red_neuronal

        # Pesos de los modelos en la votación
        puntuaciones_regresion = [puntuacion_atributos_seleccionados_regresion, puntuacion_mejor_combinacion_regresion, precision_mejor_umbral_regresion_logistica]
        self.pesos_modelos_votacion = [np.max(np.array(puntuaciones_regresion, dtype = np.float64)), puntuacion_atributos_seleccionados_arbol_decision, puntuacion_atributos_seleccionados_red_neuronal]       



    def votacionClasificacion(self, tipo_votacion, modelo_regresion, modelo_arbol_decision, modelo_red_neuronal, predictores_entrenar, y_entrenar, predictores_validacion, y_validacion):

        votacion_modelos_clasificacion = VotingClassifier(estimators = [modelo_regresion, modelo_arbol_decision, modelo_red_neuronal],
                                                          n_jobs = 1, voting = tipo_votacion, weights = self.pesos_modelos_votacion)

        atributos_seleccionados_votacion = self.seleccionador_atributos.seleccionarAtributos(votacion_modelos_clasificacion, predictores_entrenar, y_entrenar, False)
        votacion_modelos_clasificacion.fit(predictores_entrenar[atributos_seleccionados_votacion], y_entrenar)
        precision_modelo_votacion_clasificacion = votacion_modelos_clasificacion.score(predictores_validacion[atributos_seleccionados_votacion], y_validacion)    
        return precision_modelo_votacion_clasificacion, votacion_modelos_clasificacion, atributos_seleccionados_votacion


    def votacionModelos(self, modelo_regresion, modelo_arbol_decision, modelo_red_neuronal, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_validacion, y_validacion):

        if self.es_clasificacion:

            # Coger el tipo de votación con mayor puntuación en la validación (soft vs hard)
            puntuacion_modelo_votacion_hard, votacion_modelos_clasificacion_hard, atributos_seleccionados_hard = self.votacionClasificacion(self.tipo_votacion_dura, modelo_regresion, modelo_arbol_decision, modelo_red_neuronal, predictores_entrenar, y_entrenar, predictores_validacion, y_validacion)
            puntuacion_modelo_votacion_soft, votacion_modelos_clasificacion_soft, atributos_seleccionados_soft = self.votacionClasificacion(self.tipo_votacion_suave, modelo_regresion, modelo_arbol_decision, modelo_red_neuronal, predictores_entrenar, y_entrenar, predictores_validacion, y_validacion)
    
            if puntuacion_modelo_votacion_soft >= puntuacion_modelo_votacion_hard:
                self.atributos_seleccionados_votacion = atributos_seleccionados_soft
                self.votacion_modelos = votacion_modelos_clasificacion_soft
                self.tipo_mejor_votacion_modelos = self.tipo_votacion_suave
                self.puntuacion_votacion_modelos = votacion_modelos_clasificacion_soft.score(predictores_prueba[self.atributos_seleccionados_votacion], y_prueba)
            else:
                self.atributos_seleccionados_votacion = atributos_seleccionados_hard
                self.votacion_modelos = votacion_modelos_clasificacion_hard
                self.tipo_mejor_votacion_modelos = self.tipo_votacion_dura
                self.puntuacion_votacion_modelos = votacion_modelos_clasificacion_hard.score(predictores_prueba[self.atributos_seleccionados_votacion], y_prueba)

        else:
            votacion_modelos_regresion = VotingRegressor(estimators = [modelo_regresion, modelo_arbol_decision, modelo_red_neuronal], n_jobs = 1)   
            self.atributos_seleccionados_votacion = self.seleccionador_atributos.seleccionarAtributos(votacion_modelos_regresion, predictores_entrenar, y_entrenar, False)
            votacion_modelos_regresion.fit(predictores_entrenar[self.atributos_seleccionados_votacion], y_entrenar)
            self.puntuacion_votacion_modelos = votacion_modelos_regresion.score(predictores_prueba[self.atributos_seleccionados_votacion], y_prueba)
            self.votacion_modelos = votacion_modelos_regresion
        

    def importanciaAtributosSegunModelo(self, nombre_modelo, titulo):
        if self.nombre_modelo_regresion_basico in nombre_modelo:
            importancia_atributos = self.importancia_atributos_regresion
        elif self.nombre_modelo_arbol_decision_basico in nombre_modelo:
            importancia_atributos = self.importancia_atributos_arbol_decision
        else:
            importancia_atributos = self.importancia_atributos_red_neuronal

        figura = self.importancia_permutacion.graficaImportanciaAtributos(importancia_atributos, titulo)
        
        return importancia_atributos, figura


    def atributoMayorImportancia(self, predictores, y, puntuaciones_modelos):

        titulo = self.titulo_grafica_puntuaciones_globales_inicio + self.nombre_tipo_puntuacion + '<br>' + self.titulo_grafica_comparacion_modelos_final
        nombre_modelo_mayor_puntuacion = puntuaciones_modelos.index.values[-1]

        if self.nombre_modelo_votacion_basico in nombre_modelo_mayor_puntuacion:
            # El modelo de votación ha obtenido la mayor puntuación -> Calcular la importancia en el modelo de votación
            puntuaciones_globales_atributos, figura = self.calcularPuntuacionGlobal(predictores, y, titulo)
        else:
            puntuaciones_globales_atributos, figura = self.importanciaAtributosSegunModelo(nombre_modelo_mayor_puntuacion, titulo)

        return figura, puntuaciones_globales_atributos.index[0], nombre_modelo_mayor_puntuacion


    def calcularPuntuacionGlobal(self, predictores, y, titulo):
        puntuacion_global = self.importancia_permutacion.importanciaAtributos(self.votacion_modelos, predictores, y, self.atributos_seleccionados_votacion)
        figura = self.importancia_permutacion.graficaImportanciaAtributos(puntuacion_global, titulo)    
        return puntuacion_global, figura


    def figuraComparacionModelos(self):
        puntuaciones_modelos, nombres_modelos = self.puntuacionesNombresComparacionModelos()
        puntuaciones_modelos_df = pd.DataFrame(puntuaciones_modelos, index = nombres_modelos, columns = [self.nombre_puntuacion])
        puntuaciones_modelos_df.sort_values(by = [self.nombre_puntuacion], inplace = True)

        figura = px.bar(x = puntuaciones_modelos_df.index, y = puntuaciones_modelos_df[self.nombre_puntuacion], 
                        title = self.nombre_tipo_puntuacion.capitalize() + self.titulo_grafica_comparacion_modelos_intermedio + self.titulo_grafica_comparacion_modelos_final,
                        color = nombres_modelos, color_discrete_sequence = self.secuencia_colores, text_auto = True, 
                        labels = {
                                    self.etiqueta_x: self.etiqueta_atributos,
                                    self.etiqueta_y: self.nombre_tipo_puntuacion.capitalize()
                                })

        figura.update_layout(title_x = self.POSX_TITULO_GRAFICA, title_y = self.POSY_TITULO_GRAFICA, yaxis_range = (0, 1.1))
        figura.update_traces(showlegend = False, textposition = self.modo_texto_fuera_barras)
        return figura, puntuaciones_modelos_df

    
    def puntuacionesNombresComparacionModelos(self):
        puntuaciones_modelos = [self.puntuacion_mejor_predictor_individual, self.puntuacion_atributos_seleccionados_arbol_decision, self.puntuacion_atributos_seleccionados_red_neuronal]
        nombres_modelos = [self.nombre_modelo_regresion_un_atributo, self.nombre_modelo_arbol_decision_seis_atributos + ',<br>alfa ' + str(self.alfa_arbol_decision), self.nombre_modelo_red_neuronal]

        if self.puntuacion_atributos_seleccionados_regresion != -1 or self.puntuacion_mejor_combinacion_regresion != -1:

            # Modelo de regresión con los 6 atributos seleccionados
            if self.puntuacion_atributos_seleccionados_regresion != -1:
                nombre_modelo_mejor_umbral_inicio = self.puntuacion_atributos_seleccionados_regresion
                puntuaciones_modelos.append(self.puntuacion_atributos_seleccionados_regresion)
                nombres_modelos.append(self.nombre_modelo_regresion_seis_atributos)

            # Modelo de regresión con la mejor combinación de menos de 4 atributos
            if self.puntuacion_mejor_combinacion_regresion != -1:
                puntuaciones_modelos.append(self.puntuacion_mejor_combinacion_regresion)
                nombre_modelo_mejor_umbral_inicio = self.nombre_modelo_regresion_tres_atributos_inicio + str(self.num_atributos_combinacion) + self.nombre_modelo_regresion_tres_atributos_fin
                if self.es_clasificacion:
                    nombres_modelos.append(nombre_modelo_mejor_umbral_inicio + self.nombre_modelo_regresion_logistica_umbral_por_defecto)
                else:
                    nombres_modelos.append(nombre_modelo_mejor_umbral_inicio)

            if self.es_clasificacion:
                # Modelo de regresión logística con el mejor umbral
                puntuaciones_modelos.append(self.precision_mejor_umbral_regresion_logistica)
                nombres_modelos.append(nombre_modelo_mejor_umbral_inicio + ',<br>' + self.nombre_umbral + ' ' + str(self.mejor_umbral_regresion_logistica))
        

        nombre_modelo_votacion = self.nombre_modelo_votacion
        if self.es_clasificacion:
            nombre_modelo_votacion += '<br>(' + self.tipo_mejor_votacion_modelos + ')'
        
        puntuaciones_modelos.append(self.puntuacion_votacion_modelos)
        nombres_modelos.append(nombre_modelo_votacion)

        return puntuaciones_modelos, nombres_modelos
        