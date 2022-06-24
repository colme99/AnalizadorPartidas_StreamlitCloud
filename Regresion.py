


import pandas as pd
import numpy as np
import itertools

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Curva ROC
from sklearn.metrics import roc_curve

# Factor de inflación de la varianza
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Comprobar el valor de p
import statsmodels.api as sm

# Validación cruzada
from sklearn.model_selection import cross_val_score

from SeleccionadorAtributos import SeleccionadorAtributos
from ImportanciaPermutacion import ImportanciaPermutacion



class Regresion:



    MAX_COMBINACIONES = 3
    POSX_TITULO_GRAFICA = 0.5
    POSY_TITULO_GRAFICA = 0.9
    UMBRAL_INFLACION_VARIANZA = 12.5
    GROSOR_LINEA_CURVA_ROC = 3
    GROSOR_PUNTO_UMBRAL = 8
    ANCHURA_POR_ATRIBUTO_GRAFICA_PESOS = 120
    RANGO_Y_GRAFICA_PUNTUACION = (0, 1.1)
    TAM_GRAFICA_DOS_MEJORES_PREDICTORES_INDIVIDUALES = (10,5)
    ESTADO_ALEATORIO = 333
    mostrar_modelo_atributos_seleccionados = False
    etiqueta_x = 'x'
    etiqueta_y = 'y'
    etiqueta_indice = 'index'
    etiqueta_atributos = 'Atributos'
    etiqueta_pesos = 'Pesos'
    metodo_calcular_umbral_youden= 'youden'
    metodo_calcular_umbral_min_distancia= 'min_distancia'
    etiqueta_curva_ROC = 'Curva ROC'
    etiqueta_mejor_umbral = 'Umbral seleccionado: '
    etiqueta_FPR = 'Ratio de falsos positivos (FPR)'
    etiqueta_TPR = 'Ratio de verdaderos positivos (TPR)'
    nombre_inflacion_varianza = 'inflacion_varianza'
    color_azul = 'cornflowerblue'
    color_rojo = 'lightcoral'
    modo_lineas = 'lines'
    titulo_grafica_pesos_predictores_inicio = 'Pesos de los atributos '
    titulo_grafica_pesos_predictores_fin = ' atributos'
    nombre_regresion_logistica = 'regresión logística'
    nombre_regresion_lineal = 'regresión lineal'
    nombre_regresion_intermedio = ' para predecir '
    nombre_precision = 'precisión'
    nombre_modelo_regresion_generico_un_atributo = 'Regresión 1 atributo'
    nombre_modelo_regresion_generico_inicio = 'Regresión '
    nombre_modelo_regresion_generico_fin = ' atributos'
    nombre_r2 = 'R^2'
    nombre_color = 'color'
    nombre_columna_peso = 'peso'
    nombre_puntuacion = 'puntuacion'
    nombre_columna_importancia = 'importancia'
    nombre_umbral_por_defecto = ',<br>umbral 0.5'
    nombre_umbral_generico = ',<br>umbral '
    titulo_graficas_final_regresion_logistica = ' (regresión logística)'
    titulo_graficas_final_regresion_lineal = ' (regresión lineal)'
    titulo_grafica_predictores_individuales_inicio = 'Mejores predictores individuales '
    titulo_grafica_comparacion_modelos_inicio = 'Comparación de '
    titulo_grafica_comparacion_modelos_fin = ' entre modelos '
    titulo_grafica_curva_ROC_inicio = 'Curva ROC con el umbral seleccionado ('
    titulo_grafica_curva_ROC_final = ' atributos '
    titulo_grafica_dos_mejores_predictores_individuales = 'Ajuste de los dos mejores predictores individuales al modelo '
    etiqueta_pesos = 'Pesos'
    etiqueta_modelo = 'Modelo'
    nombre_modelo_seis_predictores = '6 atributos'
    nombre_modelo_un_predictor = '1 atributo'
    nombre_modelo_tres_o_menos_predictores_final = ' atributos '
    modo_texto_fuera_barras = 'outside'
    orientacion_horizontal = 'h'
    secuencia_colores = px.colors.qualitative.Set2



    def __init__(self, predictores_entrenar, predictores_validacion, predictores_prueba, y_entrenar, y_validacion, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, nombre_y, es_clasificacion):
        self.puntuaciones_modelos = []
        self.nombres_modelos = []
        self.modelos = []
        self.predictores_entrenar = predictores_entrenar
        self.y_entrenar = y_entrenar.values.ravel()
        self.predictores_validacion = predictores_validacion
        self.y_validacion = y_validacion
        self.predictores_prueba = predictores_prueba
        self.y_prueba = y_prueba.values.ravel()
        self.predictores_entrenamiento_validacion = predictores_entrenamiento_validacion
        self.y_entrenamiento_validacion = y_entrenamiento_validacion.values.ravel()
        self.indices_entrenamiento_validacion = indices_entrenamiento_validacion
        self.nombre_y = nombre_y

        self.es_clasificacion = es_clasificacion
        if es_clasificacion:
            self.modelo = LogisticRegression(random_state = self.ESTADO_ALEATORIO)
            self.nombre_puntuacion = self.nombre_precision
            self.titulo_graficas_final = self.titulo_graficas_final_regresion_logistica
            self.nombre_regresion = '<br>(' + self.nombre_regresion_logistica + self.nombre_regresion_intermedio + nombre_y + ')'
        else:
            self.modelo = LinearRegression()
            self.nombre_puntuacion = self.nombre_r2
            self.titulo_graficas_final = self.titulo_graficas_final_regresion_lineal
            self.nombre_regresion = '<br>(' + self.nombre_regresion_lineal + self.nombre_regresion_intermedio + nombre_y + ')'

        seleccionador_atributos = SeleccionadorAtributos()
        self.atributos_seleccionados = seleccionador_atributos.seleccionarAtributos(self.modelo, predictores_entrenar, y_entrenar, True)



    def graficaPesosAtributos(self, importancia_atributos):
        figura = px.bar(x = importancia_atributos[self.nombre_columna_peso], y = importancia_atributos.index,
                        color = importancia_atributos.index, color_discrete_sequence = self.secuencia_colores,
                        title = self.titulo_grafica_pesos_predictores_inicio + self.nombre_regresion + ', ' + str(importancia_atributos.shape[0]) + self.titulo_grafica_pesos_predictores_fin,
                        labels = {
                                    self.etiqueta_x: self.etiqueta_pesos,
                                    self.etiqueta_y: self.etiqueta_atributos
                                })
        figura.update_layout(title_x = self.POSX_TITULO_GRAFICA, title_y = self.POSY_TITULO_GRAFICA, height = self.ANCHURA_POR_ATRIBUTO_GRAFICA_PESOS*importancia_atributos.shape[0])
        figura.update_traces(showlegend = False)
        return figura


    def pesosAtributosSeleccionados(self):
        figura, puntuacion, importancia_atributos = self.pesosPredictores(self.atributos_seleccionados)
        if puntuacion != -1:
            self.puntuaciones_modelos.append(puntuacion)
            nombre_modelo = self.nombre_modelo_seis_predictores
            if self.es_clasificacion:
                nombre_modelo += self.nombre_umbral_por_defecto
            self.nombres_modelos.append(nombre_modelo)
            self.mostrar_modelo_atributos_seleccionados = True
            self.modelos.append([puntuacion, (self.nombre_modelo_regresion_generico_inicio + str(len(self.atributos_seleccionados)) + self.nombre_modelo_regresion_generico_fin, self.modelo)])

        return figura, puntuacion, importancia_atributos


    def puntuacionesIndividualesRegresion(self):
        puntuaciones_regresion = []
        atributos_aceptados = []
        for columna in self.predictores_entrenar.columns:
            x_entrenamiento_array2D = self.predictores_entrenar[columna].values.reshape(-1,1)
            self.modelo.fit(x_entrenamiento_array2D, self.y_entrenar)
            x_entrenamiento_validacion_array_2D = self.predictores_entrenamiento_validacion[columna].values.reshape(-1,1)
            puntuacion_validacion_cruzada = cross_val_score(estimator = self.modelo, X = x_entrenamiento_validacion_array_2D, y = self.y_entrenamiento_validacion, cv = self.indices_entrenamiento_validacion, n_jobs = 1)
            puntuacion = np.mean(puntuacion_validacion_cruzada)
            
            if self.valorPBajoUmbral(columna):
                puntuaciones_regresion.append(puntuacion)
                atributos_aceptados.append(columna)

        indice_modelo_mejor_puntuacion_atributo_individual = np.argmax(puntuaciones_regresion)
        self.mejor_predictor_individual = atributos_aceptados[indice_modelo_mejor_puntuacion_atributo_individual]
        x_entrenamiento_array_2D = self.predictores_entrenar[self.mejor_predictor_individual].values.reshape(-1,1)
        puntuacion_prueba = self.modelo.fit(x_entrenamiento_array_2D, self.y_entrenar)
        x_prueba_array_2D = self.predictores_prueba[self.mejor_predictor_individual].values.reshape(-1,1)
        puntuacion_prueba = self.modelo.score(x_prueba_array_2D, self.y_prueba)

        nombre_modelo = self.nombre_modelo_un_predictor + ' (' + self.mejor_predictor_individual + ')'
        if self.es_clasificacion:
            nombre_modelo += self.nombre_umbral_por_defecto
        self.nombres_modelos.append(nombre_modelo)
    
        self.puntuaciones_modelos.append(puntuacion_prueba)
        self.modelos.append([puntuacion, [nombre_modelo, self.modelo]])

        puntuaciones_regresion_df = pd.DataFrame(puntuaciones_regresion, columns = [self.nombre_puntuacion], index = atributos_aceptados)
        puntuaciones_regresion_df.sort_values(by = self.nombre_puntuacion, ascending = False, inplace = True)

        return puntuacion_prueba, puntuaciones_regresion_df


    def atributosSeparadosComas(self, atributos):
        # Juntar los atributos (separados por comas), para mostrarlos en la gráfica
        atributos_separados_comas = '('
        num_atributos = len(atributos)
        for i in range(num_atributos):
            atributos_separados_comas += atributos[i]
            if i < num_atributos-1:
                atributos_separados_comas += ', '
        atributos_separados_comas += ')'
        return atributos_separados_comas


    def seleccionarUmbral(self):
        figura = None
        # Modelos con 6 ó 2-3 atributos
        if self.mostrar_modelo_atributos_seleccionados or self.combinacion is not None:

            if self.mostrar_modelo_atributos_seleccionados:
                nombre_modelo = self.nombre_modelo_seis_predictores
                atributos = self.atributos_seleccionados
            else:
                nombre_modelo = str(self.num_atributos_combinacion) + self.nombre_modelo_tres_o_menos_predictores_final
                atributos = self.combinacion
                nombre_modelo += self.atributosSeparadosComas(atributos)            

            figura, puntuacion, mejor_umbral = self.puntuacionUmbralSeleccionado(atributos, nombre_modelo)
        
        else:
            # Modelo con el mejor predictor individual (1 atributo)
            figura, puntuacion, mejor_umbral = self.puntuacionUmbralSeleccionado(atributos, self.nombre_modelo_regresion_generico_un_atributo)

        return figura, puntuacion, nombre_modelo, round(mejor_umbral, 2)


    def puntuacionUmbralSeleccionado(self, atributos, nombre_modelo):
        self.modelo.fit(self.predictores_entrenar[atributos], self.y_entrenar)
        y_prediccion_probabilidad_validacion = self.modelo.predict_proba(self.predictores_validacion[atributos])[:, 1]
        ratio_falsos_positivos, ratio_verdaderos_positivos, umbrales = roc_curve(self.y_validacion, y_prediccion_probabilidad_validacion)
        indice_umbral, umbral_seleccionado, puntuacion = self.seleccionarUmbralMayorPuntuacion(ratio_verdaderos_positivos, 1 - ratio_falsos_positivos, umbrales, y_prediccion_probabilidad_validacion, atributos)
        self.puntuaciones_modelos.append(puntuacion)
        nombre_modelo += self.nombre_umbral_generico + str(round(umbral_seleccionado, 2))
        self.nombres_modelos.append(nombre_modelo)
        figura = self.figuraCurvaROCUmbralSeleccionado(ratio_verdaderos_positivos, ratio_falsos_positivos, indice_umbral, umbral_seleccionado, len(atributos))
        return figura, puntuacion, umbral_seleccionado


    def puntuacionSegunUmbral(self, y_prediccion, y_real, umbral):
        y_prediccion_umbral = (y_prediccion >= umbral).astype(int)
        return accuracy_score(y_real, y_prediccion_umbral)


    def figuraCurvaROCUmbralSeleccionado(self, ratio_verdaderos_positivos, ratio_falsos_positivos, indice_umbral_seleccionado, umbral_seleccionado, num_atributos):
        traza_curva_ROC = go.Scatter(x = ratio_falsos_positivos,y = ratio_verdaderos_positivos, mode = self.modo_lineas,
                                    name = self.etiqueta_curva_ROC, marker = dict(color = self.color_azul), line = dict(width = self.GROSOR_LINEA_CURVA_ROC))

        traza_mejor_umbral = go.Scatter(x = [ratio_falsos_positivos[indice_umbral_seleccionado]],y = [ratio_verdaderos_positivos[indice_umbral_seleccionado]],
                                        name = self.etiqueta_mejor_umbral + str(round(umbral_seleccionado, 2)), marker = dict(color = self.color_rojo, size = self.GROSOR_PUNTO_UMBRAL))

        figura = make_subplots()
        figura.add_trace(traza_curva_ROC)
        figura.add_trace(traza_mejor_umbral)
        figura.update_layout(title_x = self.POSX_TITULO_GRAFICA, title_y = self.POSY_TITULO_GRAFICA, 
                            title_text = self.titulo_grafica_curva_ROC_inicio + str(round(umbral_seleccionado, 2)) + '), ' + str(num_atributos) + self.titulo_grafica_curva_ROC_final + self.nombre_regresion, 
                            xaxis_title = self.etiqueta_FPR, yaxis_title = self.etiqueta_TPR,
                            xaxis = dict(showgrid = False), yaxis = dict(showgrid = False))

        return figura


    def seleccionarUmbralMayorPuntuacion(self, sensibilidad, especificidad, umbrales, y_prediccion_validacion, atributos):
        indice_umbral_youden, umbral_seleccionado_youden, puntuacion_umbral_youden = self.seleccionarUmbralSegunMetodo(sensibilidad, especificidad, umbrales, y_prediccion_validacion, self.metodo_calcular_umbral_youden)
        indice_umbral_min_distancia, umbral_seleccionado_min_distancia, puntuacion_umbral_min_distancia = self.seleccionarUmbralSegunMetodo(sensibilidad, especificidad, umbrales, y_prediccion_validacion, self.metodo_calcular_umbral_min_distancia)

        if puntuacion_umbral_youden >= puntuacion_umbral_min_distancia:
            indice_umbral = indice_umbral_youden
            umbral_seleccionado = umbral_seleccionado_youden

        else:
            indice_umbral = indice_umbral_min_distancia
            umbral_seleccionado = umbral_seleccionado_min_distancia

        y_prediccion_probabilidad_prueba = self.modelo.predict_proba(self.predictores_prueba[atributos])[:, 1]
        puntuacion = self.puntuacionSegunUmbral(y_prediccion_probabilidad_prueba, self.y_prueba, umbral_seleccionado)
        
        return indice_umbral, umbral_seleccionado, puntuacion


    def seleccionarUmbralSegunMetodo(self, sensibilidad, especificidad, umbrales, y_prediccion_prueba, nombre_metodo):
        
        if nombre_metodo == self.metodo_calcular_umbral_youden:
            # Índice de Youden
            y = sensibilidad - (1 - especificidad)
            indice_umbral = np.argmax(y)

        elif nombre_metodo == self.metodo_calcular_umbral_min_distancia:
            # Distancia mínima a la esquina superior izquierda (0, 1)
            d = np.sqrt(np.power(1 - sensibilidad, 2) + np.power(1 - especificidad, 2))
            indice_umbral = np.argmin(d)

        umbral_seleccionado = umbrales[indice_umbral]

        puntuacion = self.puntuacionSegunUmbral(y_prediccion_prueba, self.y_validacion, umbral_seleccionado)
        return indice_umbral, umbral_seleccionado, puntuacion


    def seleccionarModeloMayorPuntuacion(self):
        self.modelos.sort(key = lambda modelo: modelo[0], reverse = True)
        return self.modelos[0][1]


    def figuraComparacionModelos(self):

        puntuaciones_modelos_df = pd.DataFrame(self.puntuaciones_modelos, index = self.nombres_modelos, columns = [self.nombre_puntuacion])
        puntuaciones_modelos_df.sort_values(by = [self.nombre_puntuacion], inplace = True)

        figura = px.bar(y = puntuaciones_modelos_df[self.nombre_puntuacion], x = puntuaciones_modelos_df.index, text_auto = True,
                        color = self.nombres_modelos, color_discrete_sequence = self.secuencia_colores,
                        title = self.titulo_grafica_comparacion_modelos_inicio + self.nombre_puntuacion + self.titulo_grafica_comparacion_modelos_fin + self.nombre_regresion,
                        labels = {
                                    self.etiqueta_x: self.etiqueta_modelo
                                })

        figura.update_layout(title_x = self.POSX_TITULO_GRAFICA, title_y = self.POSY_TITULO_GRAFICA, yaxis_range = (0, 1.1))
        figura.update_traces(showlegend = False, textposition = self.modo_texto_fuera_barras)
        return figura, self.seleccionarModeloMayorPuntuacion()

    
    def figuraGraficaDosMejoresPredictoresIndividuales(self, puntuaciones_individuales_regresion):

        figura, ejes = plt.subplots(1, 2, figsize = self.TAM_GRAFICA_DOS_MEJORES_PREDICTORES_INDIVIDUALES)

        for i in range(2):
            columna = puntuaciones_individuales_regresion.index.values[i]
            sns.violinplot(x = self.predictores_entrenar[columna], y = self.y_entrenar, cut = 0, bw = 0.8, orient = self.orientacion_horizontal, 
                        color = self.color_azul, alpha = 0.05, ax = ejes[i])
            sns.regplot(x = self.predictores_entrenar[columna], y = self.y_entrenar, line_kws = {self.nombre_color: self.color_rojo}, logistic = True, 
                        scatter = False, ax = ejes[i])
            ejes[i].grid(False)
        
        figura.suptitle(self.titulo_grafica_dos_mejores_predictores_individuales + self.nombre_regresion.replace('<br>', ''))

        return figura


    def inflacionVarianza(self, atributos):
        inflacion = []
        for i in range(len(atributos)):
            inflacion.append(variance_inflation_factor(self.predictores_entrenar[atributos].values, i))

        inflacion_varianza = pd.DataFrame(inflacion, columns = [self.nombre_inflacion_varianza], index = atributos)
        inflacion_varianza = inflacion_varianza.sort_values(by = [self.nombre_inflacion_varianza])
        return inflacion_varianza


    def valorPBajoUmbral(self, atributos):
        datos_x = sm.add_constant(self.predictores_entrenar[atributos])
        modelo = sm.OLS(self.y_entrenar, datos_x)
        modelo = modelo.fit()
        return modelo.pvalues[modelo.pvalues <= 0.05].shape[0] == modelo.pvalues.shape[0]


    def pesosPredictores(self, atributos):
        inflacion_varianza = self.inflacionVarianza(atributos)
        
        figura = None
        importancia_atributos = None
        puntuacion = -1
        # Sólo se muestran al usuario si la inflación de la varianza no es muy elevada y se alcanza la significación estadística
        if self.valorPBajoUmbral(atributos) and self.inflacionVarianzaBajoUmbral(inflacion_varianza):
            self.modelo.fit(self.predictores_entrenar[atributos], self.y_entrenar)
            puntuacion = self.modelo.score(self.predictores_prueba[atributos], self.y_prueba)           

            if self.es_clasificacion:
                pesos = self.modelo.coef_[0]
            else:
                pesos = self.modelo.coef_

            pesos_atributos = pd.DataFrame(np.transpose([pesos, abs(pesos)]), columns = [self.nombre_columna_peso, self.nombre_columna_importancia], index = atributos)

            pesos_atributos.sort_values(by = self.nombre_columna_importancia, ascending = False, inplace = True)
            figura = self.graficaPesosAtributos(pesos_atributos)

            importancia_permutacion = ImportanciaPermutacion()
            importancia_atributos = importancia_permutacion.importanciaAtributos(self.modelo, self.predictores_entrenar, self.y_entrenar, atributos)

        return figura, puntuacion, importancia_atributos

    
    def inflacionVarianzaBajoUmbral(self, inflacion_varianza):
        return inflacion_varianza[inflacion_varianza[self.nombre_inflacion_varianza] > self.UMBRAL_INFLACION_VARIANZA].shape[0] == 0


    def figuraMejoresCombinacionesAtributosSeleccionados(self):
        puntuacionCombinacionesAtributosSeleccionados = self.combinacionesAtributosSeleccionados()

        figura = None
        self.combinacion = None
        puntuacion = -1
        self.num_atributos_combinacion = 0
        i = 0
        # Mejor combinación de los atributos seleccionados (que cumpla los criterios de baja inflación de la varianza y significación estdística) 
        while i < len(puntuacionCombinacionesAtributosSeleccionados) and puntuacion == -1:
            figura, puntuacion, importancia = self.pesosPredictores(puntuacionCombinacionesAtributosSeleccionados[i][1])
            i += 1

        if puntuacion != -1:
            self.puntuaciones_modelos.append(puntuacion)
            self.num_atributos_combinacion = len(puntuacionCombinacionesAtributosSeleccionados[i-1][1])
            self.combinacion = puntuacionCombinacionesAtributosSeleccionados[i-1][1]
            nombre_modelo = str(self.num_atributos_combinacion) + self.nombre_modelo_tres_o_menos_predictores_final + self.atributosSeparadosComas(self.combinacion)
            if self.es_clasificacion:
                nombre_modelo += self.nombre_umbral_por_defecto
            self.nombres_modelos.append(nombre_modelo)
            self.modelos.append([puntuacion, (self.nombre_modelo_regresion_generico_inicio + str(self.num_atributos_combinacion) + self.nombre_modelo_regresion_generico_fin, self.modelo)])

        return figura, puntuacion, self.num_atributos_combinacion, importancia


    def combinacionesAtributosSeleccionados(self):
        puntuacionCombinacionesAtributosSeleccionados = []
        for i in range(self.atributos_seleccionados.shape[0]):
            for tupla in itertools.combinations(self.atributos_seleccionados, i):
                num_características = len(tupla)
                # Combinaciones de 2 o 3 atributos
                if num_características > 1 and num_características <= self.MAX_COMBINACIONES:
                    combinacion = list(tupla)
                    self.modelo.fit(self.predictores_entrenar[combinacion], self.y_entrenar)
                    puntuacion_validacion_cruzada = cross_val_score(estimator = self.modelo, X = self.predictores_entrenamiento_validacion[combinacion], y = self.y_entrenamiento_validacion, cv = self.indices_entrenamiento_validacion, n_jobs = 1)
                    puntuacion = np.mean(puntuacion_validacion_cruzada)            
                    puntuacionCombinacionesAtributosSeleccionados.append([puntuacion, combinacion])

        puntuacionCombinacionesAtributosSeleccionados.sort(reverse = True)
        return puntuacionCombinacionesAtributosSeleccionados
