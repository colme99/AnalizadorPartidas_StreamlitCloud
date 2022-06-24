


import pandas as pd
import numpy as np

# Gráficos
import plotly.express as px

# Grafos
from graphviz import render

# Modelos de red neuronal
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Búsqueda de hiperparámetros
from sklearn.model_selection import GridSearchCV

from SeleccionadorAtributos import SeleccionadorAtributos
from ImportanciaPermutacion import ImportanciaPermutacion



class RedNeuronal:



    ESTADO_ALEATORIO = 33
    POSX_TITULO_GRAFICA = 0.5
    POSY_TITULO_GRAFICA = 0.9
    NUM_REPETICIONES_IMPORTANCIA_PERMUTACION = 20
    nombre_fichero_estructura_grafo_red_neuronal = 'estructura_grafo_red_neuronal.txt'
    patron_etiquetas_nodos_capa_oculta = '^^^'
    patron_orden_nodos_capa_oculta = '***'
    patron_arista_nodos_capa_oculta = '___'
    patron_anchura_grafo = '<<<'
    formato_dot = 'dot'
    formato_png = 'png'
    etiqueta_salida_estructura_grafo = 'salida'
    etiqueta_titulo_estructura_grafo = 'titulo'
    nombre_importancia = 'importancia'
    nombre_importancia_media_permutation_importance = 'importances_mean'
    nombre_generico_modelos_inicio = 'Red neuronal (6-'
    nombre_generico_modelos_intermedio = '-1),<br>activación '
    nombre_modelo_clasificador = 'clasificador'
    nombre_modelo_regresor = 'regresor'
    nombre_modelo_intermedio = ' con red neuronal para predecir '
    titulo_grafica_importancia_atributos = 'Importancia de los atributos de la red neuronal con mayor '
    secuencia_colores = px.colors.qualitative.Set2
    titulo_grafo_base = 'Estructura de la red neuronal '
    titulo_grafo_intermedio = ', activación '
    nombre_precision = 'precisión'
    nombre_r2 = 'R^2'
    formato_base_etiqueta_neurona_capa_oculta_inicio = ' [label = <n<sub>'
    formato_base_etiqueta_neurona_capa_oculta_fin = '</sub>>];'

    # Búsqueda de hiperparámetros
    MAX_ITERACIONES = 500
    ESTADO_ALEATORIO = 11
    ALFA = 0.0001
    nombre_parametro_funcion_activacion = 'activation'
    nombre_parametro_alfa = 'alpha'
    nombre_parametro_configuracion_capas_ocultas = 'hidden_layer_sizes'
    nombre_parametro_estado_aleatorio = 'random_state'
    nombre_parametro_max_iteraciones = 'max_iter'
    funciones_activacion = ['tanh', 'relu', 'logistic']
    configuraciones_capas_ocultas = [(3,), (6,), (9,)]
    puntuacion_media_validacion = 'mean_test_score'
    sufijo_parametro = 'param_'
    nombre_parametros = 'params'
    columna_puntuacion = 'Puntuación'
    columnas_num_neuronas_capa_oculta = 'Neuronas capa oculta'
    


    def __init__(self, predictores_entrenar, y_entrenar, predictores_prueba, y_prueba, predictores_entrenamiento_validacion, y_entrenamiento_validacion, indices_entrenamiento_validacion, columna_y, es_clasificacion):
        self.atributos_seleccionados = []
        self.predictores_entrenar = predictores_entrenar
        self.y_entrenar = y_entrenar.values.ravel()
        self.predictores_prueba = predictores_prueba
        self.y_prueba = y_prueba.values.ravel()
        self.predictores_entrenamiento_validacion = predictores_entrenamiento_validacion
        self.y_entrenamiento_validacion = y_entrenamiento_validacion.values.ravel()
        self.indices_entrenamiento_validacion = indices_entrenamiento_validacion
        self.columna_y = columna_y
        self.es_clasificacion = es_clasificacion
        if es_clasificacion:
            self.nombre_puntuacion = self.nombre_precision
            self.nombre_modelo = '<br>(' + self.nombre_modelo_clasificador + self.nombre_modelo_intermedio + columna_y + ')'
        else:
            self.nombre_puntuacion = self.nombre_r2
            self.nombre_modelo = '<br>(' + self.nombre_modelo_regresor + self.nombre_modelo_intermedio + columna_y + ')'


    def crearModelo(self):
        if self.es_clasificacion:
            modelo = MLPClassifier(verbose = False, max_iter = self.MAX_ITERACIONES, random_state = self.ESTADO_ALEATORIO)
        else:
            modelo = MLPRegressor(verbose = False)
        return modelo


    def seleccionarModelo(self, busqueda_hiperparametros):
        configuracion_neuronas_capa_oculta_modelos = busqueda_hiperparametros[self.sufijo_parametro + self.nombre_parametro_configuracion_capas_ocultas].data
        num_neuronas_capa_oculta_modelos = []
        for configuracion_neuronas in configuracion_neuronas_capa_oculta_modelos:
            num_neuronas_capa_oculta_modelos.append(configuracion_neuronas[0])

        puntuaciones_modelos = busqueda_hiperparametros[self.puntuacion_media_validacion]
        indices_modelos = range(0, len(puntuaciones_modelos))
        info_modelos = pd.DataFrame(np.transpose([num_neuronas_capa_oculta_modelos, puntuaciones_modelos]), columns = [self.columnas_num_neuronas_capa_oculta, self.columna_puntuacion], index = indices_modelos)
        info_modelos = info_modelos.sort_values(by = [self.columna_puntuacion], ascending = False)
        modelos_mayor_puntuacion = info_modelos[info_modelos[self.columna_puntuacion] == info_modelos[self.columna_puntuacion].max()]
        modelos_mayor_puntuacion = modelos_mayor_puntuacion.sort_values(by = [self.columnas_num_neuronas_capa_oculta])
        indice_modelo_mayor_puntuacion_menos_neuronas = modelos_mayor_puntuacion.index.values[0]
        parametros_modelo_mayor_puntuacion_menos_neuronas = busqueda_hiperparametros[self.nombre_parametros][indice_modelo_mayor_puntuacion_menos_neuronas]
        
        return parametros_modelo_mayor_puntuacion_menos_neuronas


    def seleccionarHiperparametrosModelo(self):
    
        modelo = self.crearModelo()

        parametros = {
            self.nombre_parametro_funcion_activacion: self.funciones_activacion,
            self.nombre_parametro_alfa: [self.ALFA],
            self.nombre_parametro_configuracion_capas_ocultas: self.configuraciones_capas_ocultas,
            self.nombre_parametro_estado_aleatorio: [self.ESTADO_ALEATORIO],
            self.nombre_parametro_max_iteraciones: [self.MAX_ITERACIONES]
        }

        seleccionador_atributos = SeleccionadorAtributos()
        self.atributos_seleccionados = seleccionador_atributos.seleccionarAtributos(modelo, self.predictores_entrenar, self.y_entrenar, False)
        
        # Búsqueda de hiperparámetros
        busqueda_parametros = GridSearchCV(estimator = modelo, param_grid = parametros, cv = self.indices_entrenamiento_validacion)
        resultado_busqueda = busqueda_parametros.fit(self.predictores_entrenamiento_validacion[self.atributos_seleccionados], self.y_entrenamiento_validacion)
        parametros_modelo_seleccionado = self.seleccionarModelo(resultado_busqueda.cv_results_)

        # Crear el modelo con los hiperparámetros escogidos
        self.modelo = self.crearModelo()
        self.modelo.set_params(**parametros_modelo_seleccionado)
        self.modelo = self.modelo.fit(self.predictores_entrenar[self.atributos_seleccionados], self.y_entrenar)

        # Testear el modelo    
        puntuacion = self.modelo.score(self.predictores_prueba[self.atributos_seleccionados], self.y_prueba)

        self.num_neuronas_capa_oculta = parametros_modelo_seleccionado[self.nombre_parametro_configuracion_capas_ocultas][0]
        self.funcion_activacion = parametros_modelo_seleccionado[self.nombre_parametro_funcion_activacion]
        nombre_modelo = self.nombre_generico_modelos_inicio + str(self.num_neuronas_capa_oculta) + self.nombre_generico_modelos_intermedio + self.funcion_activacion

        return puntuacion, (nombre_modelo, self.modelo)
        

    def importanciaAtributos(self):
        importancia_permutacion = ImportanciaPermutacion()
        importancia_atributos = importancia_permutacion.importanciaAtributos(self.modelo, self.predictores_entrenar, self.y_entrenar, self.atributos_seleccionados)
        titulo = self.titulo_grafica_importancia_atributos + self.nombre_puntuacion + self.nombre_modelo
        figura = importancia_permutacion.graficaImportanciaAtributos(importancia_atributos, titulo)
        return importancia_atributos, figura


    def figuraGrafoRedNeuronal(self):

        # Abrir el archivo con la estructura base del grafo
        with open(self.nombre_fichero_estructura_grafo_red_neuronal) as estructura_grafo:
            grafo_info = estructura_grafo.read()

        # Poner el nombre a los atributos que se utilizan como input
        grafo_info = self.establecerNombreAtributosEntradaGrafo(grafo_info)

        # Establecer los nodos de la capa oculta
        grafo_info = self.establecerNodosCapaOcultaGrafo(grafo_info)

        # Poner el título a la figura
        titulo_grafo = self.titulo_grafo_base + self.nombre_modelo + self.titulo_grafo_intermedio + self.funcion_activacion   
        titulo_grafo = titulo_grafo.replace('<br>', '')

        grafo_info = grafo_info.replace(self.etiqueta_titulo_estructura_grafo, titulo_grafo)

        # Guardar el archivo .dot
        fichero_info_grafo = open(titulo_grafo, 'w')
        fichero_info_grafo.write(grafo_info)
        fichero_info_grafo.close()

        # Convertir el archivo .dot a .png
        render(self.formato_dot, self.formato_png, titulo_grafo)

        return titulo_grafo + '.' + self.formato_png


    def establecerNombreAtributosEntradaGrafo(self, grafo_info):
        for i in range(len(self.atributos_seleccionados)):
            etiqueta_entrada_estructura_grafo = '<x'+str(i+1)+'>'
            # Si el nombre del atributo es muy largo, se muestra en dos líneas
            if len(self.atributos_seleccionados[i]) > 11:
                nombre_atributo = self.atributos_seleccionados[i][0:12] + '<br/>' + self.atributos_seleccionados[i][12:]
                grafo_info = grafo_info.replace(etiqueta_entrada_estructura_grafo, '<'+nombre_atributo+'>')
            else:
                grafo_info = grafo_info.replace(etiqueta_entrada_estructura_grafo, '<'+self.atributos_seleccionados[i]+'>')
        
        return grafo_info


    def establecerNodosCapaOcultaGrafo(self, grafo_info):
        if self.num_neuronas_capa_oculta > 3:
            # Aumentar el número de neuronas en la capa intermedia del grafo
            etiquetas_nodos_capa_oculta = ''
            aristas_nodos_capa_oculta = ''
            orden_nodos_capa_oculta = ''
            for i in range(4, self.num_neuronas_capa_oculta+1):
                texto_num_nodos = 'n' + str(i) + '2'
                etiquetas_nodos_capa_oculta += ' ' + texto_num_nodos + self.formato_base_etiqueta_neurona_capa_oculta_inicio + texto_num_nodos[1:] + self.formato_base_etiqueta_neurona_capa_oculta_fin
                aristas_nodos_capa_oculta += ' ' + texto_num_nodos + ';'
                orden_nodos_capa_oculta += '->' + texto_num_nodos

            valor_sustituir_nodos_capa_oculta = etiquetas_nodos_capa_oculta
            valor_sustituir_orden_nodos_capa_oculta = orden_nodos_capa_oculta
            valor_sustituir_aristar_nodos_capa_oculta = aristas_nodos_capa_oculta

        else:
            valor_sustituir_nodos_capa_oculta = ''
            valor_sustituir_orden_nodos_capa_oculta = ''
            valor_sustituir_aristar_nodos_capa_oculta = ''
        
        grafo_info = grafo_info.replace(self.patron_etiquetas_nodos_capa_oculta, valor_sustituir_nodos_capa_oculta)
        grafo_info = grafo_info.replace(self.patron_orden_nodos_capa_oculta, valor_sustituir_orden_nodos_capa_oculta)
        grafo_info = grafo_info.replace(self.patron_arista_nodos_capa_oculta, valor_sustituir_aristar_nodos_capa_oculta)
            
        grafo_info = grafo_info.replace(self.patron_anchura_grafo, str(self.num_neuronas_capa_oculta))
        grafo_info = grafo_info.replace(self.etiqueta_salida_estructura_grafo, self.columna_y)

        return grafo_info